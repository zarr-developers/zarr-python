from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from zarr.abc.codec import GetResult, SupportsSyncCodec, _codec_supports_sync
from zarr.core.indexing import is_scalar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from zarr.abc.codec import Codec
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.indexing import SelectorTuple


def evolve_codecs(codecs: Iterable[Codec], array_spec: ArraySpec) -> tuple[Codec, ...]:
    """Evolve a codec chain against ``array_spec``, threading the spec forward.

    Each codec is evolved against the spec produced by the previous one — NOT
    the original ``array_spec`` — because earlier array->array codecs may
    transform the chunk spec (e.g. ``cast_value`` widening int8 -> int16). A
    later codec (notably the array->bytes serializer) must be evolved against
    the spec it will actually see at run time; evolving every codec against the
    unthreaded original spec would, for example, strip a ``BytesCodec``'s
    ``endian`` (it sees the single-byte source dtype) and then fail at decode
    time on the multi-byte target.

    This is the single source of truth for pipeline-construction-time codec
    evolution, shared by every ``CodecPipeline.evolve_from_array_spec``. (The
    per-chunk decode/encode counterpart is ``resolve_aa_specs``.)
    """
    evolved: list[Codec] = []
    spec = array_spec
    for codec in codecs:
        evolved_codec = codec.evolve_from_array_spec(array_spec=spec)
        evolved.append(evolved_codec)
        spec = evolved_codec.resolve_metadata(spec)
    return tuple(evolved)


def encode_or_elide_chunk(
    chunk_array: NDBuffer,
    chunk_spec: ArraySpec,
    encode: Callable[[NDBuffer, ArraySpec], Buffer | None],
) -> Buffer | None:
    """Encode a merged chunk, normalizing empties to missing.

    Returns the bytes to store, or ``None`` meaning the chunk must NOT be
    stored (either it normalized to empty per `chunk_is_empty`, or the codec
    chain elided it). ``None`` is the single "missing" convention shared by
    the chunk write paths and the shard dicts.
    """

    if chunk_is_empty(chunk_array, chunk_spec):
        return None
    return encode(chunk_array, chunk_spec)


def fill_value_or_default(chunk_spec: ArraySpec) -> Any:
    fill_value = chunk_spec.fill_value
    if fill_value is None:
        # Zarr V2 allowed `fill_value` to be null in the metadata.
        # Zarr V3 requires it to be set. This has already been
        # validated when decoding the metadata, but we support reading
        # Zarr V2 data and need to support the case where fill_value
        # is None.
        return chunk_spec.dtype.default_scalar()
    else:
        return fill_value


def chunk_is_empty(chunk_array: NDBuffer, chunk_spec: ArraySpec) -> bool:
    """THE empty-chunk normalization rule, in one place.

    With ``write_empty_chunks=False`` (the default), a chunk whose decoded
    content equals the fill value normalizes to *missing*: it must not be
    stored, and readers reconstruct it from the fill value. Every write path
    (fused, async fallback, shard inner chunks) must apply this same rule —
    scattering inline ``all_equal`` checks is how the rule drifts.
    """
    return not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
        fill_value_or_default(chunk_spec)
    )


def scatter_chunk(
    selected: NDBuffer | None,
    out: NDBuffer,
    *,
    chunk_spec: ArraySpec,
    out_selection: SelectorTuple,
    drop_axes: tuple[int, ...],
) -> GetResult:
    """Scatter one chunk's (already-selected) decoded region into ``out``.

    ``None`` = the chunk is missing: the fill value is scattered instead and a
    ``missing`` status is returned. POLICY-FREE by design: whether a missing
    chunk is an error (``read_missing_chunks=False``) is decided by the array
    layer from the returned statuses — which is also what makes missing INNER
    chunks of a present shard fill rather than raise (the sharding codec
    discards the nested read's statuses; only top-level statuses reach the
    array layer).
    """
    if selected is None:
        out[out_selection] = fill_value_or_default(chunk_spec)
        return GetResult(status="missing")
    if drop_axes:
        selected = selected.squeeze(axis=drop_axes)
    out[out_selection] = selected
    return GetResult(status="present")


def _merge_chunk_array(
    existing_chunk_array: NDBuffer | None,
    value: NDBuffer,
    out_selection: SelectorTuple,
    chunk_spec: ArraySpec,
    chunk_selection: SelectorTuple,
    is_complete_chunk: bool,
    drop_axes: tuple[int, ...],
) -> NDBuffer:
    """Merge `value` into a full-chunk-shaped NDBuffer at `chunk_selection`.

    If `is_complete_chunk` and `value[out_selection]` is exactly chunk-shaped,
    that VIEW of the caller's `value` is returned without copying — callers
    (and the codecs they pass it to) must treat it as read-only, since
    mutating it would corrupt the user's source array. Otherwise, a writable
    buffer is materialized — either from `existing_chunk_array.copy()` if
    one was read from the store, or freshly allocated and filled with the
    chunk's fill value — and the relevant slice of `value` is written into it.
    """
    if is_complete_chunk and value.shape != ():
        selected = value[out_selection]
        # The shape check guards against a partial edge chunk arriving with
        # is_complete_chunk=True, and against dropped axes (size-1 integer
        # dims), where the selection is not exactly chunk-shaped.
        if selected.shape == chunk_spec.shape:
            return selected
    if existing_chunk_array is None:
        chunk_array = chunk_spec.prototype.nd_buffer.create(
            shape=chunk_spec.shape,
            dtype=chunk_spec.dtype.to_native_dtype(),
            order=chunk_spec.order,
            fill_value=fill_value_or_default(chunk_spec),
        )
    else:
        chunk_array = existing_chunk_array.copy()
    if chunk_selection == () or is_scalar(
        value.as_ndarray_like(), chunk_spec.dtype.to_native_dtype()
    ):
        chunk_value = value
    else:
        chunk_value = value[out_selection]
        if drop_axes:
            item = tuple(
                None if idx in drop_axes else slice(None) for idx in range(chunk_spec.ndim)
            )
            chunk_value = chunk_value[item]
    chunk_array[chunk_selection] = chunk_value
    return chunk_array


def merge_and_encode_chunk(
    existing_bytes: Buffer | None,
    value: NDBuffer,
    *,
    chunk_spec: ArraySpec,
    chunk_selection: SelectorTuple,
    out_selection: SelectorTuple,
    is_complete: bool,
    drop_axes: tuple[int, ...],
    decode: Callable[[Buffer, ArraySpec], NDBuffer],
    encode: Callable[[NDBuffer, ArraySpec], Buffer | None],
) -> Buffer | None:
    """The canonical single-chunk write: merge, normalize, encode.

    decode existing (``None`` = chunk currently missing) -> merge ``value`` at
    ``chunk_selection`` -> normalize empties to missing -> encode. Returns the
    bytes to store or ``None`` = do not store / delete. This is the one
    state-transition every per-chunk write path expresses; only the IO around
    it (where ``existing_bytes`` comes from, where the result goes) differs.
    """

    existing_array = decode(existing_bytes, chunk_spec) if existing_bytes is not None else None
    merged = _merge_chunk_array(
        existing_array, value, out_selection, chunk_spec, chunk_selection, is_complete, drop_axes
    )
    return encode_or_elide_chunk(merged, chunk_spec, encode)


def decode_and_scatter_chunk(
    chunk_bytes: Buffer | None,
    out: NDBuffer,
    *,
    chunk_spec: ArraySpec,
    chunk_selection: SelectorTuple,
    out_selection: SelectorTuple,
    drop_axes: tuple[int, ...],
    decode: Callable[[Buffer, ArraySpec], NDBuffer],
) -> GetResult:
    """The canonical single-chunk read: decode stored bytes (``None`` =
    missing), select, and scatter into ``out`` via `scatter_chunk`. The read
    twin of `merge_and_encode_chunk`.
    """
    if chunk_bytes is None:
        return scatter_chunk(
            None, out, chunk_spec=chunk_spec, out_selection=out_selection, drop_axes=drop_axes
        )
    selected = decode(chunk_bytes, chunk_spec)[chunk_selection]
    return scatter_chunk(
        selected, out, chunk_spec=chunk_spec, out_selection=out_selection, drop_axes=drop_axes
    )


@dataclass(slots=True, kw_only=True)
class ChunkTransform:
    """A synchronous codec chain.

    Provides `encode_chunk` and `decode_chunk` for pure-compute codec
    operations (no IO, no threading, no batching). The `chunk_spec` is
    supplied per call so the same transform can be reused across chunks
    with different shapes, prototypes, etc.

    All codecs must implement `SupportsSyncCodec`. Construction will
    raise `TypeError` if any codec does not.
    """

    codecs: tuple[Codec, ...]

    _aa_codecs: tuple[SupportsSyncCodec[NDBuffer, NDBuffer], ...] = field(
        init=False, repr=False, compare=False
    )
    _ab_codec: SupportsSyncCodec[NDBuffer, Buffer] = field(init=False, repr=False, compare=False)
    _bb_codecs: tuple[SupportsSyncCodec[Buffer, Buffer], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        from zarr.core.codec_pipeline import codecs_from_list

        # _codec_supports_sync, not a bare isinstance check: a codec can satisfy
        # the SupportsSyncCodec protocol structurally yet be unable to run
        # synchronously (ShardingCodec whose inner/index chain contains an
        # async-only codec). Such codecs opt out via `_sync_capable`, and the
        # TypeError here is what makes FusedCodecPipeline.evolve_from_array_spec
        # decline the sync fast path and fall back to the async pipeline.
        non_sync = [c for c in self.codecs if not _codec_supports_sync(c)]
        if non_sync:
            names = ", ".join(type(c).__name__ for c in non_sync)
            raise TypeError(
                f"All codecs must implement SupportsSyncCodec. The following do not: {names}"
            )

        aa, ab, bb = codecs_from_list(list(self.codecs))
        # SupportsSyncCodec was verified above; the cast is purely for mypy.
        self._aa_codecs = cast("tuple[SupportsSyncCodec[NDBuffer, NDBuffer], ...]", tuple(aa))
        self._ab_codec = cast("SupportsSyncCodec[NDBuffer, Buffer]", ab)
        self._bb_codecs = cast("tuple[SupportsSyncCodec[Buffer, Buffer], ...]", tuple(bb))

    # The whole cache entry — (key, aa_specs, ab_spec) — is stored as ONE field
    # and replaced with a single attribute write. A `ChunkTransform` is shared
    # across thread-pool workers (read_sync/write_sync with max_workers > 1), and
    # storing the key separately from the specs would race: a worker could read a
    # freshly-set key while the matching specs were still the previous (or None)
    # value. A single tuple assignment is atomic under the GIL, so a reader sees
    # either the complete old entry or the complete new one — never a torn mix.
    _cache: tuple[ArraySpec, tuple[ArraySpec, ...], ArraySpec] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def _resolve_specs(self, chunk_spec: ArraySpec) -> tuple[tuple[ArraySpec, ...], ArraySpec]:
        """Return per-AA-codec input specs and the AB spec for `chunk_spec`.

        The resolved chain depends only on the value of `chunk_spec`, so we cache
        it keyed on `chunk_spec` itself (ArraySpec is a frozen, hashable dataclass
        — value identity). Keying on `id(chunk_spec)` would be unsafe: ids are
        recycled after garbage collection, so a freed spec's id reused by a
        different spec (same shape, different prototype/dtype/config) could yield
        a stale hit. Value identity avoids that entirely.

        Thread-safety: a benign construction race is possible (two workers with
        different specs may each compute and overwrite the single-entry cache —
        last writer wins), but a torn read is not, because the entry is written
        atomically as one tuple. Worst case is a recompute, never a wrong result.
        """
        from zarr.core.codec_pipeline import resolve_aa_specs

        if not self._aa_codecs:
            return (), chunk_spec
        cache = self._cache
        if cache is not None and cache[0] == chunk_spec:
            return cache[1], cache[2]

        aa_specs_t, spec = resolve_aa_specs(cast("tuple[Codec, ...]", self._aa_codecs), chunk_spec)
        self._cache = (chunk_spec, aa_specs_t, spec)
        return aa_specs_t, spec

    def decode_chunk(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        """Decode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO.

        Parameters
        ----------
        chunk_bytes : Buffer
            The encoded chunk bytes.
        chunk_spec : ArraySpec
            The array spec describing shape, dtype, fill value, and codec
            configuration for this chunk.
        """
        aa_specs, ab_spec = self._resolve_specs(chunk_spec)

        data: Buffer = chunk_bytes
        for bb_codec in reversed(self._bb_codecs):
            data = bb_codec._decode_sync(data, ab_spec)

        chunk_array: NDBuffer = self._ab_codec._decode_sync(data, ab_spec)

        for aa_codec, aa_spec in zip(reversed(self._aa_codecs), reversed(aa_specs), strict=True):
            chunk_array = aa_codec._decode_sync(chunk_array, aa_spec)

        return chunk_array

    def encode_chunk(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
        """Encode a single chunk through the full codec chain, synchronously.

        Pure compute -- no IO.

        Parameters
        ----------
        chunk_array : NDBuffer
            The chunk data to encode.
        chunk_spec : ArraySpec
            The array spec describing shape, dtype, fill value, and codec
            configuration for this chunk.
        """
        aa_specs, ab_spec = self._resolve_specs(chunk_spec)

        aa_data: NDBuffer = chunk_array
        for aa_codec, aa_spec in zip(self._aa_codecs, aa_specs, strict=True):
            aa_result = aa_codec._encode_sync(aa_data, aa_spec)
            if aa_result is None:
                return None
            aa_data = aa_result

        ab_result = self._ab_codec._encode_sync(aa_data, ab_spec)
        if ab_result is None:
            return None

        bb_data: Buffer = ab_result
        for bb_codec in self._bb_codecs:
            bb_result = bb_codec._encode_sync(bb_data, ab_spec)
            if bb_result is None:
                return None
            bb_data = bb_result

        return bb_data

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self.codecs:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length
