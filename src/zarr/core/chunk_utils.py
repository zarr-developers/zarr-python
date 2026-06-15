from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from zarr.abc.codec import SupportsSyncCodec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from zarr.abc.codec import Codec, GetResult
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
    from zarr.core.codec_pipeline import chunk_is_empty

    if chunk_is_empty(chunk_array, chunk_spec):
        return None
    return encode(chunk_array, chunk_spec)


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
    from zarr.core.codec_pipeline import _merge_chunk_array

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
    from zarr.core.codec_pipeline import scatter_chunk

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

        non_sync = [c for c in self.codecs if not isinstance(c, SupportsSyncCodec)]
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

    _cached_key: ArraySpec | None = field(init=False, repr=False, compare=False, default=None)
    _cached_aa_specs: tuple[ArraySpec, ...] | None = field(
        init=False, repr=False, compare=False, default=None
    )
    _cached_ab_spec: ArraySpec | None = field(init=False, repr=False, compare=False, default=None)

    def _resolve_specs(self, chunk_spec: ArraySpec) -> tuple[tuple[ArraySpec, ...], ArraySpec]:
        """Return per-AA-codec input specs and the AB spec for `chunk_spec`.

        The resolved chain depends only on the value of `chunk_spec`, so we cache
        it keyed on `chunk_spec` itself (ArraySpec is a frozen, hashable dataclass
        — value identity). Keying on `id(chunk_spec)` would be unsafe: ids are
        recycled after garbage collection, so a freed spec's id reused by a
        different spec (same shape, different prototype/dtype/config) could yield
        a stale hit. Value identity avoids that entirely.
        """
        from zarr.core.codec_pipeline import resolve_aa_specs

        if not self._aa_codecs:
            return (), chunk_spec
        key = chunk_spec
        if self._cached_key == key:
            assert self._cached_aa_specs is not None
            assert self._cached_ab_spec is not None
            return self._cached_aa_specs, self._cached_ab_spec

        aa_specs_t, spec = resolve_aa_specs(cast("tuple[Codec, ...]", self._aa_codecs), chunk_spec)
        self._cached_key = key
        self._cached_aa_specs = aa_specs_t
        self._cached_ab_spec = spec
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
