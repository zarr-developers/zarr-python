from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any, cast
from warnings import warn

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
    CodecPipeline,
    GetResult,
    SupportsSyncCodec,
)
from zarr.core.common import concurrent_map
from zarr.core.config import config
from zarr.core.indexing import SelectorTuple, is_scalar
from zarr.errors import ZarrUserWarning
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType
    from zarr.core.metadata.v3 import ChunkGridMetadata


_pool: ThreadPoolExecutor | None = None
_pool_size: int = 0
_pool_lock = threading.Lock()


def _resolve_max_workers() -> int:
    """Resolve ``codec_pipeline.max_workers`` config to an effective worker count.

    ``None`` means "auto" → ``os.cpu_count()`` (or 1 if unavailable).
    Values < 1 are clamped to 1 (sequential).

    Notes
    -----
    The default (``None`` → ``cpu_count``) is tuned for large chunks
    (≳ 1 MB encoded) where per-chunk decode + scatter is real work and
    threading helps. For small chunks (≲ 64 KB) the per-task pool
    overhead (≈ 30-50 µs submit + worker handoff) outweighs the work
    and threading slows things down by 1.5-3x. If your workload uses
    many small chunks, set ``codec_pipeline.max_workers=1`` explicitly:

        zarr.config.set({"codec_pipeline.max_workers": 1})

    Approximate breakeven on uncompressed reads: 256-512 KB per chunk.
    Compressed chunks shift the threshold lower because decode is real
    CPU work that benefits from parallelism.
    """
    import os as _os

    cfg = config.get("codec_pipeline.max_workers", default=None)
    if cfg is None:
        return _os.cpu_count() or 1
    return max(1, int(cfg))


def _get_pool(max_workers: int) -> ThreadPoolExecutor:
    """Get or create the module-level thread pool, sized to ``max_workers``.

    The pool grows on demand — if a request arrives for more workers than
    the current pool has, the existing pool is shut down and replaced.
    Shrinking requests reuse the existing larger pool (it just leaves
    workers idle).

    Callers that want sequential execution should not call this — they
    should run the task list inline. ``max_workers`` must be >= 1.
    """
    global _pool, _pool_size
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if _pool is None or _pool_size < max_workers:
        with _pool_lock:
            if _pool is None or _pool_size < max_workers:
                if _pool is not None:
                    _pool.shutdown(wait=False)
                _pool = ThreadPoolExecutor(max_workers=max_workers)
                _pool_size = max_workers
    return _pool


def _unzip2[T, U](iterable: Iterable[tuple[T, U]]) -> tuple[list[T], list[U]]:
    out0: list[T] = []
    out1: list[U] = []
    for item0, item1 in iterable:
        out0.append(item0)
        out1.append(item1)
    return (out0, out1)


def batched[T](iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def resolve_batched(codec: Codec, chunk_specs: Iterable[ArraySpec]) -> Iterable[ArraySpec]:
    return [codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]


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


def _merge_chunk_array(
    existing_chunk_array: NDBuffer | None,
    value: NDBuffer,
    out_selection: SelectorTuple,
    chunk_spec: ArraySpec,
    chunk_selection: SelectorTuple,
    is_complete_chunk: bool,
    drop_axes: tuple[int, ...],
) -> NDBuffer:
    """Merge ``value`` into a full-chunk-shaped NDBuffer at ``chunk_selection``.

    If ``is_complete_chunk`` and ``value`` already covers the full chunk
    shape, ``value`` is returned directly (no copy). Otherwise, a writable
    buffer is materialized — either from ``existing_chunk_array.copy()`` if
    one was read from the store, or freshly allocated and filled with the
    chunk's fill value — and the relevant slice of ``value`` is written into it.
    """
    if (
        is_complete_chunk
        and value.shape == chunk_spec.shape
        # Guard that this is not a partial chunk at the end with is_complete_chunk=True
        and value[out_selection].shape == chunk_spec.shape
    ):
        return value
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


async def _async_read_fallback(
    pipeline: CodecPipeline,
    batch: list[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
    out: NDBuffer,
    drop_axes: tuple[int, ...],
) -> tuple[GetResult, ...]:
    """Async fallback read used when no fast-path is available.

    Fetches every chunk's bytes via ``concurrent_map`` (sized by
    ``async.concurrency``), decodes the batch through ``pipeline.decode``,
    then scatters each decoded chunk into ``out`` at its ``out_selection``.

    Used by both ``BatchedCodecPipeline.read_batch`` (non-partial-decode
    branch) and ``FusedCodecPipeline.read`` (when the store is not a
    ``SupportsGetSync`` / sync transform is unavailable).
    """
    chunk_bytes_batch = await concurrent_map(
        [(byte_getter, array_spec.prototype) for byte_getter, array_spec, *_ in batch],
        lambda byte_getter, prototype: byte_getter.get(prototype),
        config.get("async.concurrency"),
    )
    chunk_array_batch = await pipeline.decode(
        [
            (chunk_bytes, chunk_spec)
            for chunk_bytes, (_, chunk_spec, *_) in zip(chunk_bytes_batch, batch, strict=False)
        ],
    )
    results: list[GetResult] = []
    for chunk_array, (_, chunk_spec, chunk_selection, out_selection, _) in zip(
        chunk_array_batch, batch, strict=False
    ):
        if chunk_array is not None:
            tmp = chunk_array[chunk_selection]
            if drop_axes:
                tmp = tmp.squeeze(axis=drop_axes)
            out[out_selection] = tmp
            results.append(GetResult(status="present"))
        else:
            out[out_selection] = fill_value_or_default(chunk_spec)
            results.append(GetResult(status="missing"))
    return tuple(results)


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

    _cached_key: tuple[tuple[int, ...], int] | None = field(
        init=False, repr=False, compare=False, default=None
    )
    _cached_aa_specs: tuple[ArraySpec, ...] | None = field(
        init=False, repr=False, compare=False, default=None
    )
    _cached_ab_spec: ArraySpec | None = field(init=False, repr=False, compare=False, default=None)

    def _resolve_specs(self, chunk_spec: ArraySpec) -> tuple[tuple[ArraySpec, ...], ArraySpec]:
        """Return per-AA-codec input specs and the AB spec for ``chunk_spec``.

        The codec chain only changes ``shape`` (via TransposeCodec etc.) —
        ``prototype``, ``dtype``, ``fill_value``, and ``config`` are
        invariant. We cache the resolved spec chain keyed on
        ``(chunk_spec.shape, id(chunk_spec))``, and reuse it directly
        when the same ``chunk_spec`` is passed again. For a different
        ``chunk_spec`` with the same shape, we recompute (cheap).
        """
        if not self._aa_codecs:
            return (), chunk_spec
        key = (chunk_spec.shape, id(chunk_spec))
        if self._cached_key == key:
            assert self._cached_aa_specs is not None
            assert self._cached_ab_spec is not None
            return self._cached_aa_specs, self._cached_ab_spec

        aa_specs: list[ArraySpec] = []
        spec = chunk_spec
        for aa_codec in self._aa_codecs:
            aa_specs.append(spec)
            spec = aa_codec.resolve_metadata(spec)  # type: ignore[attr-defined]
        aa_specs_t = tuple(aa_specs)
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


@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    """Default codec pipeline.

    This batched codec pipeline divides the chunk batches into batches of a configurable
    batch size ("mini-batch"). Fetching, decoding, encoding and storing are performed in
    lock step for each mini-batch. Multiple mini-batches are processing concurrently.
    """

    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
    batch_size: int

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return type(self).from_codecs(c.evolve_from_array_spec(array_spec=array_spec) for c in self)

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        array_array_codecs, array_bytes_codec, bytes_bytes_codecs = codecs_from_list(codecs)

        return cls(
            array_array_codecs=array_array_codecs,
            array_bytes_codec=array_bytes_codec,
            bytes_bytes_codecs=bytes_bytes_codecs,
            batch_size=batch_size or config.get("codec_pipeline.batch_size"),
        )

    @property
    def supports_partial_decode(self) -> bool:
        """Determines whether the codec pipeline supports partial decoding.

        Currently, only codec pipelines with a single ArrayBytesCodec that supports
        partial decoding can support partial decoding. This limitation is due to the fact
        that ArrayArrayCodecs can change the slice selection leading to non-contiguous
        slices and BytesBytesCodecs can change the chunk bytes in a way that slice
        selections cannot be attributed to byte ranges anymore which renders partial
        decoding infeasible.

        This limitation may softened in the future."""
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin
        )

    @property
    def supports_partial_encode(self) -> bool:
        """Determines whether the codec pipeline supports partial encoding.

        Currently, only codec pipelines with a single ArrayBytesCodec that supports
        partial encoding can support partial encoding. This limitation is due to the fact
        that ArrayArrayCodecs can change the slice selection leading to non-contiguous
        slices and BytesBytesCodecs can change the chunk bytes in a way that slice
        selections cannot be attributed to byte ranges anymore which renders partial
        encoding infeasible.

        This limitation may softened in the future."""
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin
        )

    def __iter__(self) -> Iterator[Codec]:
        yield from self.array_array_codecs
        yield self.array_bytes_codec
        yield from self.bytes_bytes_codecs

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        for codec in self:
            codec.validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length

    def _codecs_with_resolved_metadata_batched(
        self, chunk_specs: Iterable[ArraySpec]
    ) -> tuple[
        list[tuple[ArrayArrayCodec, list[ArraySpec]]],
        tuple[ArrayBytesCodec, list[ArraySpec]],
        list[tuple[BytesBytesCodec, list[ArraySpec]]],
    ]:
        aa_codecs_with_spec: list[tuple[ArrayArrayCodec, list[ArraySpec]]] = []
        chunk_specs = list(chunk_specs)
        for aa_codec in self.array_array_codecs:
            aa_codecs_with_spec.append((aa_codec, chunk_specs))
            chunk_specs = [aa_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]

        ab_codec_with_spec = (self.array_bytes_codec, chunk_specs)
        chunk_specs = [
            self.array_bytes_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs
        ]

        bb_codecs_with_spec: list[tuple[BytesBytesCodec, list[ArraySpec]]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_codecs_with_spec.append((bb_codec, chunk_specs))
            chunk_specs = [bb_codec.resolve_metadata(chunk_spec) for chunk_spec in chunk_specs]

        return (aa_codecs_with_spec, ab_codec_with_spec, bb_codecs_with_spec)

    async def decode_batch(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        chunk_bytes_batch: Iterable[Buffer | None]
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)
        (
            aa_codecs_with_spec,
            ab_codec_with_spec,
            bb_codecs_with_spec,
        ) = self._codecs_with_resolved_metadata_batched(chunk_specs)

        for bb_codec, chunk_spec_batch in bb_codecs_with_spec[::-1]:
            chunk_bytes_batch = await bb_codec.decode(
                zip(chunk_bytes_batch, chunk_spec_batch, strict=False)
            )

        ab_codec, chunk_spec_batch = ab_codec_with_spec
        chunk_array_batch = await ab_codec.decode(
            zip(chunk_bytes_batch, chunk_spec_batch, strict=False)
        )

        for aa_codec, chunk_spec_batch in aa_codecs_with_spec[::-1]:
            chunk_array_batch = await aa_codec.decode(
                zip(chunk_array_batch, chunk_spec_batch, strict=False)
            )

        return chunk_array_batch

    async def decode_partial_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, SelectorTuple, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        assert self.supports_partial_decode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)
        return await self.array_bytes_codec.decode_partial(batch_info)

    async def encode_batch(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        chunk_array_batch: Iterable[NDBuffer | None]
        chunk_specs: Iterable[ArraySpec]
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in self.array_array_codecs:
            chunk_array_batch = await aa_codec.encode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
            chunk_specs = resolve_batched(aa_codec, chunk_specs)

        chunk_bytes_batch = await self.array_bytes_codec.encode(
            zip(chunk_array_batch, chunk_specs, strict=False)
        )
        chunk_specs = resolve_batched(self.array_bytes_codec, chunk_specs)

        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes_batch = await bb_codec.encode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
            chunk_specs = resolve_batched(bb_codec, chunk_specs)

        return chunk_bytes_batch

    async def encode_partial_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, NDBuffer, SelectorTuple, ArraySpec]],
    ) -> None:
        assert self.supports_partial_encode
        assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)
        await self.array_bytes_codec.encode_partial(batch_info)

    async def read_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        if self.supports_partial_decode:
            results: list[GetResult] = []
            batch_info_list = list(batch_info)
            chunk_array_batch = await self.decode_partial_batch(
                [
                    (byte_getter, chunk_selection, chunk_spec)
                    for byte_getter, chunk_spec, chunk_selection, *_ in batch_info_list
                ]
            )
            for chunk_array, (_, chunk_spec, _, out_selection, _) in zip(
                chunk_array_batch, batch_info_list, strict=False
            ):
                if chunk_array is not None:
                    if drop_axes:
                        chunk_array = chunk_array.squeeze(axis=drop_axes)
                    out[out_selection] = chunk_array
                    results.append(GetResult(status="present"))
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)
                    results.append(GetResult(status="missing"))
            return tuple(results)
        return await _async_read_fallback(self, list(batch_info), out, drop_axes)

    async def write_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        if self.supports_partial_encode:
            # Pass scalar values as is
            if len(value.shape) == 0:
                await self.encode_partial_batch(
                    [
                        (byte_setter, value, chunk_selection, chunk_spec)
                        for byte_setter, chunk_spec, chunk_selection, out_selection, _ in batch_info
                    ],
                )
            else:
                await self.encode_partial_batch(
                    [
                        (byte_setter, value[out_selection], chunk_selection, chunk_spec)
                        for byte_setter, chunk_spec, chunk_selection, out_selection, _ in batch_info
                    ],
                )

        else:
            # Read existing bytes if not total slice
            async def _read_key(
                byte_setter: ByteSetter | None, prototype: BufferPrototype
            ) -> Buffer | None:
                if byte_setter is None:
                    return None
                return await byte_setter.get(prototype=prototype)

            chunk_bytes_batch: Iterable[Buffer | None]
            chunk_bytes_batch = await concurrent_map(
                [
                    (
                        None if is_complete_chunk else byte_setter,
                        chunk_spec.prototype,
                    )
                    for byte_setter, chunk_spec, chunk_selection, _, is_complete_chunk in batch_info
                ],
                _read_key,
                config.get("async.concurrency"),
            )
            chunk_array_decoded = await self.decode_batch(
                [
                    (chunk_bytes, chunk_spec)
                    for chunk_bytes, (_, chunk_spec, *_) in zip(
                        chunk_bytes_batch, batch_info, strict=False
                    )
                ],
            )

            chunk_array_merged = [
                _merge_chunk_array(
                    chunk_array,
                    value,
                    out_selection,
                    chunk_spec,
                    chunk_selection,
                    is_complete_chunk,
                    drop_axes,
                )
                for chunk_array, (
                    _,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                ) in zip(chunk_array_decoded, batch_info, strict=False)
            ]
            chunk_array_batch: list[NDBuffer | None] = []
            for chunk_array, (_, chunk_spec, *_) in zip(
                chunk_array_merged, batch_info, strict=False
            ):
                if chunk_array is None:
                    chunk_array_batch.append(None)  # type: ignore[unreachable]
                else:
                    if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                        fill_value_or_default(chunk_spec)
                    ):
                        chunk_array_batch.append(None)
                    else:
                        chunk_array_batch.append(chunk_array)

            chunk_bytes_batch = await self.encode_batch(
                [
                    (chunk_array, chunk_spec)
                    for chunk_array, (_, chunk_spec, *_) in zip(
                        chunk_array_batch, batch_info, strict=False
                    )
                ],
            )

            async def _write_key(byte_setter: ByteSetter, chunk_bytes: Buffer | None) -> None:
                if chunk_bytes is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(chunk_bytes)

            await concurrent_map(
                [
                    (byte_setter, chunk_bytes)
                    for chunk_bytes, (byte_setter, *_) in zip(
                        chunk_bytes_batch, batch_info, strict=False
                    )
                ],
                _write_key,
                config.get("async.concurrency"),
            )

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        output: list[NDBuffer | None] = []
        for batch_info in batched(chunk_bytes_and_specs, self.batch_size):
            output.extend(await self.decode_batch(batch_info))
        return output

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        output: list[Buffer | None] = []
        for single_batch_info in batched(chunk_arrays_and_specs, self.batch_size):
            output.extend(await self.encode_batch(single_batch_info))
        return output

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        batch_results = await concurrent_map(
            [
                (single_batch_info, out, drop_axes)
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.read_batch,
            config.get("async.concurrency"),
        )
        results: list[GetResult] = []
        for batch in batch_results:
            results.extend(batch)
        return tuple(results)

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await concurrent_map(
            [
                (single_batch_info, value, drop_axes)
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.write_batch,
            config.get("async.concurrency"),
        )


def codecs_from_list(
    codecs: Iterable[Codec],
) -> tuple[tuple[ArrayArrayCodec, ...], ArrayBytesCodec, tuple[BytesBytesCodec, ...]]:
    from zarr.codecs.sharding import ShardingCodec

    codecs = tuple(codecs)  # materialize to avoid generator consumption issues

    array_array: tuple[ArrayArrayCodec, ...] = ()
    array_bytes_maybe: ArrayBytesCodec | None = None
    bytes_bytes: tuple[BytesBytesCodec, ...] = ()

    if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(codecs) > 1:
        warn(
            "Combining a `sharding_indexed` codec disables partial reads and "
            "writes, which may lead to inefficient performance.",
            category=ZarrUserWarning,
            stacklevel=3,
        )

    for prev_codec, cur_codec in pairwise((None, *codecs)):
        if isinstance(cur_codec, ArrayArrayCodec):
            if isinstance(prev_codec, ArrayBytesCodec | BytesBytesCodec):
                msg = (
                    f"Invalid codec order. ArrayArrayCodec {cur_codec}"
                    "must be preceded by another ArrayArrayCodec. "
                    f"Got {type(prev_codec)} instead."
                )
                raise TypeError(msg)
            array_array += (cur_codec,)

        elif isinstance(cur_codec, ArrayBytesCodec):
            if isinstance(prev_codec, BytesBytesCodec):
                msg = (
                    f"Invalid codec order. ArrayBytes codec {cur_codec}"
                    f" must be preceded by an ArrayArrayCodec. Got {type(prev_codec)} instead."
                )
                raise TypeError(msg)

            if array_bytes_maybe is not None:
                msg = (
                    f"Got two instances of ArrayBytesCodec: {array_bytes_maybe} and {cur_codec}. "
                    "Only one array-to-bytes codec is allowed."
                )
                raise ValueError(msg)

            array_bytes_maybe = cur_codec

        elif isinstance(cur_codec, BytesBytesCodec):
            if isinstance(prev_codec, ArrayArrayCodec):
                msg = (
                    f"Invalid codec order. BytesBytesCodec {cur_codec}"
                    "must be preceded by either another BytesBytesCodec, or an ArrayBytesCodec. "
                    f"Got {type(prev_codec)} instead."
                )
            bytes_bytes += (cur_codec,)
        else:
            raise TypeError

    if array_bytes_maybe is None:
        raise ValueError("Required ArrayBytesCodec was not found.")
    else:
        return array_array, array_bytes_maybe, bytes_bytes


register_pipeline(BatchedCodecPipeline)


@dataclass(frozen=True)
class FusedCodecPipeline(CodecPipeline):
    """Codec pipeline that uses the codec chain directly.

    Separates IO from compute without an intermediate layout abstraction.
    The ShardingCodec handles shard IO internally via its ``_decode_sync``
    and ``_encode_sync`` methods, so the pipeline simply:

    1. Fetches the raw blob from the store (one key per chunk/shard).
    2. Decodes/encodes through the codec chain (pure compute).
    3. Writes the result back.

    A ``ChunkTransform`` wraps the codec chain for fast synchronous
    decode/encode when all codecs support ``SupportsSyncCodec``.
    """

    codecs: tuple[Codec, ...]
    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
    _sync_transform: ChunkTransform | None
    batch_size: int

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        codec_list = tuple(codecs)
        aa, ab, bb = codecs_from_list(codec_list)

        if batch_size is None:
            batch_size = config.get("codec_pipeline.batch_size")

        return cls(
            codecs=codec_list,
            array_array_codecs=aa,
            array_bytes_codec=ab,
            bytes_bytes_codecs=bb,
            _sync_transform=None,
            batch_size=batch_size,
        )

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        evolved_codecs = tuple(c.evolve_from_array_spec(array_spec=array_spec) for c in self.codecs)
        aa, ab, bb = codecs_from_list(evolved_codecs)

        try:
            sync_transform: ChunkTransform | None = ChunkTransform(codecs=evolved_codecs)
        except TypeError:
            sync_transform = None

        return type(self)(
            codecs=evolved_codecs,
            array_array_codecs=aa,
            array_bytes_codec=ab,
            bytes_bytes_codecs=bb,
            _sync_transform=sync_transform,
            batch_size=self.batch_size,
        )

    def __iter__(self) -> Iterator[Codec]:
        return iter(self.codecs)

    @property
    def supports_partial_decode(self) -> bool:
        return isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)

    @property
    def supports_partial_encode(self) -> bool:
        return isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        for codec in self.codecs:
            codec.validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length

    # -- async decode/encode (required by ABC) --

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        chunk_bytes_batch: Iterable[Buffer | None]
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)

        for bb_codec in self.bytes_bytes_codecs[::-1]:
            chunk_bytes_batch = await bb_codec.decode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
        chunk_array_batch = await self.array_bytes_codec.decode(
            zip(chunk_bytes_batch, chunk_specs, strict=False)
        )
        for aa_codec in self.array_array_codecs[::-1]:
            chunk_array_batch = await aa_codec.decode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
        return chunk_array_batch

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        chunk_array_batch: Iterable[NDBuffer | None]
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in self.array_array_codecs:
            chunk_array_batch = await aa_codec.encode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            )
        chunk_bytes_batch = await self.array_bytes_codec.encode(
            zip(chunk_array_batch, chunk_specs, strict=False)
        )
        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes_batch = await bb_codec.encode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            )
        return chunk_bytes_batch

    # -- sync read/write --

    def read_sync(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
        max_workers: int = 1,
    ) -> tuple[GetResult, ...]:
        """Synchronous read: fetch -> decode -> scatter, per chunk.

        When ``max_workers > 1`` and there are multiple chunks, each
        chunk's full lifecycle (fetch + decode + scatter) runs as one
        task on a thread pool sized to ``max_workers`` — overlapping IO
        of one chunk with decode/scatter of another. Scatter is
        thread-safe because the chunks have non-overlapping output
        selections.

        ``max_workers=1`` runs everything sequentially in the calling
        thread (no pool involvement).

        Mirrors ``BatchedCodecPipeline.read_batch``: when the AB codec
        supports partial decoding (e.g. sharding), the codec handles its
        own IO and only fetches the inner-chunk byte ranges that overlap
        the read selection. Otherwise the pipeline fetches the full
        blob and decodes the whole chunk.
        """
        assert self._sync_transform is not None
        transform = self._sync_transform

        batch = list(batch_info)
        if not batch:
            return ()

        fill = fill_value_or_default(batch[0][1])
        _missing = GetResult(status="missing")

        # Partial-decode fast path: the AB codec owns IO (read only the
        # byte ranges needed for the requested selection). Same condition
        # and dispatch as BatchedCodecPipeline.read_batch.
        if self.supports_partial_decode:
            codec = self.array_bytes_codec
            assert hasattr(codec, "_decode_partial_sync")

            def _read_one_partial(
                item: tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool],
            ) -> GetResult:
                byte_getter, chunk_spec, chunk_selection, out_selection, _ = item
                decoded = codec._decode_partial_sync(byte_getter, chunk_selection, chunk_spec)
                if decoded is None:
                    out[out_selection] = fill
                    return _missing
                if drop_axes:
                    decoded = decoded.squeeze(axis=drop_axes)
                out[out_selection] = decoded
                return GetResult(status="present")

            if max_workers > 1 and len(batch) > 1:
                pool = _get_pool(max_workers)
                return tuple(pool.map(_read_one_partial, batch))
            return tuple(_read_one_partial(item) for item in batch)

        # Per-chunk fused path: fetch + decode + scatter as one task.
        def _read_one(
            item: tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool],
        ) -> GetResult:
            byte_getter, chunk_spec, chunk_selection, out_selection, _ = item
            raw = byte_getter.get_sync(prototype=chunk_spec.prototype)
            if raw is None:
                out[out_selection] = fill
                return _missing
            decoded = transform.decode_chunk(raw, chunk_spec)
            selected = decoded[chunk_selection]
            if drop_axes:
                selected = selected.squeeze(axis=drop_axes)
            out[out_selection] = selected
            return GetResult(status="present")

        if max_workers > 1 and len(batch) > 1:
            pool = _get_pool(max_workers)
            return tuple(pool.map(_read_one, batch))
        return tuple(_read_one(item) for item in batch)

    def write_sync(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
        max_workers: int = 1,
    ) -> None:
        """Synchronous write: fetch existing -> merge+encode -> store.

        When ``max_workers > 1`` and there are multiple chunks, each
        chunk's full lifecycle (get-existing + merge + encode + set/delete)
        runs as one task on a thread pool sized to ``max_workers`` —
        overlapping IO of one chunk with compute of another.

        ``max_workers=1`` runs everything sequentially in the calling
        thread (no pool involvement).

        When the codec pipeline supports partial encoding (e.g. a
        sharding codec with no outer AA/BB codecs), the AB codec handles
        the full write cycle — reading existing data, merging, encoding,
        and writing — matching the async ``BatchedCodecPipeline`` path.
        """
        assert self._sync_transform is not None
        transform = self._sync_transform

        batch = list(batch_info)
        if not batch:
            return

        # Partial-encode path: the AB codec owns IO (read, merge, encode,
        # write).  Same condition and calling convention as
        # BatchedCodecPipeline.write_batch.
        if self.supports_partial_encode:
            codec = self.array_bytes_codec
            assert hasattr(codec, "_encode_partial_sync")
            scalar = len(value.shape) == 0

            def _encode_one_partial(
                item: tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool],
            ) -> None:
                bs, chunk_spec, chunk_selection, out_selection, _is_complete = item
                chunk_value = value if scalar else value[out_selection]
                codec._encode_partial_sync(bs, chunk_value, chunk_selection, chunk_spec)

            if max_workers > 1 and len(batch) > 1:
                pool = _get_pool(max_workers)
                # consume the iterator to surface exceptions
                list(pool.map(_encode_one_partial, batch))
            else:
                for item in batch:
                    _encode_one_partial(item)
            return

        # Per-chunk fused path: get-existing + merge + encode + set/delete as one task.
        def _write_one(
            item: tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool],
        ) -> None:
            bs, chunk_spec, chunk_selection, out_selection, is_complete = item
            existing_bytes: Buffer | None = None
            if not is_complete:
                existing_bytes = bs.get_sync(prototype=chunk_spec.prototype)

            existing_chunk_array: NDBuffer | None = None
            if existing_bytes is not None:
                existing_chunk_array = transform.decode_chunk(existing_bytes, chunk_spec)

            chunk_array = _merge_chunk_array(
                existing_chunk_array,
                value,
                out_selection,
                chunk_spec,
                chunk_selection,
                is_complete,
                drop_axes,
            )

            if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                fill_value_or_default(chunk_spec)
            ):
                bs.delete_sync()
                return

            encoded = transform.encode_chunk(chunk_array, chunk_spec)
            if encoded is None:
                bs.delete_sync()
            else:
                bs.set_sync(encoded)

        if max_workers > 1 and len(batch) > 1:
            pool = _get_pool(max_workers)
            list(pool.map(_write_one, batch))
        else:
            for item in batch:
                _write_one(item)

    # -- async read/write --

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        batch = list(batch_info)
        if not batch:
            return ()

        # Fast path: sync store with sync transform
        from zarr.abc.store import SupportsGetSync
        from zarr.storage._common import StorePath

        first_bg = batch[0][0]
        if (
            self._sync_transform is not None
            and isinstance(first_bg, StorePath)
            and isinstance(first_bg.store, SupportsGetSync)
        ):
            return self.read_sync(batch, out, drop_axes, max_workers=_resolve_max_workers())

        return await _async_read_fallback(self, batch, out, drop_axes)

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch = list(batch_info)
        if not batch:
            return

        # Fast path: sync store with sync transform
        from zarr.abc.store import SupportsSetSync
        from zarr.storage._common import StorePath

        first_bs = batch[0][0]
        if (
            self._sync_transform is not None
            and isinstance(first_bs, StorePath)
            and isinstance(first_bs.store, SupportsSetSync)
        ):
            self.write_sync(batch, value, drop_axes, max_workers=_resolve_max_workers())
            return

        # Async fallback: same pattern as BatchedCodecPipeline.write_batch
        async def _read_key(
            byte_setter: ByteSetter | None, prototype: BufferPrototype
        ) -> Buffer | None:
            if byte_setter is None:
                return None
            return await byte_setter.get(prototype=prototype)

        chunk_bytes_batch: Iterable[Buffer | None]
        chunk_bytes_batch = await concurrent_map(
            [
                (
                    None if is_complete_chunk else byte_setter,
                    chunk_spec.prototype,
                )
                for byte_setter, chunk_spec, chunk_selection, _, is_complete_chunk in batch
            ],
            _read_key,
            config.get("async.concurrency"),
        )
        chunk_array_decoded = await self.decode(
            [
                (chunk_bytes, chunk_spec)
                for chunk_bytes, (_, chunk_spec, *_) in zip(chunk_bytes_batch, batch, strict=False)
            ],
        )

        chunk_array_merged = [
            _merge_chunk_array(
                chunk_array,
                value,
                out_selection,
                chunk_spec,
                chunk_selection,
                is_complete_chunk,
                drop_axes,
            )
            for chunk_array, (
                _,
                chunk_spec,
                chunk_selection,
                out_selection,
                is_complete_chunk,
            ) in zip(chunk_array_decoded, batch, strict=False)
        ]
        chunk_array_batch: list[NDBuffer | None] = []
        for chunk_array, (_, chunk_spec, *_) in zip(chunk_array_merged, batch, strict=False):
            if chunk_array is None:
                chunk_array_batch.append(None)  # type: ignore[unreachable]
            else:
                if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                    fill_value_or_default(chunk_spec)
                ):
                    chunk_array_batch.append(None)
                else:
                    chunk_array_batch.append(chunk_array)

        chunk_bytes_batch = await self.encode(
            [
                (chunk_array, chunk_spec)
                for chunk_array, (_, chunk_spec, *_) in zip(chunk_array_batch, batch, strict=False)
            ],
        )

        async def _write_key(byte_setter: ByteSetter, chunk_bytes: Buffer | None) -> None:
            if chunk_bytes is None:
                await byte_setter.delete()
            else:
                await byte_setter.set(chunk_bytes)

        await concurrent_map(
            [
                (byte_setter, chunk_bytes)
                for chunk_bytes, (byte_setter, *_) in zip(chunk_bytes_batch, batch, strict=False)
            ],
            _write_key,
            config.get("async.concurrency"),
        )


register_pipeline(FusedCodecPipeline)
