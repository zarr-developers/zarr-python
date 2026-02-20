from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice, pairwise
from typing import TYPE_CHECKING, Any, TypeVar, cast
from warnings import warn

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
    CodecPipeline,
    SupportsSyncCodec,
)
from zarr.core.common import concurrent_map, product
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
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

T = TypeVar("T")
U = TypeVar("U")


def _unzip2(iterable: Iterable[tuple[T, U]]) -> tuple[list[T], list[U]]:
    out0: list[T] = []
    out1: list[U] = []
    for item0, item1 in iterable:
        out0.append(item0)
        out1.append(item1)
    return (out0, out1)


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
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


# ---------------------------------------------------------------------------
# Work estimation for thread pool sizing
# ---------------------------------------------------------------------------

# Approximate nanoseconds-per-byte for codec decode and encode, measured on
# typical hardware. These don't need to be exact — they just need to rank
# codecs correctly so the pool-sizing heuristic makes good decisions.
#
# Decode and encode have very different costs for many codecs:
#   - gzip decode ~5-10 ns/byte vs encode ~50-100 ns/byte
#   - zstd decode ~1-2 ns/byte vs encode ~2-10 ns/byte
#   - blosc decode ~0.5-1 ns/byte vs encode ~1-5 ns/byte
#
# "Cheap" codecs (memcpy-like): BytesCodec, Crc32cCodec, TransposeCodec
#   → ~0.1-1 ns/byte, dominated by memcpy; no benefit from threading.
# "Medium" codecs: ZstdCodec, BloscCodec
#   → decode ~1-2 ns/byte, encode ~2-5 ns/byte; GIL released in C.
# "Expensive" codecs: GzipCodec
#   → decode ~5-10 ns/byte, encode ~50-100 ns/byte; GIL released in C.
#
# For unknown codecs (e.g. third-party numcodecs wrappers), we assume
# "medium" cost — better to over-parallelize slightly than miss a win.

_CODEC_DECODE_NS_PER_BYTE: dict[str, float] = {
    # Near-zero cost — just reshaping/copying/checksumming
    "BytesCodec": 0,
    "Crc32cCodec": 0,
    "TransposeCodec": 0,
    "VLenUTF8Codec": 0,
    "VLenBytesCodec": 0,
    # Medium cost — fast C codecs, GIL released
    "ZstdCodec": 1,
    "BloscCodec": 0.5,
    # High cost — slower C codecs, GIL released
    "GzipCodec": 8,
}

_CODEC_ENCODE_NS_PER_BYTE: dict[str, float] = {
    # Near-zero cost — just reshaping/copying/checksumming
    "BytesCodec": 0,
    "Crc32cCodec": 0,
    "TransposeCodec": 0,
    "VLenUTF8Codec": 0,
    "VLenBytesCodec": 0,
    # Medium cost — fast C codecs, GIL released
    "ZstdCodec": 3,
    "BloscCodec": 2,
    # High cost — slower C codecs, GIL released
    "GzipCodec": 50,
}

_DEFAULT_DECODE_NS_PER_BYTE = 1  # assume medium for unknown codecs
_DEFAULT_ENCODE_NS_PER_BYTE = 3  # encode is typically slower

# Thread pool dispatch overhead in nanoseconds (~50-100us per task).
# We only parallelize when the estimated per-chunk work exceeds this.
_POOL_OVERHEAD_NS = 200_000


def _estimate_chunk_work_ns(
    chunk_nbytes: int,
    codecs: Iterable[Codec],
    *,
    is_encode: bool = False,
) -> float:
    """Estimate nanoseconds of codec work for one chunk."""
    table = _CODEC_ENCODE_NS_PER_BYTE if is_encode else _CODEC_DECODE_NS_PER_BYTE
    default = _DEFAULT_ENCODE_NS_PER_BYTE if is_encode else _DEFAULT_DECODE_NS_PER_BYTE
    total_ns_per_byte = 0.0
    for codec in codecs:
        name = type(codec).__name__
        total_ns_per_byte += table.get(name, default)
    return chunk_nbytes * total_ns_per_byte


def _choose_workers(
    n_chunks: int,
    chunk_nbytes: int,
    codecs: Iterable[Codec],
    *,
    is_encode: bool = False,
) -> int:
    """Decide how many thread pool workers to use (0 = don't use pool)."""
    if n_chunks < 2:
        return 0

    per_chunk_ns = _estimate_chunk_work_ns(chunk_nbytes, codecs, is_encode=is_encode)

    if per_chunk_ns < _POOL_OVERHEAD_NS:
        return 0

    total_work_ns = per_chunk_ns * n_chunks
    total_dispatch_ns = n_chunks * 50_000  # ~50us per task
    if total_work_ns < total_dispatch_ns * 3:
        return 0

    target_per_worker_ns = 1_000_000  # 1ms
    workers = max(1, int(total_work_ns / target_per_worker_ns))

    cpu_count = os.cpu_count() or 4
    return min(workers, n_chunks, cpu_count)


def _get_pool(max_workers: int) -> ThreadPoolExecutor:
    """Get a thread pool with at most *max_workers* threads."""
    global _pool
    if _pool is None or _pool._max_workers < max_workers:
        _pool = ThreadPoolExecutor(max_workers=max_workers)
    return _pool


_pool: ThreadPoolExecutor | None = None

# Sentinel to distinguish "delete this key" from None.
_DELETED = object()


@dataclass(frozen=True)
class BatchedCodecPipeline(CodecPipeline):
    """Codec pipeline that automatically selects the optimal execution strategy.

    When all codecs support synchronous operations and the store supports
    sync IO, this pipeline runs the entire read/write path on the calling
    thread with zero async overhead, using a thread pool for parallel codec
    compute on multi-chunk operations.

    When the store requires async IO (e.g. cloud stores), this pipeline uses
    the async path with concurrent IO overlap via ``concurrent_map``.

    This automatic dispatch eliminates the need for users to choose between
    pipeline implementations — the right strategy is selected based on codec
    and store capabilities.
    """

    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
    batch_size: int

    @property
    def _all_sync(self) -> bool:
        """True when every codec in the chain implements SupportsSyncCodec."""
        return all(isinstance(c, SupportsSyncCodec) for c in self)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return type(self).from_codecs(c.evolve_from_array_spec(array_spec=array_spec) for c in self)

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        array_array_codecs, array_bytes_codec, bytes_bytes_codecs = codecs_from_list(list(codecs))

        return cls(
            array_array_codecs=array_array_codecs,
            array_bytes_codec=array_bytes_codec,
            bytes_bytes_codecs=bytes_bytes_codecs,
            batch_size=batch_size or config.get("codec_pipeline.batch_size"),
        )

    @property
    def supports_partial_decode(self) -> bool:
        return (len(self.array_array_codecs) + len(self.bytes_bytes_codecs)) == 0 and isinstance(
            self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin
        )

    @property
    def supports_partial_encode(self) -> bool:
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
        chunk_grid: ChunkGrid,
    ) -> None:
        for codec in self:
            codec.validate(shape=shape, dtype=dtype, chunk_grid=chunk_grid)

    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        for codec in self:
            byte_length = codec.compute_encoded_size(byte_length, array_spec)
            array_spec = codec.resolve_metadata(array_spec)
        return byte_length

    # -------------------------------------------------------------------
    # Per-chunk sync codec chain
    # -------------------------------------------------------------------

    def _resolve_metadata_chain(
        self, chunk_spec: ArraySpec
    ) -> tuple[
        list[tuple[ArrayArrayCodec, ArraySpec]],
        tuple[ArrayBytesCodec, ArraySpec],
        list[tuple[BytesBytesCodec, ArraySpec]],
    ]:
        """Resolve metadata through the codec chain for a single chunk_spec."""
        aa_codecs_with_spec: list[tuple[ArrayArrayCodec, ArraySpec]] = []
        spec = chunk_spec
        for aa_codec in self.array_array_codecs:
            aa_codecs_with_spec.append((aa_codec, spec))
            spec = aa_codec.resolve_metadata(spec)

        ab_codec_with_spec = (self.array_bytes_codec, spec)
        spec = self.array_bytes_codec.resolve_metadata(spec)

        bb_codecs_with_spec: list[tuple[BytesBytesCodec, ArraySpec]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_codecs_with_spec.append((bb_codec, spec))
            spec = bb_codec.resolve_metadata(spec)

        return (aa_codecs_with_spec, ab_codec_with_spec, bb_codecs_with_spec)

    def _decode_one(
        self,
        chunk_bytes: Buffer | None,
        chunk_spec: ArraySpec,
        aa_chain: list[tuple[ArrayArrayCodec, ArraySpec]],
        ab_pair: tuple[ArrayBytesCodec, ArraySpec],
        bb_chain: list[tuple[BytesBytesCodec, ArraySpec]],
    ) -> NDBuffer | None:
        """Decode a single chunk through the full codec chain, synchronously.

        Only called when ``_all_sync`` is True, so every codec implements
        ``SupportsSyncCodec``.
        """
        if chunk_bytes is None:
            return None

        # Use Any to avoid verbose casts on every codec call — we know
        # all codecs satisfy SupportsSyncCodec because _all_sync is True.
        bb_out: Any = chunk_bytes
        for bb_codec, spec in reversed(bb_chain):
            bb_out = cast("SupportsSyncCodec", bb_codec)._decode_sync(bb_out, spec)

        ab_codec, ab_spec = ab_pair
        ab_out: Any = cast("SupportsSyncCodec", ab_codec)._decode_sync(bb_out, ab_spec)

        for aa_codec, spec in reversed(aa_chain):
            ab_out = cast("SupportsSyncCodec", aa_codec)._decode_sync(ab_out, spec)

        return ab_out  # type: ignore[no-any-return]

    def _encode_one(
        self,
        chunk_array: NDBuffer | None,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        """Encode a single chunk through the full codec chain, synchronously.

        Only called when ``_all_sync`` is True, so every codec implements
        ``SupportsSyncCodec``.
        """
        if chunk_array is None:
            return None

        spec = chunk_spec
        aa_out: Any = chunk_array

        for aa_codec in self.array_array_codecs:
            if aa_out is None:
                return None
            aa_out = cast("SupportsSyncCodec", aa_codec)._encode_sync(aa_out, spec)
            spec = aa_codec.resolve_metadata(spec)

        if aa_out is None:
            return None
        bb_out: Any = cast("SupportsSyncCodec", self.array_bytes_codec)._encode_sync(aa_out, spec)
        spec = self.array_bytes_codec.resolve_metadata(spec)

        for bb_codec in self.bytes_bytes_codecs:
            if bb_out is None:
                return None
            bb_out = cast("SupportsSyncCodec", bb_codec)._encode_sync(bb_out, spec)
            spec = bb_codec.resolve_metadata(spec)

        return bb_out  # type: ignore[no-any-return]

    # -------------------------------------------------------------------
    # Batched async decode/encode (layer-by-layer across all chunks)
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Top-level decode / encode
    # -------------------------------------------------------------------

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        items = list(chunk_bytes_and_specs)
        if not items:
            return []

        if self._all_sync:
            # All codecs support sync -- run the full chain inline (no threading).
            _, first_spec = items[0]
            aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(first_spec)
            return [
                self._decode_one(chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain)
                for chunk_bytes, chunk_spec in items
            ]

        # Async fallback: layer-by-layer across all chunks.
        output: list[NDBuffer | None] = []
        for batch_info in batched(items, self.batch_size):
            output.extend(await self.decode_batch(batch_info))
        return output

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        items = list(chunk_arrays_and_specs)
        if not items:
            return []

        if self._all_sync:
            # All codecs support sync -- run the full chain inline (no threading).
            return [self._encode_one(chunk_array, chunk_spec) for chunk_array, chunk_spec in items]

        # Async fallback: layer-by-layer across all chunks.
        output: list[Buffer | None] = []
        for single_batch_info in batched(items, self.batch_size):
            output.extend(await self.encode_batch(single_batch_info))
        return output

    # -------------------------------------------------------------------
    # Async read / write (IO overlap via concurrent_map)
    # -------------------------------------------------------------------

    async def read_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)

        if self.supports_partial_decode:
            chunk_array_batch = await self.decode_partial_batch(
                [
                    (byte_getter, chunk_selection, chunk_spec)
                    for byte_getter, chunk_spec, chunk_selection, *_ in batch_info
                ]
            )
            for chunk_array, (_, chunk_spec, _, out_selection, _) in zip(
                chunk_array_batch, batch_info, strict=False
            ):
                if chunk_array is not None:
                    out[out_selection] = chunk_array
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)
            return

        if self._all_sync:
            # Streaming per-chunk pipeline: each chunk flows through
            # fetch → decode → scatter as a single task. Running N tasks
            # concurrently overlaps IO with codec compute.
            _, first_spec, *_ = batch_info[0]
            aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(first_spec)

            async def _read_chunk(
                byte_getter: ByteGetter,
                chunk_spec: ArraySpec,
                chunk_selection: SelectorTuple,
                out_selection: SelectorTuple,
            ) -> None:
                # 1) Fetch
                chunk_bytes = await byte_getter.get(prototype=chunk_spec.prototype)

                # 2) Decode (full chain, sync)
                chunk_array = self._decode_one(chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain)

                # 3) Scatter
                if chunk_array is not None:
                    tmp = chunk_array[chunk_selection]
                    if drop_axes != ():
                        tmp = tmp.squeeze(axis=drop_axes)
                    out[out_selection] = tmp
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)

            await concurrent_map(
                [
                    (byte_getter, chunk_spec, chunk_selection, out_selection)
                    for byte_getter, chunk_spec, chunk_selection, out_selection, _ in batch_info
                ],
                _read_chunk,
                config.get("async.concurrency"),
            )
        else:
            # Async fallback: fetch all → decode all (async codec API) → scatter.
            # Used for codecs that don't implement _decode_sync (e.g. numcodecs).

            async def _fetch(byte_getter: ByteGetter, prototype: BufferPrototype) -> Buffer | None:
                return await byte_getter.get(prototype=prototype)

            chunk_bytes_batch = await concurrent_map(
                [(byte_getter, chunk_spec.prototype) for byte_getter, chunk_spec, *_ in batch_info],
                _fetch,
                config.get("async.concurrency"),
            )
            chunk_array_batch = await self.decode_batch(
                zip(
                    chunk_bytes_batch,
                    [chunk_spec for _, chunk_spec, *_ in batch_info],
                    strict=False,
                )
            )
            self._scatter(chunk_array_batch, batch_info, out, drop_axes)

    @staticmethod
    def _scatter(
        chunk_array_batch: Iterable[NDBuffer | None],
        batch_info: list[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> None:
        for chunk_array, (_, chunk_spec, chunk_selection, out_selection, _) in zip(
            chunk_array_batch, batch_info, strict=False
        ):
            if chunk_array is not None:
                tmp = chunk_array[chunk_selection]
                if drop_axes != ():
                    tmp = tmp.squeeze(axis=drop_axes)
                out[out_selection] = tmp
            else:
                out[out_selection] = fill_value_or_default(chunk_spec)

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await concurrent_map(
            [
                (single_batch_info, out, drop_axes)
                for single_batch_info in batched(batch_info, self.batch_size)
            ],
            self.read_batch,
            config.get("async.concurrency"),
        )

    def _merge_chunk_array(
        self,
        existing_chunk_array: NDBuffer | None,
        value: NDBuffer,
        out_selection: SelectorTuple,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        is_complete_chunk: bool,
        drop_axes: tuple[int, ...],
    ) -> NDBuffer:
        if (
            is_complete_chunk
            and value.shape == chunk_spec.shape
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
            if drop_axes != ():
                item = tuple(
                    None if idx in drop_axes else slice(None) for idx in range(chunk_spec.ndim)
                )
                chunk_value = chunk_value[item]
        chunk_array[chunk_selection] = chunk_value
        return chunk_array

    async def write_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)

        if self.supports_partial_encode:
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
            return

        if self._all_sync:
            # Streaming per-chunk pipeline: each chunk flows through
            # read_existing → decode → merge → encode → write as a single
            # task. Running N tasks concurrently overlaps IO with compute.
            async def _write_chunk(
                byte_setter: ByteSetter,
                chunk_spec: ArraySpec,
                chunk_selection: SelectorTuple,
                out_selection: SelectorTuple,
                is_complete_chunk: bool,
            ) -> None:
                # 1) Read existing chunk (for partial writes)
                existing_bytes: Buffer | None = None
                if not is_complete_chunk:
                    existing_bytes = await byte_setter.get(prototype=chunk_spec.prototype)

                # 2) Compute: decode existing, merge, encode
                chunk_bytes = self._write_chunk_compute(
                    existing_bytes,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                    value,
                    drop_axes,
                )

                # 3) Write result
                if chunk_bytes is _DELETED or chunk_bytes is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(chunk_bytes)  # type: ignore[arg-type]

            await concurrent_map(
                [
                    (byte_setter, chunk_spec, chunk_selection, out_selection, is_complete_chunk)
                    for byte_setter, chunk_spec, chunk_selection, out_selection, is_complete_chunk in batch_info
                ],
                _write_chunk,
                config.get("async.concurrency"),
            )
        else:
            # Async fallback: phased approach for codecs without sync support.
            # Phase 1: Fetch existing chunks for partial writes.

            async def _fetch_existing(
                byte_setter: ByteSetter, chunk_spec: ArraySpec, is_complete_chunk: bool
            ) -> Buffer | None:
                if is_complete_chunk:
                    return None
                return await byte_setter.get(prototype=chunk_spec.prototype)

            existing_bytes_list: list[Buffer | None] = await concurrent_map(
                [
                    (byte_setter, chunk_spec, is_complete_chunk)
                    for byte_setter, chunk_spec, _, _, is_complete_chunk in batch_info
                ],
                _fetch_existing,
                config.get("async.concurrency"),
            )

            # Phase 2: Decode → merge → encode (async codec API).
            decode_items: list[tuple[Buffer | None, ArraySpec]] = [
                (existing_bytes if not is_complete_chunk else None, chunk_spec)
                for existing_bytes, (_, chunk_spec, _, _, is_complete_chunk) in zip(
                    existing_bytes_list, batch_info, strict=False
                )
            ]
            encoded_list = await self._write_batch_compute(
                decode_items, batch_info, value, drop_axes
            )

            # Phase 3: Write encoded chunks to store.
            async def _write_out(byte_setter: ByteSetter, chunk_bytes: Buffer | None) -> None:
                if chunk_bytes is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(chunk_bytes)

            await concurrent_map(
                [
                    (byte_setter, chunk_bytes)
                    for (byte_setter, *_), chunk_bytes in zip(
                        batch_info, encoded_list, strict=False
                    )
                ],
                _write_out,
                config.get("async.concurrency"),
            )

    async def _write_batch_compute(
        self,
        decode_items: list[tuple[Buffer | None, ArraySpec]],
        batch_info: list[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> list[Buffer | None]:
        chunk_array_decoded: Iterable[NDBuffer | None] = await self.decode(decode_items)

        chunk_array_batch = self._merge_and_filter(
            chunk_array_decoded, batch_info, value, drop_axes
        )

        encoded_batch: Iterable[Buffer | None] = await self.encode(
            [
                (chunk_array, chunk_spec)
                for chunk_array, (_, chunk_spec, *_) in zip(
                    chunk_array_batch, batch_info, strict=False
                )
            ]
        )
        return list(encoded_batch)

    def _merge_and_filter(
        self,
        chunk_array_decoded: Iterable[NDBuffer | None],
        batch_info: list[tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> list[NDBuffer | None]:
        chunk_array_merged = [
            self._merge_chunk_array(
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
        for chunk_array, (_, chunk_spec, *_) in zip(chunk_array_merged, batch_info, strict=False):
            if chunk_array is None:
                chunk_array_batch.append(None)  # type: ignore[unreachable]
            else:
                if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                    fill_value_or_default(chunk_spec)
                ):
                    chunk_array_batch.append(None)
                else:
                    chunk_array_batch.append(chunk_array)
        return chunk_array_batch

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

    # -------------------------------------------------------------------
    # Fully synchronous read / write (no event loop)
    # -------------------------------------------------------------------

    @property
    def supports_sync_io(self) -> bool:
        return self._all_sync

    def read_sync(
        self,
        batch_info: Iterable[tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info_list = list(batch_info)
        if not batch_info_list:
            return

        if self.supports_partial_decode:
            ab_codec: Any = self.array_bytes_codec
            for byte_getter, chunk_spec, chunk_selection, out_selection, _ in batch_info_list:
                chunk_array: NDBuffer | None = ab_codec._decode_partial_sync(
                    byte_getter, chunk_selection, chunk_spec
                )
                if chunk_array is not None:
                    out[out_selection] = chunk_array
                else:
                    out[out_selection] = fill_value_or_default(chunk_spec)
            return

        _, first_spec, *_ = batch_info_list[0]
        aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(first_spec)

        # Phase 1: IO — fetch all chunk bytes sequentially.
        chunk_bytes_list: list[Buffer | None] = [
            byte_getter.get_sync(prototype=chunk_spec.prototype)
            for byte_getter, chunk_spec, *_ in batch_info_list
        ]

        # Phase 2: Decode — run the codec chain for each chunk.
        dtype_item_size = getattr(first_spec.dtype, "item_size", 1)
        chunk_nbytes = product(first_spec.shape) * dtype_item_size
        n_workers = _choose_workers(len(batch_info_list), chunk_nbytes, self)
        if n_workers > 0:
            pool = _get_pool(n_workers)
            chunk_arrays: list[NDBuffer | None] = list(
                pool.map(
                    self._decode_one,
                    chunk_bytes_list,
                    [chunk_spec for _, chunk_spec, *_ in batch_info_list],
                    [aa_chain] * len(batch_info_list),
                    [ab_pair] * len(batch_info_list),
                    [bb_chain] * len(batch_info_list),
                )
            )
        else:
            chunk_arrays = [
                self._decode_one(chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain)
                for chunk_bytes, (_, chunk_spec, *_) in zip(
                    chunk_bytes_list, batch_info_list, strict=False
                )
            ]

        # Phase 3: Scatter decoded chunk data into the output buffer.
        self._scatter(chunk_arrays, batch_info_list, out, drop_axes)

    def _write_chunk_compute(
        self,
        existing_bytes: Buffer | None,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        is_complete_chunk: bool,
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> Buffer | None | object:
        """Per-chunk compute for write: decode existing -> merge -> encode."""
        existing_array: NDBuffer | None = None
        if existing_bytes is not None:
            aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(chunk_spec)
            existing_array = self._decode_one(
                existing_bytes, chunk_spec, aa_chain, ab_pair, bb_chain
            )

        chunk_array: NDBuffer | None = self._merge_chunk_array(
            existing_array,
            value,
            out_selection,
            chunk_spec,
            chunk_selection,
            is_complete_chunk,
            drop_axes,
        )

        if (
            chunk_array is not None
            and not chunk_spec.config.write_empty_chunks
            and chunk_array.all_equal(fill_value_or_default(chunk_spec))
        ):
            chunk_array = None

        if chunk_array is None:
            return _DELETED
        chunk_bytes = self._encode_one(chunk_array, chunk_spec)
        if chunk_bytes is None:
            return _DELETED
        return chunk_bytes

    def write_sync(
        self,
        batch_info: Iterable[tuple[Any, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info_list = list(batch_info)
        if not batch_info_list:
            return

        if self.supports_partial_encode:
            ab_codec: Any = self.array_bytes_codec
            if len(value.shape) == 0:
                for byte_setter, chunk_spec, chunk_selection, _, _ in batch_info_list:
                    ab_codec._encode_partial_sync(byte_setter, value, chunk_selection, chunk_spec)
            else:
                for byte_setter, chunk_spec, chunk_selection, out_selection, _ in batch_info_list:
                    ab_codec._encode_partial_sync(
                        byte_setter, value[out_selection], chunk_selection, chunk_spec
                    )
            return

        # Phase 1: IO — read existing chunk bytes for partial writes.
        existing_bytes_list: list[Buffer | None] = [
            byte_setter.get_sync(prototype=chunk_spec.prototype) if not is_complete_chunk else None
            for byte_setter, chunk_spec, _, _, is_complete_chunk in batch_info_list
        ]

        # Phase 2: Compute — decode existing, merge new data, encode.
        _, first_spec, *_ = batch_info_list[0]
        dtype_item_size = getattr(first_spec.dtype, "item_size", 1)
        chunk_nbytes = product(first_spec.shape) * dtype_item_size
        n_workers = _choose_workers(len(batch_info_list), chunk_nbytes, self, is_encode=True)
        if n_workers > 0:
            pool = _get_pool(n_workers)
            encoded_list: list[Buffer | None | object] = list(
                pool.map(
                    self._write_chunk_compute,
                    existing_bytes_list,
                    [chunk_spec for _, chunk_spec, *_ in batch_info_list],
                    [chunk_selection for _, _, chunk_selection, _, _ in batch_info_list],
                    [out_selection for _, _, _, out_selection, _ in batch_info_list],
                    [is_complete for _, _, _, _, is_complete in batch_info_list],
                    [value] * len(batch_info_list),
                    [drop_axes] * len(batch_info_list),
                )
            )
        else:
            encoded_list = [
                self._write_chunk_compute(
                    existing_bytes,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                    value,
                    drop_axes,
                )
                for existing_bytes, (
                    _,
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                ) in zip(existing_bytes_list, batch_info_list, strict=False)
            ]

        # Phase 3: IO — write encoded chunks to store.
        for encoded, (byte_setter, *_) in zip(encoded_list, batch_info_list, strict=False):
            if encoded is _DELETED:
                byte_setter.delete_sync()
            elif encoded is not None:
                byte_setter.set_sync(encoded)
            else:
                byte_setter.delete_sync()


def codecs_from_list(
    codecs: Iterable[Codec],
) -> tuple[tuple[ArrayArrayCodec, ...], ArrayBytesCodec, tuple[BytesBytesCodec, ...]]:
    from zarr.codecs.sharding import ShardingCodec

    array_array: tuple[ArrayArrayCodec, ...] = ()
    array_bytes_maybe: ArrayBytesCodec | None = None
    bytes_bytes: tuple[BytesBytesCodec, ...] = ()

    if any(isinstance(codec, ShardingCodec) for codec in codecs) and len(tuple(codecs)) > 1:
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
