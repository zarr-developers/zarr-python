"""Experimental synchronous codec pipeline.

The standard zarr codec pipeline (``BatchedCodecPipeline``) wraps fundamentally
synchronous operations (e.g. gzip compress/decompress) in ``asyncio.to_thread``.
The ``SyncCodecPipeline`` in this module eliminates that overhead by running
per-chunk codec chains synchronously, achieving 2-11x throughput improvements.

Usage::

    import zarr

    zarr.config.set({"codec_pipeline.path": "zarr.experimental.sync_codecs.SyncCodecPipeline"})
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING, TypeVar

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    BytesBytesCodec,
    Codec,
    CodecPipeline,
)
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.codec_pipeline import _unzip2, codecs_from_list, resolve_batched
from zarr.core.common import concurrent_map, product
from zarr.core.config import config
from zarr.core.indexing import SelectorTuple, is_scalar
from zarr.registry import register_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, Self

    from zarr.abc.store import ByteGetter, ByteSetter
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import BufferPrototype
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

__all__ = ["SyncCodecPipeline"]

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def _fill_value_or_default(chunk_spec: ArraySpec) -> Any:
    fill_value = chunk_spec.fill_value
    if fill_value is None:
        return chunk_spec.dtype.default_scalar()
    return fill_value


def _get_pool(max_workers: int) -> ThreadPoolExecutor:
    """Get a thread pool with at most *max_workers* threads.

    Reuses a cached pool when the requested size is <= the cached size.
    CPU-heavy codecs (zstd, gzip, blosc) release the GIL during their C-level
    compress/decompress calls, so real parallelism is achieved across threads.
    """
    global _pool
    if _pool is None or _pool._max_workers < max_workers:
        _pool = ThreadPoolExecutor(max_workers=max_workers)
    return _pool


_pool: ThreadPoolExecutor | None = None

# Sentinel to distinguish "delete this key" from None (which _encode_one
# can return when a chunk encodes to nothing).
_DELETED = object()

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
    """Estimate nanoseconds of codec work for one chunk.

    Sums the per-byte cost of each codec in the chain, multiplied by the
    chunk size. Uses separate decode/encode cost tables since compression
    is typically much more expensive than decompression.

    This is a rough estimate — compression ratios, cache effects,
    and hardware differences mean the actual time can vary 2-5x. But the
    estimate is good enough to decide "use pool" vs "don't use pool".
    """
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
    """Decide how many thread pool workers to use (0 = don't use pool).

    The model:
    1. Estimate per-chunk codec work in nanoseconds (decode or encode).
    2. If per-chunk work < pool dispatch overhead, return 0 (sequential).
       Small chunks with fast codecs aren't worth the pool dispatch cost.
    3. Check that total codec work significantly exceeds total dispatch
       overhead (n_chunks * per-task cost). If not, sequential is faster.
    4. Scale workers with total work, capped at CPU count and chunk count.
    """
    if n_chunks < 2:
        return 0

    per_chunk_ns = _estimate_chunk_work_ns(chunk_nbytes, codecs, is_encode=is_encode)

    if per_chunk_ns < _POOL_OVERHEAD_NS:
        return 0

    # Total codec work must exceed total dispatch overhead by a margin.
    # Each task submitted to pool.map has ~50us dispatch overhead.
    total_work_ns = per_chunk_ns * n_chunks
    total_dispatch_ns = n_chunks * 50_000  # ~50us per task
    if total_work_ns < total_dispatch_ns * 3:
        return 0

    # Scale workers: each worker should do at least 1ms of work to
    # amortize pool overhead.
    target_per_worker_ns = 1_000_000  # 1ms
    workers = max(1, int(total_work_ns / target_per_worker_ns))

    cpu_count = os.cpu_count() or 4
    return min(workers, n_chunks, cpu_count)


# ---------------------------------------------------------------------------
# SyncCodecPipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncCodecPipeline(CodecPipeline):
    """A codec pipeline that runs per-chunk codec chains synchronously.

    When all codecs implement ``_decode_sync`` / ``_encode_sync`` (i.e.
    ``supports_sync`` is ``True``), the per-chunk codec chain runs synchronously
    without any ``asyncio.to_thread`` overhead.

    When a codec does *not* support sync (e.g. ``ShardingCodec``), the pipeline
    falls back to the standard async ``decode`` / ``encode`` path, preserving
    correctness while still benefiting from sync dispatch for the inner pipeline.
    """

    array_array_codecs: tuple[ArrayArrayCodec, ...]
    array_bytes_codec: ArrayBytesCodec
    bytes_bytes_codecs: tuple[BytesBytesCodec, ...]
    batch_size: int

    @property
    def _all_sync(self) -> bool:
        """True when every codec in the chain supports synchronous dispatch."""
        return all(c.supports_sync for c in self)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return type(self).from_codecs(c.evolve_from_array_spec(array_spec=array_spec) for c in self)

    @classmethod
    def from_codecs(cls, codecs: Iterable[Codec], *, batch_size: int | None = None) -> Self:
        array_array, array_bytes, bytes_bytes = codecs_from_list(list(codecs))
        return cls(
            array_array_codecs=array_array,
            array_bytes_codec=array_bytes,
            bytes_bytes_codecs=bytes_bytes,
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
        """Decode a single chunk through the full codec chain, synchronously."""
        if chunk_bytes is None:
            return None

        # bytes-bytes decode (reverse order)
        for bb_codec, spec in reversed(bb_chain):
            chunk_bytes = bb_codec._decode_sync(chunk_bytes, spec)

        # array-bytes decode
        ab_codec, ab_spec = ab_pair
        chunk_array = ab_codec._decode_sync(chunk_bytes, ab_spec)

        # array-array decode (reverse order)
        for aa_codec, spec in reversed(aa_chain):
            chunk_array = aa_codec._decode_sync(chunk_array, spec)

        return chunk_array

    def _encode_one(
        self,
        chunk_array: NDBuffer | None,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        """Encode a single chunk through the full codec chain, synchronously."""
        if chunk_array is None:
            return None

        spec = chunk_spec

        # array-array encode
        for aa_codec in self.array_array_codecs:
            chunk_array = aa_codec._encode_sync(chunk_array, spec)
            spec = aa_codec.resolve_metadata(spec)

        # array-bytes encode
        chunk_bytes = self.array_bytes_codec._encode_sync(chunk_array, spec)
        spec = self.array_bytes_codec.resolve_metadata(spec)

        # bytes-bytes encode
        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes = bb_codec._encode_sync(chunk_bytes, spec)
            spec = bb_codec.resolve_metadata(spec)

        return chunk_bytes

    # -------------------------------------------------------------------
    # Async fallback for codecs that don't support sync (e.g. sharding)
    # -------------------------------------------------------------------

    async def _decode_async(
        self,
        chunk_bytes_and_specs: list[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Async fallback: walk codecs one at a time (like BatchedCodecPipeline).

        Metadata must be resolved forward through the codec chain so each codec
        gets the correct spec during reverse (decode) traversal. This matches
        BatchedCodecPipeline._codecs_with_resolved_metadata_batched.
        """
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)

        # Resolve metadata forward: aa → ab → bb, recording the spec at each step.
        aa_specs: list[list[ArraySpec]] = []
        specs = list(chunk_specs)
        for aa_codec in self.array_array_codecs:
            aa_specs.append(specs)
            specs = [aa_codec.resolve_metadata(s) for s in specs]

        ab_specs = specs
        specs = [self.array_bytes_codec.resolve_metadata(s) for s in specs]

        bb_specs: list[list[ArraySpec]] = []
        for bb_codec in self.bytes_bytes_codecs:
            bb_specs.append(specs)
            specs = [bb_codec.resolve_metadata(s) for s in specs]

        # Decode in reverse, using the forward-resolved specs.
        for bb_codec, bb_spec in zip(
            self.bytes_bytes_codecs[::-1], bb_specs[::-1], strict=False
        ):
            chunk_bytes_batch = list(
                await bb_codec.decode(zip(chunk_bytes_batch, bb_spec, strict=False))
            )

        chunk_array_batch: list[NDBuffer | None] = list(
            await self.array_bytes_codec.decode(
                zip(chunk_bytes_batch, ab_specs, strict=False)
            )
        )

        for aa_codec, aa_spec in zip(
            self.array_array_codecs[::-1], aa_specs[::-1], strict=False
        ):
            chunk_array_batch = list(
                await aa_codec.decode(zip(chunk_array_batch, aa_spec, strict=False))
            )

        return chunk_array_batch

    async def _encode_async(
        self,
        chunk_arrays_and_specs: list[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Async fallback: walk codecs one at a time (like BatchedCodecPipeline)."""
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in self.array_array_codecs:
            chunk_array_batch = list(
                await aa_codec.encode(zip(chunk_array_batch, chunk_specs, strict=False))
            )
            chunk_specs = list(resolve_batched(aa_codec, chunk_specs))

        chunk_bytes_batch: list[Buffer | None] = list(
            await self.array_bytes_codec.encode(zip(chunk_array_batch, chunk_specs, strict=False))
        )
        chunk_specs = list(resolve_batched(self.array_bytes_codec, chunk_specs))

        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes_batch = list(
                await bb_codec.encode(zip(chunk_bytes_batch, chunk_specs, strict=False))
            )
            chunk_specs = list(resolve_batched(bb_codec, chunk_specs))

        return chunk_bytes_batch

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

        if not self._all_sync:
            return await self._decode_async(items)

        # All codecs support sync -- run the full chain inline (no threading).
        _, first_spec = items[0]
        aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(first_spec)

        return [
            self._decode_one(chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain)
            for chunk_bytes, chunk_spec in items
        ]

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        items = list(chunk_arrays_and_specs)
        if not items:
            return []

        if not self._all_sync:
            return await self._encode_async(items)

        # All codecs support sync -- run the full chain inline (no threading).
        return [self._encode_one(chunk_array, chunk_spec) for chunk_array, chunk_spec in items]

    # -------------------------------------------------------------------
    # read / write (IO stays async, compute runs inline)
    # -------------------------------------------------------------------

    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await concurrent_map(
            [
                (single_batch_info, out, drop_axes)
                for single_batch_info in _batched(batch_info, self.batch_size)
            ],
            self._read_batch,
            config.get("async.concurrency"),
        )

    async def _read_batch(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)

        if self.supports_partial_decode:
            assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialDecodeMixin)
            chunk_array_batch = await self.array_bytes_codec.decode_partial(
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
                    out[out_selection] = _fill_value_or_default(chunk_spec)
            return

        # Phase 1: IO -- fetch bytes from store (always async)
        chunk_bytes_batch = await concurrent_map(
            [(byte_getter, array_spec.prototype) for byte_getter, array_spec, *_ in batch_info],
            lambda byte_getter, prototype: byte_getter.get(prototype),
            config.get("async.concurrency"),
        )

        # Phase 2: Compute -- decode + scatter
        decode_items = [
            (chunk_bytes, chunk_spec)
            for chunk_bytes, (_, chunk_spec, *_) in zip(chunk_bytes_batch, batch_info, strict=False)
        ]

        chunk_array_batch_decoded: Iterable[NDBuffer | None] = await self.decode(decode_items)
        self._scatter(chunk_array_batch_decoded, batch_info, out, drop_axes)

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
                out[out_selection] = _fill_value_or_default(chunk_spec)

    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        await concurrent_map(
            [
                (single_batch_info, value, drop_axes)
                for single_batch_info in _batched(batch_info, self.batch_size)
            ],
            self._write_batch,
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
                fill_value=_fill_value_or_default(chunk_spec),
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

    async def _write_batch(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch_info = list(batch_info)

        if self.supports_partial_encode:
            assert isinstance(self.array_bytes_codec, ArrayBytesCodecPartialEncodeMixin)
            if len(value.shape) == 0:
                await self.array_bytes_codec.encode_partial(
                    [
                        (byte_setter, value, chunk_selection, chunk_spec)
                        for byte_setter, chunk_spec, chunk_selection, _, _ in batch_info
                    ],
                )
            else:
                await self.array_bytes_codec.encode_partial(
                    [
                        (byte_setter, value[out_selection], chunk_selection, chunk_spec)
                        for byte_setter, chunk_spec, chunk_selection, out_selection, _ in batch_info
                    ],
                )
            return

        # Phase 1: IO -- read existing bytes for non-complete chunks
        async def _read_key(
            byte_setter: ByteSetter | None, prototype: BufferPrototype
        ) -> Buffer | None:
            if byte_setter is None:
                return None
            return await byte_setter.get(prototype=prototype)

        chunk_bytes_batch: list[Buffer | None]
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

        # Phase 2: Compute -- decode, merge, encode
        decode_items = [
            (chunk_bytes, chunk_spec)
            for chunk_bytes, (_, chunk_spec, *_) in zip(chunk_bytes_batch, batch_info, strict=False)
        ]

        encoded_batch = await self._write_batch_compute(decode_items, batch_info, value, drop_axes)

        # Phase 3: IO -- write to store
        async def _write_key(byte_setter: ByteSetter, chunk_bytes: Buffer | None) -> None:
            if chunk_bytes is None:
                await byte_setter.delete()
            else:
                await byte_setter.set(chunk_bytes)

        await concurrent_map(
            [
                (byte_setter, chunk_bytes)
                for chunk_bytes, (byte_setter, *_) in zip(encoded_batch, batch_info, strict=False)
            ],
            _write_key,
            config.get("async.concurrency"),
        )

    async def _write_batch_compute(
        self,
        decode_items: list[tuple[Buffer | None, ArraySpec]],
        batch_info: list[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> list[Buffer | None]:
        """Async fallback for compute phase of _write_batch."""
        chunk_array_decoded: Iterable[NDBuffer | None] = await self.decode(decode_items)

        chunk_array_batch = self._merge_and_filter(
            chunk_array_decoded, batch_info, value, drop_axes
        )

        encode_items = [
            (chunk_array, chunk_spec)
            for chunk_array, (_, chunk_spec, *_) in zip(chunk_array_batch, batch_info, strict=False)
        ]
        return list(await self.encode(encode_items))

    def _merge_and_filter(
        self,
        chunk_array_decoded: Iterable[NDBuffer | None],
        batch_info: list[tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]],
        value: NDBuffer,
        drop_axes: tuple[int, ...],
    ) -> list[NDBuffer | None]:
        """Merge decoded chunks with new data and filter empty chunks."""
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

        result: list[NDBuffer | None] = []
        for chunk_array, (_, chunk_spec, *_) in zip(chunk_array_merged, batch_info, strict=False):
            if chunk_array is None:
                result.append(None)
            else:
                if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                    _fill_value_or_default(chunk_spec)
                ):
                    result.append(None)
                else:
                    result.append(chunk_array)
        return result

    # -------------------------------------------------------------------
    # Fully synchronous read / write (bypass event loop entirely)
    #
    # These methods implement the same logic as the async read/write
    # methods above, but run entirely on the calling thread:
    #
    #   - Store IO uses byte_getter.get_sync() / byte_setter.set_sync()
    #     instead of the async get()/set() — direct dict lookup for
    #     MemoryStore, direct file IO for LocalStore.
    #
    #   - Codec compute uses _decode_one() / _encode_one(), which call
    #     each codec's _decode_sync/_encode_sync inline (no to_thread).
    #
    #   - When there are multiple chunks, codec compute is parallelized
    #     across a thread pool. CPU-heavy codecs (zstd, gzip, blosc)
    #     release the GIL during C-level compress/decompress, so real
    #     parallelism is achieved. Store IO remains sequential (fast
    #     for local/memory stores).
    #
    # The byte_getter/byte_setter parameters are typed as `Any` because
    # the ByteGetter/ByteSetter protocols only define async methods.
    # At runtime, these are always StorePath instances which have the
    # get_sync/set_sync/delete_sync methods. See docs/design/sync-bypass.md.
    #
    # These methods are only called when supports_sync_io is True (i.e.
    # _all_sync is True), which guarantees every codec in the chain has
    # _decode_sync/_encode_sync implementations.
    # -------------------------------------------------------------------

    @property
    def supports_sync_io(self) -> bool:
        # Enable the fully-sync path when every codec in the chain supports
        # synchronous dispatch. This includes ShardingCodec, which has
        # _decode_sync/_encode_sync (full shard) and _decode_partial_sync/
        # _encode_partial_sync (byte-range reads for partial shard access).
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

        # Partial decode path: when the array_bytes_codec supports partial
        # decode (e.g. ShardingCodec), delegate to its _decode_partial_sync.
        # This handles shard index fetch + per-chunk byte-range reads + inner
        # codec decode, all synchronously.
        if self.supports_partial_decode:
            # The array_bytes_codec is a ShardingCodec (or similar) that has
            # _decode_partial_sync. We use getattr to avoid coupling to the
            # concrete type — the type system can't express this through the
            # ArrayBytesCodecPartialDecodeMixin protocol.
            ab_codec: Any = self.array_bytes_codec
            for byte_getter, chunk_spec, chunk_selection, out_selection, _ in batch_info_list:
                chunk_array: NDBuffer | None = ab_codec._decode_partial_sync(
                    byte_getter, chunk_selection, chunk_spec
                )
                if chunk_array is not None:
                    out[out_selection] = chunk_array
                else:
                    out[out_selection] = _fill_value_or_default(chunk_spec)
            return

        # Non-partial path: standard sync decode through the full codec chain.
        # Resolve the metadata chain once: compute the ArraySpec at each
        # codec boundary. All chunks in a single array share the same codec
        # structure, so this is invariant across the loop.
        _, first_spec, *_ = batch_info_list[0]
        aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(first_spec)

        # Phase 1: IO — fetch all chunk bytes from the store sequentially.
        # For MemoryStore this is a dict lookup (~1us), for LocalStore a
        # file read that benefits from OS page cache. Sequential is fine.
        chunk_bytes_list: list[Buffer | None] = [
            byte_getter.get_sync(prototype=chunk_spec.prototype)
            for byte_getter, chunk_spec, *_ in batch_info_list
        ]

        # Phase 2: Decode — run the codec chain for each chunk.
        # Estimate per-chunk codec work and decide whether to parallelize.
        chunk_nbytes = product(first_spec.shape) * first_spec.dtype.item_size
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
    ) -> Buffer | None | object:  # object is _DELETED sentinel
        """Per-chunk compute for write: decode existing → merge → encode.

        Returns encoded bytes, or _DELETED sentinel if the chunk should
        be removed from the store. Thread-safe: operates only on its own
        chunk data, no shared mutable state.
        """
        # Decode existing chunk (for partial writes)
        existing_array: NDBuffer | None = None
        if existing_bytes is not None:
            aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(chunk_spec)
            existing_array = self._decode_one(
                existing_bytes, chunk_spec, aa_chain, ab_pair, bb_chain
            )

        # Merge new data into the chunk
        chunk_array: NDBuffer | None = self._merge_chunk_array(
            existing_array, value, out_selection, chunk_spec,
            chunk_selection, is_complete_chunk, drop_axes,
        )

        # Filter empty chunks
        if (
            chunk_array is not None
            and not chunk_spec.config.write_empty_chunks
            and chunk_array.all_equal(_fill_value_or_default(chunk_spec))
        ):
            chunk_array = None

        # Encode
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

        # Partial encode path: when the array_bytes_codec supports partial
        # encode (e.g. ShardingCodec), delegate to its _encode_partial_sync.
        # This reads the existing shard, merges new data, encodes and writes
        # back, all synchronously.
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
            byte_setter.get_sync(prototype=chunk_spec.prototype)
            if not is_complete_chunk
            else None
            for byte_setter, chunk_spec, _, _, is_complete_chunk in batch_info_list
        ]

        # Phase 2: Compute — decode existing, merge new data, encode.
        # Estimate per-chunk work to decide whether to parallelize.
        # Use encode cost model since writes are dominated by compression.
        _, first_spec, *_ = batch_info_list[0]
        chunk_nbytes = product(first_spec.shape) * first_spec.dtype.item_size
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
                    existing_bytes, chunk_spec, chunk_selection,
                    out_selection, is_complete_chunk, value, drop_axes,
                )
                for existing_bytes, (
                    _, chunk_spec, chunk_selection, out_selection, is_complete_chunk,
                ) in zip(existing_bytes_list, batch_info_list, strict=False)
            ]

        # Phase 3: IO — write encoded chunks to store.
        # A sentinel _DELETED object distinguishes "delete key" from
        # "no-op" (which doesn't arise here, but keeps the logic clean).
        for encoded, (byte_setter, *_) in zip(encoded_list, batch_info_list, strict=False):
            if encoded is _DELETED:
                byte_setter.delete_sync()
            elif encoded is not None:
                byte_setter.set_sync(encoded)
            else:
                byte_setter.delete_sync()


register_pipeline(SyncCodecPipeline)
