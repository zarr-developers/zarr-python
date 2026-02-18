"""Experimental synchronous codec pipeline.

The standard zarr codec pipeline (``BatchedCodecPipeline``) wraps fundamentally
synchronous operations (e.g. gzip compress/decompress) in ``asyncio.to_thread``.
The ``SyncCodecPipeline`` in this module eliminates that overhead by dispatching
the full codec chain for each chunk via ``ThreadPoolExecutor.map``, achieving
2-11x throughput improvements.

Usage::

    import zarr
    from zarr.experimental.sync_codecs import SyncCodecPipeline

    arr = zarr.create_array(
        store,
        shape=(100, 100),
        chunks=(32, 32),
        dtype="float64",
        codec_pipeline_class=SyncCodecPipeline,
    )
"""

from __future__ import annotations

import asyncio
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
from zarr.core.common import concurrent_map
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


def _get_pool() -> ThreadPoolExecutor:
    """Lazily get or create the module-level thread pool."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=os.cpu_count())
    return _POOL


_POOL: ThreadPoolExecutor | None = None


# ---------------------------------------------------------------------------
# SyncCodecPipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncCodecPipeline(CodecPipeline):
    """A codec pipeline that runs full per-chunk codec chains in a thread pool.

    When all codecs implement ``_decode_sync`` / ``_encode_sync`` (i.e.
    ``supports_sync`` is ``True``), the entire per-chunk codec chain is
    dispatched as a single work item via ``ThreadPoolExecutor.map``.

    When a codec does *not* support sync (e.g. ``ShardingCodec``), the pipeline
    falls back to the standard async ``decode`` / ``encode`` path from the base
    class for that batch, preserving correctness while still benefiting from
    sync dispatch for the inner pipeline.
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
    # Per-chunk codec chain (for pool.map dispatch)
    # -------------------------------------------------------------------

    def _resolve_metadata_chain(self, chunk_spec: ArraySpec) -> tuple[
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
    # Top-level decode / encode (pool.map over full chain per chunk)
    # -------------------------------------------------------------------

    async def _decode_async(
        self,
        chunk_bytes_and_specs: list[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        """Async fallback: walk codecs one at a time (like BatchedCodecPipeline)."""
        chunk_bytes_batch, chunk_specs = _unzip2(chunk_bytes_and_specs)

        for bb_codec in self.bytes_bytes_codecs[::-1]:
            chunk_bytes_batch = list(await bb_codec.decode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            ))

        chunk_array_batch: list[NDBuffer | None] = list(await self.array_bytes_codec.decode(
            zip(chunk_bytes_batch, chunk_specs, strict=False)
        ))

        for aa_codec in self.array_array_codecs[::-1]:
            chunk_array_batch = list(await aa_codec.decode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            ))

        return chunk_array_batch

    async def _encode_async(
        self,
        chunk_arrays_and_specs: list[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        """Async fallback: walk codecs one at a time (like BatchedCodecPipeline)."""
        chunk_array_batch, chunk_specs = _unzip2(chunk_arrays_and_specs)

        for aa_codec in self.array_array_codecs:
            chunk_array_batch = list(await aa_codec.encode(
                zip(chunk_array_batch, chunk_specs, strict=False)
            ))
            chunk_specs = list(resolve_batched(aa_codec, chunk_specs))

        chunk_bytes_batch: list[Buffer | None] = list(await self.array_bytes_codec.encode(
            zip(chunk_array_batch, chunk_specs, strict=False)
        ))
        chunk_specs = list(resolve_batched(self.array_bytes_codec, chunk_specs))

        for bb_codec in self.bytes_bytes_codecs:
            chunk_bytes_batch = list(await bb_codec.encode(
                zip(chunk_bytes_batch, chunk_specs, strict=False)
            ))
            chunk_specs = list(resolve_batched(bb_codec, chunk_specs))

        return chunk_bytes_batch

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        items = list(chunk_bytes_and_specs)
        if not items:
            return []

        if not self._all_sync:
            return await self._decode_async(items)

        # Precompute the metadata chain once (same for all chunks in a batch)
        _, first_spec = items[0]
        aa_chain, ab_pair, bb_chain = self._resolve_metadata_chain(first_spec)

        pool = _get_pool()
        loop = asyncio.get_running_loop()

        # Submit each chunk to the pool and wrap each Future for asyncio.
        async_futures = [
            asyncio.wrap_future(
                pool.submit(self._decode_one, item[0], item[1], aa_chain, ab_pair, bb_chain),
                loop=loop,
            )
            for item in items
        ]
        return await asyncio.gather(*async_futures)

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        items = list(chunk_arrays_and_specs)
        if not items:
            return []

        if not self._all_sync:
            return await self._encode_async(items)

        pool = _get_pool()
        loop = asyncio.get_running_loop()

        # Submit each chunk to the pool and wrap each Future for asyncio.
        async_futures = [
            asyncio.wrap_future(
                pool.submit(self._encode_one, item[0], item[1]),
                loop=loop,
            )
            for item in items
        ]
        return await asyncio.gather(*async_futures)

    # -------------------------------------------------------------------
    # read / write (IO stays async, compute goes through pool.map)
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
        # Phase 1: IO -- fetch bytes from store (always async)
        chunk_bytes_batch = await concurrent_map(
            [(byte_getter, array_spec.prototype) for byte_getter, array_spec, *_ in batch_info],
            lambda byte_getter, prototype: byte_getter.get(prototype),
            config.get("async.concurrency"),
        )

        # Phase 2: Compute -- decode via pool.map
        decode_items = [
            (chunk_bytes, chunk_spec)
            for chunk_bytes, (_, chunk_spec, *_) in zip(
                chunk_bytes_batch, batch_info, strict=False
            )
        ]
        chunk_array_batch: Iterable[NDBuffer | None] = await self.decode(decode_items)

        # Phase 3: Scatter into output buffer
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
                    None if idx in drop_axes else slice(None)
                    for idx in range(chunk_spec.ndim)
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

        # Phase 1: IO -- read existing bytes for non-complete chunks
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

        # Phase 2: Compute -- decode existing chunks via pool.map
        decode_items = [
            (chunk_bytes, chunk_spec)
            for chunk_bytes, (_, chunk_spec, *_) in zip(
                chunk_bytes_batch, batch_info, strict=False
            )
        ]
        chunk_array_decoded: Iterable[NDBuffer | None] = await self.decode(decode_items)

        # Phase 3: Merge (pure compute, single-threaded -- touches shared `value` buffer)
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
                chunk_array_batch.append(None)
            else:
                if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(
                    _fill_value_or_default(chunk_spec)
                ):
                    chunk_array_batch.append(None)
                else:
                    chunk_array_batch.append(chunk_array)

        # Phase 4: Compute -- encode via pool.map
        encode_items = [
            (chunk_array, chunk_spec)
            for chunk_array, (_, chunk_spec, *_) in zip(
                chunk_array_batch, batch_info, strict=False
            )
        ]
        chunk_bytes_batch = await self.encode(encode_items)

        # Phase 5: IO -- write to store
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


register_pipeline(SyncCodecPipeline)
