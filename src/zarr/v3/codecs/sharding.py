from __future__ import annotations
from functools import cached_property, lru_cache

from typing import (
    Awaitable,
    Callable,
    Iterator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
)
from attr import field, frozen

import numpy as np
from zarr.v3.abc.codec import (
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
)

from zarr.v3.codecs import CodecPipeline
from zarr.v3.codecs.registry import get_codec_from_metadata, register_codec
from zarr.v3.common import (
    BytesLike,
    ChunkCoords,
    SliceSelection,
    concurrent_map,
    product,
)
from zarr.v3.indexing import (
    BasicIndexer,
    c_order_iter,
    is_total_slice,
    morton_order_iter,
)
from zarr.v3.metadata import (
    ArrayMetadata,
    ArraySpec,
    DataType,
    CodecMetadata,
    RegularChunkGridMetadata,
    ShardingCodecIndexLocation,
    RuntimeConfiguration,
    runtime_configuration as make_runtime_configuration,
)
from zarr.v3.store import StorePath

MAX_UINT_64 = 2**64 - 1


@frozen
class ShardingCodecConfigurationMetadata:
    chunk_shape: ChunkCoords
    codecs: Tuple[CodecMetadata, ...]
    index_codecs: Tuple[CodecMetadata, ...]
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end


@frozen
class ShardingCodecMetadata:
    configuration: ShardingCodecConfigurationMetadata
    name: Literal["sharding_indexed"] = field(default="sharding_indexed", init=False)


class _ShardIndex(NamedTuple):
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: np.ndarray

    @property
    def chunks_per_shard(self) -> ChunkCoords:
        return self.offsets_and_lengths.shape[0:-1]

    def _localize_chunk(self, chunk_coords: ChunkCoords) -> ChunkCoords:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk_coords, self.offsets_and_lengths.shape)
        )

    def is_all_empty(self) -> bool:
        return bool(np.array_equiv(self.offsets_and_lengths, MAX_UINT_64))

    def get_chunk_slice(self, chunk_coords: ChunkCoords) -> Optional[Tuple[int, int]]:
        localized_chunk = self._localize_chunk(chunk_coords)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return (int(chunk_start), int(chunk_start) + int(chunk_len))

    def set_chunk_slice(self, chunk_coords: ChunkCoords, chunk_slice: Optional[slice]) -> None:
        localized_chunk = self._localize_chunk(chunk_coords)
        if chunk_slice is None:
            self.offsets_and_lengths[localized_chunk] = (MAX_UINT_64, MAX_UINT_64)
        else:
            self.offsets_and_lengths[localized_chunk] = (
                chunk_slice.start,
                chunk_slice.stop - chunk_slice.start,
            )

    def is_dense(self, chunk_byte_length: int) -> bool:
        sorted_offsets_and_lengths = sorted(
            [
                (offset, length)
                for offset, length in self.offsets_and_lengths
                if offset != MAX_UINT_64
            ],
            key=lambda entry: entry[0],
        )

        # Are all non-empty offsets unique?
        if len(
            set(offset for offset, _ in sorted_offsets_and_lengths if offset != MAX_UINT_64)
        ) != len(sorted_offsets_and_lengths):
            return False

        return all(
            offset % chunk_byte_length == 0 and length == chunk_byte_length
            for offset, length in sorted_offsets_and_lengths
        )

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> _ShardIndex:
        offsets_and_lengths = np.zeros(chunks_per_shard + (2,), dtype="<u8", order="C")
        offsets_and_lengths.fill(MAX_UINT_64)
        return cls(offsets_and_lengths)


class _ShardProxy(Mapping):
    index: _ShardIndex
    buf: BytesLike

    @classmethod
    async def from_bytes(
        cls, buf: BytesLike, codec: ShardingCodec, chunks_per_shard: ChunkCoords
    ) -> _ShardProxy:
        shard_index_size = codec._shard_index_size(chunks_per_shard)
        obj = cls()
        obj.buf = memoryview(buf)
        if codec.configuration.index_location == ShardingCodecIndexLocation.start:
            shard_index_bytes = obj.buf[:shard_index_size]
        else:
            shard_index_bytes = obj.buf[-shard_index_size:]

        obj.index = await codec._decode_shard_index(shard_index_bytes, chunks_per_shard)
        return obj

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> _ShardProxy:
        index = _ShardIndex.create_empty(chunks_per_shard)
        obj = cls()
        obj.buf = memoryview(b"")
        obj.index = index
        return obj

    def __getitem__(self, chunk_coords: ChunkCoords) -> Optional[BytesLike]:
        chunk_byte_slice = self.index.get_chunk_slice(chunk_coords)
        if chunk_byte_slice:
            return self.buf[chunk_byte_slice[0] : chunk_byte_slice[1]]
        return None

    def __len__(self) -> int:
        return int(self.index.offsets_and_lengths.size / 2)

    def __iter__(self) -> Iterator[ChunkCoords]:
        return c_order_iter(self.index.offsets_and_lengths.shape[:-1])


class _ShardBuilder(_ShardProxy):
    buf: bytearray
    index: _ShardIndex

    @classmethod
    def merge_with_morton_order(
        cls,
        chunks_per_shard: ChunkCoords,
        tombstones: Set[ChunkCoords],
        *shard_dicts: Mapping[ChunkCoords, BytesLike],
    ) -> _ShardBuilder:
        obj = cls.create_empty(chunks_per_shard)
        for chunk_coords in morton_order_iter(chunks_per_shard):
            if tombstones is not None and chunk_coords in tombstones:
                continue
            for shard_dict in shard_dicts:
                maybe_value = shard_dict.get(chunk_coords, None)
                if maybe_value is not None:
                    obj.append(chunk_coords, maybe_value)
                    break
        return obj

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> _ShardBuilder:
        obj = cls()
        obj.buf = bytearray()
        obj.index = _ShardIndex.create_empty(chunks_per_shard)
        return obj

    def append(self, chunk_coords: ChunkCoords, value: BytesLike):
        chunk_start = len(self.buf)
        chunk_length = len(value)
        self.buf.extend(value)
        self.index.set_chunk_slice(chunk_coords, slice(chunk_start, chunk_start + chunk_length))

    async def finalize(
        self,
        index_location: ShardingCodecIndexLocation,
        index_encoder: Callable[[_ShardIndex], Awaitable[BytesLike]],
    ) -> BytesLike:
        index_bytes = await index_encoder(self.index)
        if index_location == ShardingCodecIndexLocation.start:
            self.index.offsets_and_lengths[..., 0] += len(index_bytes)
            index_bytes = await index_encoder(self.index)  # encode again with corrected offsets
            out_buf = bytearray(index_bytes)
            out_buf.extend(self.buf)
        else:
            out_buf = self.buf
            out_buf.extend(index_bytes)
        return out_buf


@frozen
class ShardingCodec(
    ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin, ArrayBytesCodecPartialEncodeMixin
):
    configuration: ShardingCodecConfigurationMetadata

    @classmethod
    def from_metadata(
        cls,
        codec_metadata: CodecMetadata,
    ) -> ShardingCodec:
        assert isinstance(codec_metadata, ShardingCodecMetadata)
        return cls(configuration=codec_metadata.configuration)

    @classmethod
    def get_metadata_class(cls) -> Type[ShardingCodecMetadata]:
        return ShardingCodecMetadata

    def validate(self, array_metadata: ArrayMetadata) -> None:
        assert len(self.configuration.chunk_shape) == array_metadata.ndim, (
            "The shard's `chunk_shape` and array's `shape` need to have the "
            + "same number of dimensions."
        )
        assert isinstance(
            array_metadata.chunk_grid, RegularChunkGridMetadata
        ), "Sharding is only compatible with regular chunk grids."
        assert all(
            s % c == 0
            for s, c in zip(
                array_metadata.chunk_grid.configuration.chunk_shape,
                self.configuration.chunk_shape,
            )
        ), (
            "The array's `chunk_shape` needs to be divisible by the "
            + "shard's inner `chunk_shape`."
        )

    async def decode(
        self,
        shard_bytes: BytesLike,
        shard_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        # print("decode")
        shard_shape = shard_spec.shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_shape=chunk_shape,
        )

        # setup output array
        out = np.zeros(
            shard_shape,
            dtype=shard_spec.dtype,
            order=runtime_configuration.order,
        )
        shard_dict = await _ShardProxy.from_bytes(shard_bytes, self, chunks_per_shard)

        if shard_dict.index.is_all_empty():
            out.fill(shard_spec.fill_value)
            return out

        # decoding chunks and writing them into the output buffer
        await concurrent_map(
            [
                (
                    shard_dict,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    shard_spec,
                    runtime_configuration,
                    out,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._read_chunk,
            runtime_configuration.concurrency,
        )

        return out

    async def decode_partial(
        self,
        store_path: StorePath,
        selection: SliceSelection,
        shard_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        shard_shape = shard_spec.shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)

        indexer = BasicIndexer(
            selection,
            shape=shard_shape,
            chunk_shape=chunk_shape,
        )

        # setup output array
        out = np.zeros(
            indexer.shape,
            dtype=shard_spec.dtype,
            order=runtime_configuration.order,
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = set(chunk_coords for chunk_coords, _, _ in indexed_chunks)

        # reading bytes of all requested chunks
        shard_dict: Mapping[ChunkCoords, BytesLike] = {}
        if self._is_total_shard(all_chunk_coords, chunks_per_shard):
            # read entire shard
            shard_dict_maybe = await self._load_full_shard_maybe(store_path, chunks_per_shard)
            if shard_dict_maybe is None:
                return None
            shard_dict = shard_dict_maybe
        else:
            # read some chunks within the shard
            shard_index = await self._load_shard_index_maybe(store_path, chunks_per_shard)
            if shard_index is None:
                return None
            shard_dict = {}
            for chunk_coords in all_chunk_coords:
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice:
                    chunk_bytes = await store_path.get(chunk_byte_slice)
                    if chunk_bytes:
                        shard_dict[chunk_coords] = chunk_bytes

        # decoding chunks and writing them into the output buffer
        await concurrent_map(
            [
                (
                    shard_dict,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    shard_spec,
                    runtime_configuration,
                    out,
                )
                for chunk_coords, chunk_selection, out_selection in indexed_chunks
            ],
            self._read_chunk,
            runtime_configuration.concurrency,
        )

        return out

    async def _read_chunk(
        self,
        shard_dict: Mapping[ChunkCoords, Optional[BytesLike]],
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
        shard_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
        out: np.ndarray,
    ):
        chunk_spec = self._get_chunk_spec(shard_spec)
        chunk_bytes = shard_dict.get(chunk_coords, None)
        if chunk_bytes is not None:
            chunk_array = await self._codec_pipeline.decode(
                chunk_bytes, chunk_spec, runtime_configuration
            )
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp
        else:
            out[out_selection] = chunk_spec.fill_value

    async def encode(
        self,
        shard_array: np.ndarray,
        shard_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        shard_shape = shard_spec.shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)

        indexer = list(
            BasicIndexer(
                tuple(slice(0, s) for s in shard_shape),
                shape=shard_shape,
                chunk_shape=chunk_shape,
            )
        )

        async def _write_chunk(
            shard_array: np.ndarray,
            chunk_coords: ChunkCoords,
            chunk_selection: SliceSelection,
            out_selection: SliceSelection,
        ) -> Tuple[ChunkCoords, Optional[BytesLike]]:
            if is_total_slice(chunk_selection, chunk_shape):
                chunk_array = shard_array[out_selection]
            else:
                # handling writing partial chunks
                chunk_array = np.empty(
                    chunk_shape,
                    dtype=shard_spec.dtype,
                )
                chunk_array.fill(shard_spec.fill_value)
                chunk_array[chunk_selection] = shard_array[out_selection]
            if not np.array_equiv(chunk_array, shard_spec.fill_value):
                chunk_spec = self._get_chunk_spec(shard_spec)
                return (
                    chunk_coords,
                    await self._codec_pipeline.encode(
                        chunk_array, chunk_spec, runtime_configuration
                    ),
                )
            return (chunk_coords, None)

        # assembling and encoding chunks within the shard
        encoded_chunks: List[Tuple[ChunkCoords, Optional[BytesLike]]] = await concurrent_map(
            [
                (shard_array, chunk_coords, chunk_selection, out_selection)
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            _write_chunk,
            runtime_configuration.concurrency,
        )
        if len(encoded_chunks) == 0:
            return None

        shard_builder = _ShardBuilder.create_empty(chunks_per_shard)
        for chunk_coords, chunk_bytes in encoded_chunks:
            if chunk_bytes is not None:
                shard_builder.append(chunk_coords, chunk_bytes)

        return await shard_builder.finalize(
            self.configuration.index_location, self._encode_shard_index
        )

    async def encode_partial(
        self,
        store_path: StorePath,
        shard_array: np.ndarray,
        selection: SliceSelection,
        shard_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        # print("encode_partial")
        shard_shape = shard_spec.shape
        chunk_shape = self.configuration.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        old_shard_dict = (
            await self._load_full_shard_maybe(store_path, chunks_per_shard)
        ) or _ShardProxy.create_empty(chunks_per_shard)
        new_shard_builder = _ShardBuilder.create_empty(chunks_per_shard)
        tombstones: Set[ChunkCoords] = set()

        indexer = list(
            BasicIndexer(
                selection,
                shape=shard_shape,
                chunk_shape=chunk_shape,
            )
        )

        async def _write_chunk(
            chunk_coords: ChunkCoords,
            chunk_selection: SliceSelection,
            out_selection: SliceSelection,
        ) -> Tuple[ChunkCoords, Optional[BytesLike]]:
            chunk_array = None
            if is_total_slice(chunk_selection, self.configuration.chunk_shape):
                chunk_array = shard_array[out_selection]
            else:
                # handling writing partial chunks
                # read chunk first
                chunk_bytes = old_shard_dict.get(chunk_coords, None)

                # merge new value
                if chunk_bytes is None:
                    chunk_array = np.empty(
                        self.configuration.chunk_shape,
                        dtype=shard_spec.dtype,
                    )
                    chunk_array.fill(shard_spec.fill_value)
                else:
                    chunk_array = (
                        await self._codec_pipeline.decode(
                            chunk_bytes, chunk_spec, runtime_configuration
                        )
                    ).copy()  # make a writable copy
                chunk_array[chunk_selection] = shard_array[out_selection]

            if not np.array_equiv(chunk_array, shard_spec.fill_value):
                return (
                    chunk_coords,
                    await self._codec_pipeline.encode(
                        chunk_array, chunk_spec, runtime_configuration
                    ),
                )
            else:
                return (chunk_coords, None)

        encoded_chunks: List[Tuple[ChunkCoords, Optional[BytesLike]]] = await concurrent_map(
            [
                (
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            _write_chunk,
            runtime_configuration.concurrency,
        )

        for chunk_coords, chunk_bytes in encoded_chunks:
            if chunk_bytes is not None:
                new_shard_builder.append(chunk_coords, chunk_bytes)
            else:
                tombstones.add(chunk_coords)

        shard_builder = _ShardBuilder.merge_with_morton_order(
            chunks_per_shard,
            tombstones,
            new_shard_builder,
            old_shard_dict,
        )

        if shard_builder.index.is_all_empty():
            await store_path.delete()
        else:
            await store_path.set(
                await shard_builder.finalize(
                    self.configuration.index_location,
                    self._encode_shard_index,
                )
            )

    def _is_total_shard(
        self, all_chunk_coords: Set[ChunkCoords], chunks_per_shard: ChunkCoords
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords for chunk_coords in c_order_iter(chunks_per_shard)
        )

    async def _decode_shard_index(
        self, index_bytes: BytesLike, chunks_per_shard: ChunkCoords
    ) -> _ShardIndex:
        return _ShardIndex(
            await self._index_codec_pipeline.decode(
                index_bytes,
                self._get_index_chunk_spec(chunks_per_shard),
                make_runtime_configuration("C"),
            )
        )

    async def _encode_shard_index(self, index: _ShardIndex) -> BytesLike:
        index_bytes = await self._index_codec_pipeline.encode(
            index.offsets_and_lengths,
            self._get_index_chunk_spec(index.chunks_per_shard),
            make_runtime_configuration("C"),
        )
        assert index_bytes is not None
        return index_bytes

    def _shard_index_size(self, chunks_per_shard: ChunkCoords) -> int:
        return self._index_codec_pipeline.compute_encoded_size(
            16 * product(chunks_per_shard), self._get_index_chunk_spec(chunks_per_shard)
        )

    @lru_cache
    def _get_index_chunk_spec(self, chunks_per_shard: ChunkCoords) -> ArraySpec:
        return ArraySpec(
            shape=chunks_per_shard + (2,),
            data_type=DataType.uint64,
            fill_value=MAX_UINT_64,
        )

    @lru_cache
    def _get_chunk_spec(self, shard_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            shape=self.configuration.chunk_shape,
            data_type=shard_spec.data_type,
            fill_value=shard_spec.fill_value,
        )

    @lru_cache
    def _get_chunks_per_shard(self, shard_spec: ArraySpec) -> ChunkCoords:
        return tuple(
            s // c
            for s, c in zip(
                shard_spec.shape,
                self.configuration.chunk_shape,
            )
        )

    @cached_property
    def _index_codec_pipeline(self) -> CodecPipeline:
        return CodecPipeline.create(
            [get_codec_from_metadata(c) for c in self.configuration.index_codecs]
        )

    @cached_property
    def _codec_pipeline(self) -> CodecPipeline:
        return CodecPipeline.create([get_codec_from_metadata(c) for c in self.configuration.codecs])

    async def _load_shard_index_maybe(
        self, store_path: StorePath, chunks_per_shard: ChunkCoords
    ) -> Optional[_ShardIndex]:
        shard_index_size = self._shard_index_size(chunks_per_shard)
        if self.configuration.index_location == ShardingCodecIndexLocation.start:
            index_bytes = await store_path.get((0, shard_index_size))
        else:
            index_bytes = await store_path.get((-shard_index_size, None))
        if index_bytes is not None:
            return await self._decode_shard_index(index_bytes, chunks_per_shard)
        return None

    async def _load_shard_index(
        self, store_path: StorePath, chunks_per_shard: ChunkCoords
    ) -> _ShardIndex:
        return (
            await self._load_shard_index_maybe(store_path, chunks_per_shard)
        ) or _ShardIndex.create_empty(chunks_per_shard)

    async def _load_full_shard_maybe(
        self, store_path: StorePath, chunks_per_shard: ChunkCoords
    ) -> Optional[_ShardProxy]:
        shard_bytes = await store_path.get()

        return (
            await _ShardProxy.from_bytes(shard_bytes, self, chunks_per_shard)
            if shard_bytes
            else None
        )

    def compute_encoded_size(self, input_byte_length: int, shard_spec: ArraySpec) -> int:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        return input_byte_length + self._shard_index_size(chunks_per_shard)


register_codec("sharding_indexed", ShardingCodec)
