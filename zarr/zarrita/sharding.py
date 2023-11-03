from __future__ import annotations

from typing import Iterator, List, Mapping, NamedTuple, Optional, Set, Tuple

import numpy as np
from attrs import frozen

from zarrita.codecs import ArrayBytesCodec, CodecPipeline
from zarrita.common import (
    BytesLike,
    ChunkCoords,
    SliceSelection,
    concurrent_map,
    product,
)
from zarrita.indexing import (
    BasicIndexer,
    c_order_iter,
    is_total_slice,
    morton_order_iter,
)
from zarrita.metadata import (
    CoreArrayMetadata,
    DataType,
    ShardingCodecConfigurationMetadata,
    ShardingCodecMetadata,
)
from zarrita.store import StorePath

MAX_UINT_64 = 2**64 - 1


class _ShardIndex(NamedTuple):
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: np.ndarray

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
            return (int(chunk_start), int(chunk_start + chunk_len))

    def set_chunk_slice(
        self, chunk_coords: ChunkCoords, chunk_slice: Optional[slice]
    ) -> None:
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
            set(
                offset
                for offset, _ in sorted_offsets_and_lengths
                if offset != MAX_UINT_64
            )
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
    async def from_bytes(cls, buf: BytesLike, codec: ShardingCodec) -> _ShardProxy:
        obj = cls()
        obj.buf = memoryview(buf)
        obj.index = await codec._decode_shard_index(
            obj.buf[-codec._shard_index_size() :]
        )
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
        self.index.set_chunk_slice(
            chunk_coords, slice(chunk_start, chunk_start + chunk_length)
        )

    def finalize(self, index_bytes: BytesLike) -> BytesLike:
        self.buf.extend(index_bytes)
        return self.buf


@frozen
class ShardingCodec(ArrayBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: ShardingCodecConfigurationMetadata
    codec_pipeline: CodecPipeline
    index_codec_pipeline: CodecPipeline
    chunks_per_shard: Tuple[int, ...]

    @classmethod
    def from_metadata(
        cls,
        codec_metadata: ShardingCodecMetadata,
        array_metadata: CoreArrayMetadata,
    ) -> ShardingCodec:
        chunks_per_shard = tuple(
            s // c
            for s, c in zip(
                array_metadata.chunk_shape,
                codec_metadata.configuration.chunk_shape,
            )
        )
        # rewriting the metadata to scope it to the shard
        shard_metadata = CoreArrayMetadata(
            shape=array_metadata.chunk_shape,
            chunk_shape=codec_metadata.configuration.chunk_shape,
            data_type=array_metadata.data_type,
            fill_value=array_metadata.fill_value,
            runtime_configuration=array_metadata.runtime_configuration,
        )
        codec_pipeline = CodecPipeline.from_metadata(
            codec_metadata.configuration.codecs, shard_metadata
        )
        index_codec_pipeline = CodecPipeline.from_metadata(
            codec_metadata.configuration.index_codecs,
            CoreArrayMetadata(
                shape=chunks_per_shard + (2,),
                chunk_shape=chunks_per_shard + (2,),
                data_type=DataType.uint64,
                fill_value=MAX_UINT_64,
                runtime_configuration=array_metadata.runtime_configuration,
            ),
        )
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
            codec_pipeline=codec_pipeline,
            index_codec_pipeline=index_codec_pipeline,
            chunks_per_shard=chunks_per_shard,
        )

    async def decode(
        self,
        shard_bytes: BytesLike,
    ) -> np.ndarray:
        # print("decode")
        shard_shape = self.array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_shape=chunk_shape,
        )

        # setup output array
        out = np.zeros(
            shard_shape,
            dtype=self.array_metadata.dtype,
            order=self.array_metadata.runtime_configuration.order,
        )
        shard_dict = await _ShardProxy.from_bytes(shard_bytes, self)

        if shard_dict.index.is_all_empty():
            out.fill(self.array_metadata.fill_value)
            return out

        # decoding chunks and writing them into the output buffer
        await concurrent_map(
            [
                (
                    shard_dict,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    out,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._read_chunk,
            self.array_metadata.runtime_configuration.concurrency,
        )

        return out

    async def decode_partial(
        self,
        store_path: StorePath,
        selection: SliceSelection,
    ) -> Optional[np.ndarray]:
        # print("decode_partial")
        shard_shape = self.array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape

        indexer = BasicIndexer(
            selection,
            shape=shard_shape,
            chunk_shape=chunk_shape,
        )

        # setup output array
        out = np.zeros(
            indexer.shape,
            dtype=self.array_metadata.dtype,
            order=self.array_metadata.runtime_configuration.order,
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = set(chunk_coords for chunk_coords, _, _ in indexed_chunks)

        # reading bytes of all requested chunks
        shard_dict: Mapping[ChunkCoords, BytesLike] = {}
        if self._is_total_shard(all_chunk_coords):
            # read entire shard
            shard_dict_maybe = await self._load_full_shard_maybe(store_path)
            if shard_dict_maybe is None:
                return None
            shard_dict = shard_dict_maybe
        else:
            # read some chunks within the shard
            shard_index = await self._load_shard_index_maybe(store_path)
            if shard_index is None:
                return None
            shard_dict = {}
            for chunk_coords in all_chunk_coords:
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice:
                    chunk_bytes = await store_path.get_async(chunk_byte_slice)
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
                    out,
                )
                for chunk_coords, chunk_selection, out_selection in indexed_chunks
            ],
            self._read_chunk,
            self.array_metadata.runtime_configuration.concurrency,
        )

        return out

    async def _read_chunk(
        self,
        shard_dict: Mapping[ChunkCoords, Optional[BytesLike]],
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
        out: np.ndarray,
    ):
        chunk_bytes = shard_dict.get(chunk_coords, None)
        if chunk_bytes is not None:
            chunk_array = await self.codec_pipeline.decode(chunk_bytes)
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp
        else:
            out[out_selection] = self.array_metadata.fill_value

    async def encode(
        self,
        shard_array: np.ndarray,
    ) -> Optional[BytesLike]:
        shard_shape = self.array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape

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
                    dtype=self.array_metadata.dtype,
                )
                chunk_array.fill(self.array_metadata.fill_value)
                chunk_array[chunk_selection] = shard_array[out_selection]
            if not np.array_equiv(chunk_array, self.array_metadata.fill_value):
                return (
                    chunk_coords,
                    await self.codec_pipeline.encode(chunk_array),
                )
            return (chunk_coords, None)

        # assembling and encoding chunks within the shard
        encoded_chunks: List[
            Tuple[ChunkCoords, Optional[BytesLike]]
        ] = await concurrent_map(
            [
                (shard_array, chunk_coords, chunk_selection, out_selection)
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            _write_chunk,
            self.array_metadata.runtime_configuration.concurrency,
        )
        if len(encoded_chunks) == 0:
            return None

        shard_builder = _ShardBuilder.create_empty(self.chunks_per_shard)
        for chunk_coords, chunk_bytes in encoded_chunks:
            if chunk_bytes is not None:
                shard_builder.append(chunk_coords, chunk_bytes)

        return shard_builder.finalize(
            await self._encode_shard_index(shard_builder.index)
        )

    async def encode_partial(
        self,
        store_path: StorePath,
        shard_array: np.ndarray,
        selection: SliceSelection,
    ) -> None:
        # print("encode_partial")
        shard_shape = self.array_metadata.chunk_shape
        chunk_shape = self.configuration.chunk_shape

        old_shard_dict = (
            await self._load_full_shard_maybe(store_path)
        ) or _ShardProxy.create_empty(self.chunks_per_shard)
        new_shard_builder = _ShardBuilder.create_empty(self.chunks_per_shard)
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
                        dtype=self.array_metadata.dtype,
                    )
                    chunk_array.fill(self.array_metadata.fill_value)
                else:
                    chunk_array = (
                        await self.codec_pipeline.decode(chunk_bytes)
                    ).copy()  # make a writable copy
                chunk_array[chunk_selection] = shard_array[out_selection]

            if not np.array_equiv(chunk_array, self.array_metadata.fill_value):
                return (
                    chunk_coords,
                    await self.codec_pipeline.encode(chunk_array),
                )
            else:
                return (chunk_coords, None)

        encoded_chunks: List[
            Tuple[ChunkCoords, Optional[BytesLike]]
        ] = await concurrent_map(
            [
                (
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            _write_chunk,
            self.array_metadata.runtime_configuration.concurrency,
        )

        for chunk_coords, chunk_bytes in encoded_chunks:
            if chunk_bytes is not None:
                new_shard_builder.append(chunk_coords, chunk_bytes)
            else:
                tombstones.add(chunk_coords)

        shard_builder = _ShardBuilder.merge_with_morton_order(
            self.chunks_per_shard, tombstones, new_shard_builder, old_shard_dict
        )

        if shard_builder.index.is_all_empty():
            await store_path.delete_async()
        else:
            await store_path.set_async(
                shard_builder.finalize(
                    await self._encode_shard_index(shard_builder.index)
                )
            )

    def _is_total_shard(self, all_chunk_coords: Set[ChunkCoords]) -> bool:
        return len(all_chunk_coords) == product(self.chunks_per_shard) and all(
            chunk_coords in all_chunk_coords
            for chunk_coords in c_order_iter(self.chunks_per_shard)
        )

    async def _decode_shard_index(self, index_bytes: BytesLike) -> _ShardIndex:
        return _ShardIndex(await self.index_codec_pipeline.decode(index_bytes))

    async def _encode_shard_index(self, index: _ShardIndex) -> BytesLike:
        index_bytes = await self.index_codec_pipeline.encode(index.offsets_and_lengths)
        assert index_bytes is not None
        return index_bytes

    def _shard_index_size(self) -> int:
        return self.index_codec_pipeline.compute_encoded_size(
            16 * product(self.chunks_per_shard)
        )

    async def _load_shard_index_maybe(
        self, store_path: StorePath
    ) -> Optional[_ShardIndex]:
        index_bytes = await store_path.get_async((-self._shard_index_size(), None))
        if index_bytes is not None:
            return await self._decode_shard_index(index_bytes)
        return None

    async def _load_shard_index(self, store_path: StorePath) -> _ShardIndex:
        return (
            await self._load_shard_index_maybe(store_path)
        ) or _ShardIndex.create_empty(self.chunks_per_shard)

    async def _load_full_shard_maybe(
        self, store_path: StorePath
    ) -> Optional[_ShardProxy]:
        shard_bytes = await store_path.get_async()

        return await _ShardProxy.from_bytes(shard_bytes, self) if shard_bytes else None

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length + self._shard_index_size()
