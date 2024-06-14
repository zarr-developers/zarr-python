from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from operator import itemgetter
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import (
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    Codec,
    CodecPipeline,
)
from zarr.abc.store import ByteGetter, ByteSetter
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, BufferPrototype, NDBuffer, default_buffer_prototype
from zarr.chunk_grids import RegularChunkGrid
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.codecs.pipeline import BatchedCodecPipeline
from zarr.codecs.registry import register_codec
from zarr.common import (
    ChunkCoords,
    ChunkCoordsLike,
    parse_enum,
    parse_named_configuration,
    parse_shapelike,
    product,
)
from zarr.indexing import BasicIndexer, SelectorTuple, c_order_iter, get_indexer, morton_order_iter
from zarr.metadata import ArrayMetadata, parse_codecs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from typing_extensions import Self

    from zarr.common import JSON

MAX_UINT_64 = 2**64 - 1
ShardMapping = Mapping[ChunkCoords, Buffer]
ShardMutableMapping = MutableMapping[ChunkCoords, Buffer]


class ShardingCodecIndexLocation(Enum):
    start = "start"
    end = "end"


def parse_index_location(data: JSON) -> ShardingCodecIndexLocation:
    return parse_enum(data, ShardingCodecIndexLocation)


@dataclass(frozen=True)
class _ShardingByteGetter(ByteGetter):
    shard_dict: ShardMapping
    chunk_coords: ChunkCoords

    async def get(
        self, prototype: BufferPrototype, byte_range: tuple[int, int | None] | None = None
    ) -> Buffer | None:
        assert byte_range is None, "byte_range is not supported within shards"
        assert (
            prototype is default_buffer_prototype
        ), "prototype is not supported within shards currently"
        return self.shard_dict.get(self.chunk_coords)


@dataclass(frozen=True)
class _ShardingByteSetter(_ShardingByteGetter, ByteSetter):
    shard_dict: ShardMutableMapping

    async def set(self, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        assert byte_range is None, "byte_range is not supported within shards"
        self.shard_dict[self.chunk_coords] = value

    async def delete(self) -> None:
        del self.shard_dict[self.chunk_coords]


class _ShardIndex(NamedTuple):
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: npt.NDArray[np.uint64]

    @property
    def chunks_per_shard(self) -> ChunkCoords:
        return self.offsets_and_lengths.shape[0:-1]

    def _localize_chunk(self, chunk_coords: ChunkCoords) -> ChunkCoords:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk_coords, self.offsets_and_lengths.shape, strict=False)
        )

    def is_all_empty(self) -> bool:
        return bool(np.array_equiv(self.offsets_and_lengths, MAX_UINT_64))

    def get_full_chunk_map(self) -> npt.NDArray[np.bool_]:
        return np.not_equal(self.offsets_and_lengths[..., 0], MAX_UINT_64)

    def get_chunk_slice(self, chunk_coords: ChunkCoords) -> tuple[int, int] | None:
        localized_chunk = self._localize_chunk(chunk_coords)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return (int(chunk_start), int(chunk_start) + int(chunk_len))

    def set_chunk_slice(self, chunk_coords: ChunkCoords, chunk_slice: slice | None) -> None:
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
            key=itemgetter(0),
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


class _ShardReader(ShardMapping):
    buf: Buffer
    index: _ShardIndex

    @classmethod
    async def from_bytes(
        cls, buf: Buffer, codec: ShardingCodec, chunks_per_shard: ChunkCoords
    ) -> _ShardReader:
        shard_index_size = codec._shard_index_size(chunks_per_shard)
        obj = cls()
        obj.buf = buf
        if codec.index_location == ShardingCodecIndexLocation.start:
            shard_index_bytes = obj.buf[:shard_index_size]
        else:
            shard_index_bytes = obj.buf[-shard_index_size:]

        obj.index = await codec._decode_shard_index(shard_index_bytes, chunks_per_shard)
        return obj

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> _ShardReader:
        index = _ShardIndex.create_empty(chunks_per_shard)
        obj = cls()
        obj.buf = Buffer.create_zero_length()
        obj.index = index
        return obj

    def __getitem__(self, chunk_coords: ChunkCoords) -> Buffer:
        chunk_byte_slice = self.index.get_chunk_slice(chunk_coords)
        if chunk_byte_slice:
            return self.buf[chunk_byte_slice[0] : chunk_byte_slice[1]]
        raise KeyError

    def __len__(self) -> int:
        return int(self.index.offsets_and_lengths.size / 2)

    def __iter__(self) -> Iterator[ChunkCoords]:
        return c_order_iter(self.index.offsets_and_lengths.shape[:-1])

    def is_empty(self) -> bool:
        return self.index.is_all_empty()


class _ShardBuilder(_ShardReader, ShardMutableMapping):
    buf: Buffer
    index: _ShardIndex

    @classmethod
    def merge_with_morton_order(
        cls,
        chunks_per_shard: ChunkCoords,
        tombstones: set[ChunkCoords],
        *shard_dicts: ShardMapping,
    ) -> _ShardBuilder:
        obj = cls.create_empty(chunks_per_shard)
        for chunk_coords in morton_order_iter(chunks_per_shard):
            if chunk_coords in tombstones:
                continue
            for shard_dict in shard_dicts:
                maybe_value = shard_dict.get(chunk_coords, None)
                if maybe_value is not None:
                    obj[chunk_coords] = maybe_value
                    break
        return obj

    @classmethod
    def create_empty(cls, chunks_per_shard: ChunkCoords) -> _ShardBuilder:
        obj = cls()
        obj.buf = Buffer.create_zero_length()
        obj.index = _ShardIndex.create_empty(chunks_per_shard)
        return obj

    def __setitem__(self, chunk_coords: ChunkCoords, value: Buffer) -> None:
        chunk_start = len(self.buf)
        chunk_length = len(value)
        self.buf = self.buf + value
        self.index.set_chunk_slice(chunk_coords, slice(chunk_start, chunk_start + chunk_length))

    def __delitem__(self, chunk_coords: ChunkCoords) -> None:
        raise NotImplementedError

    async def finalize(
        self,
        index_location: ShardingCodecIndexLocation,
        index_encoder: Callable[[_ShardIndex], Awaitable[Buffer]],
    ) -> Buffer:
        index_bytes = await index_encoder(self.index)
        if index_location == ShardingCodecIndexLocation.start:
            self.index.offsets_and_lengths[..., 0] += len(index_bytes)
            index_bytes = await index_encoder(self.index)  # encode again with corrected offsets
            out_buf = index_bytes + self.buf
        else:
            out_buf = self.buf + index_bytes
        return out_buf


@dataclass(frozen=True)
class _MergingShardBuilder(ShardMutableMapping):
    old_dict: _ShardReader
    new_dict: _ShardBuilder
    tombstones: set[ChunkCoords] = field(default_factory=set)

    def __getitem__(self, chunk_coords: ChunkCoords) -> Buffer:
        chunk_bytes_maybe = self.new_dict.get(chunk_coords)
        if chunk_bytes_maybe is not None:
            return chunk_bytes_maybe
        return self.old_dict[chunk_coords]

    def __setitem__(self, chunk_coords: ChunkCoords, value: Buffer) -> None:
        self.new_dict[chunk_coords] = value

    def __delitem__(self, chunk_coords: ChunkCoords) -> None:
        self.tombstones.add(chunk_coords)

    def __len__(self) -> int:
        return self.old_dict.__len__()

    def __iter__(self) -> Iterator[ChunkCoords]:
        return self.old_dict.__iter__()

    def is_empty(self) -> bool:
        full_chunk_coords_map = self.old_dict.index.get_full_chunk_map()
        full_chunk_coords_map = np.logical_or(
            full_chunk_coords_map, self.new_dict.index.get_full_chunk_map()
        )
        for tombstone in self.tombstones:
            full_chunk_coords_map[tombstone] = False
        return bool(np.array_equiv(full_chunk_coords_map, False))

    async def finalize(
        self,
        index_location: ShardingCodecIndexLocation,
        index_encoder: Callable[[_ShardIndex], Awaitable[Buffer]],
    ) -> Buffer:
        shard_builder = _ShardBuilder.merge_with_morton_order(
            self.new_dict.index.chunks_per_shard,
            self.tombstones,
            self.new_dict,
            self.old_dict,
        )
        return await shard_builder.finalize(index_location, index_encoder)


@dataclass(frozen=True)
class ShardingCodec(
    ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin, ArrayBytesCodecPartialEncodeMixin
):
    chunk_shape: ChunkCoords
    codecs: CodecPipeline
    index_codecs: CodecPipeline
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end

    def __init__(
        self,
        *,
        chunk_shape: ChunkCoordsLike,
        codecs: Iterable[Codec | JSON] | None = None,
        index_codecs: Iterable[Codec | JSON] | None = None,
        index_location: ShardingCodecIndexLocation | None = ShardingCodecIndexLocation.end,
    ) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        codecs_parsed = (
            parse_codecs(codecs)
            if codecs is not None
            else BatchedCodecPipeline.from_list([BytesCodec()])
        )
        index_codecs_parsed = (
            parse_codecs(index_codecs)
            if index_codecs is not None
            else BatchedCodecPipeline.from_list([BytesCodec(), Crc32cCodec()])
        )
        index_location_parsed = (
            parse_index_location(index_location)
            if index_location is not None
            else ShardingCodecIndexLocation.end
        )

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "index_codecs", index_codecs_parsed)
        object.__setattr__(self, "index_location", index_location_parsed)

        # Use instance-local lru_cache to avoid memory leaks
        object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "sharding_indexed")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": list(self.chunk_shape),
                "codecs": self.codecs.to_dict(),
                "index_codecs": self.index_codecs.to_dict(),
                "index_location": self.index_location,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        shard_spec = self._get_chunk_spec(array_spec)
        evolved_codecs = self.codecs.evolve_from_array_spec(shard_spec)
        if evolved_codecs != self.codecs:
            return replace(self, codecs=evolved_codecs)
        return self

    def validate(self, array_metadata: ArrayMetadata) -> None:
        if len(self.chunk_shape) != array_metadata.ndim:
            raise ValueError(
                "The shard's `chunk_shape` and array's `shape` need to have the same number of dimensions."
            )
        if not isinstance(array_metadata.chunk_grid, RegularChunkGrid):
            raise ValueError("Sharding is only compatible with regular chunk grids.")
        if not all(
            s % c == 0
            for s, c in zip(
                array_metadata.chunk_grid.chunk_shape,
                self.chunk_shape,
                strict=False,
            )
        ):
            raise ValueError(
                "The array's `chunk_shape` needs to be divisible by the shard's inner `chunk_shape`."
            )

    async def _decode_single(
        self,
        shard_bytes: Buffer,
        shard_spec: ArraySpec,
    ) -> NDBuffer:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
        )

        # setup output array
        out = chunk_spec.prototype.nd_buffer.create(
            shape=shard_shape, dtype=shard_spec.dtype, order=shard_spec.order, fill_value=0
        )
        shard_dict = await _ShardReader.from_bytes(shard_bytes, self, chunks_per_shard)

        if shard_dict.index.is_all_empty():
            out.fill(shard_spec.fill_value)
            return out

        # decoding chunks and writing them into the output buffer
        await self.codecs.read(
            [
                (
                    _ShardingByteGetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            out,
        )

        return out

    async def _decode_partial_single(
        self,
        byte_getter: ByteGetter,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> NDBuffer | None:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = get_indexer(
            selection,
            shape=shard_shape,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
        )

        # setup output array
        out = shard_spec.prototype.nd_buffer.create(
            shape=indexer.shape, dtype=shard_spec.dtype, order=shard_spec.order, fill_value=0
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = set(chunk_coords for chunk_coords, _, _ in indexed_chunks)

        # reading bytes of all requested chunks
        shard_dict: ShardMapping = {}
        if self._is_total_shard(all_chunk_coords, chunks_per_shard):
            # read entire shard
            shard_dict_maybe = await self._load_full_shard_maybe(
                byte_getter=byte_getter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
            if shard_dict_maybe is None:
                return None
            shard_dict = shard_dict_maybe
        else:
            # read some chunks within the shard
            shard_index = await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
            if shard_index is None:
                return None
            shard_dict = {}
            for chunk_coords in all_chunk_coords:
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice:
                    chunk_bytes = await byte_getter.get(
                        prototype=chunk_spec.prototype, byte_range=chunk_byte_slice
                    )
                    if chunk_bytes:
                        shard_dict[chunk_coords] = chunk_bytes

        # decoding chunks and writing them into the output buffer
        await self.codecs.read(
            [
                (
                    _ShardingByteGetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            out,
        )
        return out

    async def _encode_single(
        self,
        shard_array: NDBuffer,
        shard_spec: ArraySpec,
    ) -> Buffer | None:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        indexer = list(
            BasicIndexer(
                tuple(slice(0, s) for s in shard_shape),
                shape=shard_shape,
                chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            )
        )

        shard_builder = _ShardBuilder.create_empty(chunks_per_shard)

        await self.codecs.write(
            [
                (
                    _ShardingByteSetter(shard_builder, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            shard_array,
        )

        return await shard_builder.finalize(self.index_location, self._encode_shard_index)

    async def _encode_partial_single(
        self,
        byte_setter: ByteSetter,
        shard_array: NDBuffer,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> None:
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)

        shard_dict = _MergingShardBuilder(
            await self._load_full_shard_maybe(
                byte_getter=byte_setter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
            or _ShardReader.create_empty(chunks_per_shard),
            _ShardBuilder.create_empty(chunks_per_shard),
        )

        indexer = list(
            get_indexer(
                selection, shape=shard_shape, chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape)
            )
        )

        await self.codecs.write(
            [
                (
                    _ShardingByteSetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            shard_array,
        )

        if shard_dict.is_empty():
            await byte_setter.delete()
        else:
            await byte_setter.set(
                await shard_dict.finalize(
                    self.index_location,
                    self._encode_shard_index,
                )
            )

    def _is_total_shard(
        self, all_chunk_coords: set[ChunkCoords], chunks_per_shard: ChunkCoords
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords for chunk_coords in c_order_iter(chunks_per_shard)
        )

    async def _decode_shard_index(
        self, index_bytes: Buffer, chunks_per_shard: ChunkCoords
    ) -> _ShardIndex:
        index_array = next(
            iter(
                await self.index_codecs.decode(
                    [(index_bytes, self._get_index_chunk_spec(chunks_per_shard))],
                )
            )
        )
        assert index_array is not None
        return _ShardIndex(index_array.as_numpy_array())

    async def _encode_shard_index(self, index: _ShardIndex) -> Buffer:
        index_bytes = next(
            iter(
                await self.index_codecs.encode(
                    [
                        (
                            NDBuffer.from_numpy_array(index.offsets_and_lengths),
                            self._get_index_chunk_spec(index.chunks_per_shard),
                        )
                    ],
                )
            )
        )
        assert index_bytes is not None
        assert isinstance(index_bytes, Buffer)
        return index_bytes

    def _shard_index_size(self, chunks_per_shard: ChunkCoords) -> int:
        return self.index_codecs.compute_encoded_size(
            16 * product(chunks_per_shard), self._get_index_chunk_spec(chunks_per_shard)
        )

    def _get_index_chunk_spec(self, chunks_per_shard: ChunkCoords) -> ArraySpec:
        return ArraySpec(
            shape=chunks_per_shard + (2,),
            dtype=np.dtype("<u8"),
            fill_value=MAX_UINT_64,
            order="C",  # Note: this is hard-coded for simplicity -- it is not surfaced into user code
            prototype=default_buffer_prototype,
        )

    def _get_chunk_spec(self, shard_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            shape=self.chunk_shape,
            dtype=shard_spec.dtype,
            fill_value=shard_spec.fill_value,
            order=shard_spec.order,
            prototype=shard_spec.prototype,
        )

    def _get_chunks_per_shard(self, shard_spec: ArraySpec) -> ChunkCoords:
        return tuple(
            s // c
            for s, c in zip(
                shard_spec.shape,
                self.chunk_shape,
                strict=False,
            )
        )

    async def _load_shard_index_maybe(
        self, byte_getter: ByteGetter, chunks_per_shard: ChunkCoords
    ) -> _ShardIndex | None:
        shard_index_size = self._shard_index_size(chunks_per_shard)
        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = await byte_getter.get(
                prototype=default_buffer_prototype, byte_range=(0, shard_index_size)
            )
        else:
            index_bytes = await byte_getter.get(
                prototype=default_buffer_prototype, byte_range=(-shard_index_size, None)
            )
        if index_bytes is not None:
            return await self._decode_shard_index(index_bytes, chunks_per_shard)
        return None

    async def _load_shard_index(
        self, byte_getter: ByteGetter, chunks_per_shard: ChunkCoords
    ) -> _ShardIndex:
        return (
            await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
        ) or _ShardIndex.create_empty(chunks_per_shard)

    async def _load_full_shard_maybe(
        self, byte_getter: ByteGetter, prototype: BufferPrototype, chunks_per_shard: ChunkCoords
    ) -> _ShardReader | None:
        shard_bytes = await byte_getter.get(prototype=prototype)

        return (
            await _ShardReader.from_bytes(shard_bytes, self, chunks_per_shard)
            if shard_bytes
            else None
        )

    def compute_encoded_size(self, input_byte_length: int, shard_spec: ArraySpec) -> int:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        return input_byte_length + self._shard_index_size(chunks_per_shard)


register_codec("sharding_indexed", ShardingCodec)
