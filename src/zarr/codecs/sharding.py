from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from operator import itemgetter
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import (
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    Codec,
)
from zarr.abc.store import (
    ByteGetter,
    ByteSetter,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.crc32c_ import Crc32cCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import (
    Buffer,
    BufferPrototype,
    NDBuffer,
    default_buffer_prototype,
    numpy_buffer_prototype,
)
from zarr.core.chunk_grids import ChunkGrid, RegularChunkGrid
from zarr.core.codec_pipeline import ChunkTransform, fill_value_or_default
from zarr.core.common import (
    ShapeLike,
    parse_enum,
    parse_named_configuration,
    parse_shapelike,
    product,
)
from zarr.core.dtype.npy.int import UInt64
from zarr.core.indexing import (
    BasicIndexer,
    SelectorTuple,
    _morton_order,
    _morton_order_keys,
    c_order_iter,
    get_indexer,
    morton_order_iter,
)
from zarr.core.metadata.v3 import parse_codecs
from zarr.registry import get_ndbuffer_class, get_pipeline_class

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    from zarr.core.common import JSON
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

MAX_UINT_64 = 2**64 - 1
ShardMapping = Mapping[tuple[int, ...], Buffer | None]


class ShardingCodecIndexLocation(Enum):
    """
    Enum for index location used by the sharding codec.
    """

    start = "start"
    end = "end"


def parse_index_location(data: object) -> ShardingCodecIndexLocation:
    return parse_enum(data, ShardingCodecIndexLocation)


class _ShardIndex(NamedTuple):
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: npt.NDArray[np.uint64]

    @property
    def chunks_per_shard(self) -> tuple[int, ...]:
        result = tuple(self.offsets_and_lengths.shape[0:-1])
        # The cast is required until https://github.com/numpy/numpy/pull/27211 is merged
        return cast("tuple[int, ...]", result)

    def _localize_chunk(self, chunk_coords: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk_coords, self.offsets_and_lengths.shape, strict=False)
        )

    def is_all_empty(self) -> bool:
        return bool(np.array_equiv(self.offsets_and_lengths, MAX_UINT_64))

    def get_full_chunk_map(self) -> npt.NDArray[np.bool_]:
        return np.not_equal(self.offsets_and_lengths[..., 0], MAX_UINT_64)

    def get_chunk_slice(self, chunk_coords: tuple[int, ...]) -> tuple[int, int] | None:
        localized_chunk = self._localize_chunk(chunk_coords)
        chunk_start, chunk_len = self.offsets_and_lengths[localized_chunk]
        if (chunk_start, chunk_len) == (MAX_UINT_64, MAX_UINT_64):
            return None
        else:
            return (int(chunk_start), int(chunk_start + chunk_len))

    def get_chunk_slices_vectorized(
        self, chunk_coords_array: npt.NDArray[np.integer[Any]]
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], npt.NDArray[np.bool_]]:
        """Get chunk slices for multiple coordinates at once.

        Parameters
        ----------
        chunk_coords_array : ndarray of shape (n_chunks, n_dims)
            Array of chunk coordinates to look up.

        Returns
        -------
        starts : ndarray of shape (n_chunks,)
            Start byte positions for each chunk.
        ends : ndarray of shape (n_chunks,)
            End byte positions for each chunk.
        valid : ndarray of shape (n_chunks,)
            Boolean mask indicating which chunks are non-empty.
        """
        # Localize coordinates via modulo (vectorized)
        shard_shape = np.array(self.offsets_and_lengths.shape[:-1], dtype=np.uint64)
        localized = chunk_coords_array.astype(np.uint64) % shard_shape

        # Build index tuple for advanced indexing
        index_tuple = tuple(localized[:, i] for i in range(localized.shape[1]))

        # Fetch all offsets and lengths at once
        offsets_and_lengths = self.offsets_and_lengths[index_tuple]
        starts = offsets_and_lengths[:, 0]
        lengths = offsets_and_lengths[:, 1]

        # Check for valid (non-empty) chunks
        valid = starts != MAX_UINT_64

        # Compute end positions
        ends = starts + lengths

        return starts, ends, valid

    def set_chunk_slice(self, chunk_coords: tuple[int, ...], chunk_slice: slice | None) -> None:
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
            {offset for offset, _ in sorted_offsets_and_lengths if offset != MAX_UINT_64}
        ) != len(sorted_offsets_and_lengths):
            return False

        return all(
            offset % chunk_byte_length == 0 and length == chunk_byte_length
            for offset, length in sorted_offsets_and_lengths
        )

    @classmethod
    def create_empty(cls, chunks_per_shard: tuple[int, ...]) -> _ShardIndex:
        offsets_and_lengths = np.zeros(chunks_per_shard + (2,), dtype="<u8", order="C")
        offsets_and_lengths.fill(MAX_UINT_64)
        return cls(offsets_and_lengths)


class _ShardReader(ShardMapping):
    buf: Buffer
    index: _ShardIndex

    @classmethod
    async def from_bytes(
        cls, buf: Buffer, codec: ShardingCodec, chunks_per_shard: tuple[int, ...]
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
    def create_empty(
        cls, chunks_per_shard: tuple[int, ...], buffer_prototype: BufferPrototype | None = None
    ) -> _ShardReader:
        if buffer_prototype is None:
            buffer_prototype = default_buffer_prototype()
        index = _ShardIndex.create_empty(chunks_per_shard)
        obj = cls()
        obj.buf = buffer_prototype.buffer.create_zero_length()
        obj.index = index
        return obj

    def __getitem__(self, chunk_coords: tuple[int, ...]) -> Buffer:
        chunk_byte_slice = self.index.get_chunk_slice(chunk_coords)
        if chunk_byte_slice:
            return self.buf[chunk_byte_slice[0] : chunk_byte_slice[1]]
        raise KeyError

    def __len__(self) -> int:
        return int(self.index.offsets_and_lengths.size / 2)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return c_order_iter(self.index.offsets_and_lengths.shape[:-1])

    def to_dict_vectorized(
        self,
        chunk_coords_array: npt.NDArray[np.integer[Any]],
    ) -> dict[tuple[int, ...], Buffer | None]:
        """Build a dict of chunk coordinates to buffers using vectorized lookup.

        Parameters
        ----------
        chunk_coords_array : ndarray of shape (n_chunks, n_dims)
            Array of chunk coordinates for vectorized index lookup.

        Returns
        -------
        dict mapping chunk coordinate tuples to Buffer or None
        """
        starts, ends, valid = self.index.get_chunk_slices_vectorized(chunk_coords_array)
        chunks_per_shard = tuple(self.index.offsets_and_lengths.shape[:-1])
        chunk_coords_keys = _morton_order_keys(chunks_per_shard)

        result: dict[tuple[int, ...], Buffer | None] = {}
        for i, coords in enumerate(chunk_coords_keys):
            if valid[i]:
                result[coords] = self.buf[int(starts[i]) : int(ends[i])]
            else:
                result[coords] = None

        return result


@dataclass(frozen=True)
class ShardingCodec(
    ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin, ArrayBytesCodecPartialEncodeMixin
):
    """Sharding codec"""

    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end

    def __init__(
        self,
        *,
        chunk_shape: ShapeLike,
        codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(),),
        index_codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(), Crc32cCodec()),
        index_location: ShardingCodecIndexLocation | str = ShardingCodecIndexLocation.end,
    ) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        codecs_parsed = parse_codecs(codecs)
        index_codecs_parsed = parse_codecs(index_codecs)
        index_location_parsed = parse_index_location(index_location)

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "index_codecs", index_codecs_parsed)
        object.__setattr__(self, "index_location", index_location_parsed)

        # Use instance-local lru_cache to avoid memory leaks

        # numpy void scalars are not hashable, which means an array spec with a fill value that is
        # a numpy void scalar will break the lru_cache. This is commented for now but should be
        # fixed. See https://github.com/zarr-developers/zarr-python/issues/3054
        # object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))

    # todo: typedict return type
    def __getstate__(self) -> dict[str, Any]:
        return self.to_dict()

    def __setstate__(self, state: dict[str, Any]) -> None:
        config = state["configuration"]
        object.__setattr__(self, "chunk_shape", parse_shapelike(config["chunk_shape"]))
        object.__setattr__(self, "codecs", parse_codecs(config["codecs"]))
        object.__setattr__(self, "index_codecs", parse_codecs(config["index_codecs"]))
        object.__setattr__(self, "index_location", parse_index_location(config["index_location"]))

        # Use instance-local lru_cache to avoid memory leaks
        # object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "sharding_indexed")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def _get_chunk_transform(self, chunk_spec: ArraySpec) -> ChunkTransform:
        return ChunkTransform(codecs=self.codecs, array_spec=chunk_spec)

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": self.chunk_shape,
                "codecs": tuple(s.to_dict() for s in self.codecs),
                "index_codecs": tuple(s.to_dict() for s in self.index_codecs),
                "index_location": self.index_location.value,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        shard_spec = self._get_chunk_spec(array_spec)
        evolved_codecs = tuple(c.evolve_from_array_spec(array_spec=shard_spec) for c in self.codecs)
        if evolved_codecs != self.codecs:
            return replace(self, codecs=evolved_codecs)
        return self

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        if len(self.chunk_shape) != len(shape):
            raise ValueError(
                "The shard's `chunk_shape` and array's `shape` need to have the same number of dimensions."
            )
        if not isinstance(chunk_grid, RegularChunkGrid):
            raise TypeError("Sharding is only compatible with regular chunk grids.")
        if not all(
            s % c == 0
            for s, c in zip(
                chunk_grid.chunk_shape,
                self.chunk_shape,
                strict=False,
            )
        ):
            raise ValueError(
                f"The array's `chunk_shape` (got {chunk_grid.chunk_shape}) "
                f"needs to be divisible by the shard's inner `chunk_shape` (got {self.chunk_shape})."
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
        out = chunk_spec.prototype.nd_buffer.empty(
            shape=shard_shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )
        shard_dict = await _ShardReader.from_bytes(shard_bytes, self, chunks_per_shard)

        if shard_dict.index.is_all_empty():
            out.fill(shard_spec.fill_value)
            return out

        transform = self._get_chunk_transform(chunk_spec)
        fill_value = fill_value_or_default(chunk_spec)
        for chunk_coords, chunk_selection, out_selection, _is_complete in indexer:
            chunk_bytes = shard_dict.get(chunk_coords)
            if chunk_bytes is not None:
                chunk_array = await transform.decode_chunk_async(chunk_bytes)
                out[out_selection] = chunk_array[chunk_selection]
            else:
                out[out_selection] = fill_value

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
        out = shard_spec.prototype.nd_buffer.empty(
            shape=indexer.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = {chunk_coords for chunk_coords, *_ in indexed_chunks}

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
                        prototype=chunk_spec.prototype,
                        byte_range=RangeByteRequest(chunk_byte_slice[0], chunk_byte_slice[1]),
                    )
                    if chunk_bytes:
                        shard_dict[chunk_coords] = chunk_bytes

        # decode chunks and write them into the output buffer
        transform = self._get_chunk_transform(chunk_spec)
        fill_value = fill_value_or_default(chunk_spec)
        for chunk_coords, chunk_selection, out_selection, _is_complete in indexed_chunks:
            chunk_bytes = shard_dict.get(chunk_coords)
            if chunk_bytes is not None:
                chunk_array = await transform.decode_chunk_async(chunk_bytes)
                out[out_selection] = chunk_array[chunk_selection]
            else:
                out[out_selection] = fill_value

        if hasattr(indexer, "sel_shape"):
            return out.reshape(indexer.sel_shape)
        else:
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

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
        )

        transform = self._get_chunk_transform(chunk_spec)
        fill_value = fill_value_or_default(chunk_spec)
        shard_builder: dict[tuple[int, ...], Buffer | None] = {}

        for chunk_coords, _, out_selection, _is_complete in indexer:
            chunk_array = shard_array[out_selection]
            if not chunk_spec.config.write_empty_chunks and chunk_array.all_equal(fill_value):
                continue
            encoded = await transform.encode_chunk_async(chunk_array)
            if encoded is not None:
                shard_builder[chunk_coords] = encoded

        return await self._encode_shard_dict(
            shard_builder,
            chunks_per_shard=chunks_per_shard,
            buffer_prototype=default_buffer_prototype(),
        )

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

        shard_reader = await self._load_full_shard_maybe(
            byte_getter=byte_setter,
            prototype=chunk_spec.prototype,
            chunks_per_shard=chunks_per_shard,
        )
        shard_reader = shard_reader or _ShardReader.create_empty(chunks_per_shard)
        # Use vectorized lookup for better performance
        shard_dict: dict[tuple[int, ...], Buffer | None] = shard_reader.to_dict_vectorized(
            np.asarray(_morton_order(chunks_per_shard))
        )

        indexer = list(
            get_indexer(
                selection, shape=shard_shape, chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape)
            )
        )

        transform = self._get_chunk_transform(chunk_spec)
        fill_value = fill_value_or_default(chunk_spec)

        is_scalar = len(shard_array.shape) == 0
        for chunk_coords, chunk_selection, out_selection, is_complete in indexer:
            value = shard_array if is_scalar else shard_array[out_selection]
            if is_complete and not is_scalar and value.shape == chunk_spec.shape:
                # Complete overwrite with matching shape — use value directly
                chunk_data = value
            else:
                # Read-modify-write: decode existing or create new, merge data
                if is_complete:
                    existing_bytes = None
                else:
                    existing_bytes = shard_dict.get(chunk_coords)
                if existing_bytes is not None:
                    chunk_data = (await transform.decode_chunk_async(existing_bytes)).copy()
                else:
                    chunk_data = chunk_spec.prototype.nd_buffer.create(
                        shape=chunk_spec.shape,
                        dtype=chunk_spec.dtype.to_native_dtype(),
                        order=chunk_spec.order,
                        fill_value=fill_value,
                    )
                chunk_data[chunk_selection] = value

            if not chunk_spec.config.write_empty_chunks and chunk_data.all_equal(fill_value):
                shard_dict[chunk_coords] = None
            else:
                shard_dict[chunk_coords] = await transform.encode_chunk_async(chunk_data)

        buf = await self._encode_shard_dict(
            shard_dict,
            chunks_per_shard=chunks_per_shard,
            buffer_prototype=default_buffer_prototype(),
        )

        if buf is None:
            await byte_setter.delete()
        else:
            await byte_setter.set(buf)

    async def _encode_shard_dict(
        self,
        map: ShardMapping,
        chunks_per_shard: tuple[int, ...],
        buffer_prototype: BufferPrototype,
    ) -> Buffer | None:
        index = _ShardIndex.create_empty(chunks_per_shard)

        buffers = []

        template = buffer_prototype.buffer.create_zero_length()
        chunk_start = 0
        for chunk_coords in morton_order_iter(chunks_per_shard):
            value = map.get(chunk_coords)
            if value is None:
                continue

            if len(value) == 0:
                continue

            chunk_length = len(value)
            buffers.append(value)
            index.set_chunk_slice(chunk_coords, slice(chunk_start, chunk_start + chunk_length))
            chunk_start += chunk_length

        if len(buffers) == 0:
            return None

        index_bytes = await self._encode_shard_index(index)
        if self.index_location == ShardingCodecIndexLocation.start:
            empty_chunks_mask = index.offsets_and_lengths[..., 0] == MAX_UINT_64
            index.offsets_and_lengths[~empty_chunks_mask, 0] += len(index_bytes)
            index_bytes = await self._encode_shard_index(
                index
            )  # encode again with corrected offsets
            buffers.insert(0, index_bytes)
        else:
            buffers.append(index_bytes)

        return template.combine(buffers)

    def _is_total_shard(
        self, all_chunk_coords: set[tuple[int, ...]], chunks_per_shard: tuple[int, ...]
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords for chunk_coords in c_order_iter(chunks_per_shard)
        )

    async def _decode_shard_index(
        self, index_bytes: Buffer, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex:
        index_array = next(
            iter(
                await get_pipeline_class()
                .from_codecs(self.index_codecs)
                .decode(
                    [(index_bytes, self._get_index_chunk_spec(chunks_per_shard))],
                )
            )
        )
        # This cannot be None because we have the bytes already
        index_array = cast(NDBuffer, index_array)
        return _ShardIndex(index_array.as_numpy_array())

    async def _encode_shard_index(self, index: _ShardIndex) -> Buffer:
        index_bytes = next(
            iter(
                await get_pipeline_class()
                .from_codecs(self.index_codecs)
                .encode(
                    [
                        (
                            get_ndbuffer_class().from_numpy_array(index.offsets_and_lengths),
                            self._get_index_chunk_spec(index.chunks_per_shard),
                        )
                    ],
                )
            )
        )
        assert index_bytes is not None
        assert isinstance(index_bytes, Buffer)
        return index_bytes

    def _shard_index_size(self, chunks_per_shard: tuple[int, ...]) -> int:
        return (
            get_pipeline_class()
            .from_codecs(self.index_codecs)
            .compute_encoded_size(
                16 * product(chunks_per_shard), self._get_index_chunk_spec(chunks_per_shard)
            )
        )

    def _get_index_chunk_spec(self, chunks_per_shard: tuple[int, ...]) -> ArraySpec:
        return ArraySpec(
            shape=chunks_per_shard + (2,),
            dtype=UInt64(endianness="little"),
            fill_value=MAX_UINT_64,
            config=ArrayConfig(
                order="C", write_empty_chunks=False
            ),  # Note: this is hard-coded for simplicity -- it is not surfaced into user code,
            prototype=default_buffer_prototype(),
        )

    def _get_chunk_spec(self, shard_spec: ArraySpec) -> ArraySpec:
        return ArraySpec(
            shape=self.chunk_shape,
            dtype=shard_spec.dtype,
            fill_value=shard_spec.fill_value,
            config=shard_spec.config,
            prototype=shard_spec.prototype,
        )

    def _get_chunks_per_shard(self, shard_spec: ArraySpec) -> tuple[int, ...]:
        return tuple(
            s // c
            for s, c in zip(
                shard_spec.shape,
                self.chunk_shape,
                strict=False,
            )
        )

    async def _load_shard_index_maybe(
        self, byte_getter: ByteGetter, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex | None:
        shard_index_size = self._shard_index_size(chunks_per_shard)
        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = await byte_getter.get(
                prototype=numpy_buffer_prototype(),
                byte_range=RangeByteRequest(0, shard_index_size),
            )
        else:
            index_bytes = await byte_getter.get(
                prototype=numpy_buffer_prototype(), byte_range=SuffixByteRequest(shard_index_size)
            )
        if index_bytes is not None:
            return await self._decode_shard_index(index_bytes, chunks_per_shard)
        return None

    async def _load_shard_index(
        self, byte_getter: ByteGetter, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex:
        return (
            await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
        ) or _ShardIndex.create_empty(chunks_per_shard)

    async def _load_full_shard_maybe(
        self, byte_getter: ByteGetter, prototype: BufferPrototype, chunks_per_shard: tuple[int, ...]
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
