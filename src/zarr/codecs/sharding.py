from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import (
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    Codec,
    CodecPipeline,
)
from zarr.abc.store import (
    ByteGetter,
    ByteRequest,
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
from zarr.core.chunk_grids import ChunkGrid
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
    ChunkProjection,
    SelectorTuple,
    _lexicographic_order,
    colexicographic_order_coords,
    get_indexer,
    lexicographic_order_coords,
    morton_order_coords,
)
from zarr.core.metadata.v3 import (
    ChunkGridMetadata,
    RectilinearChunkGridMetadata,
    RegularChunkGridMetadata,
    parse_codecs,
)
from zarr.registry import get_ndbuffer_class, get_pipeline_class
from zarr.storage._common import StorePath
from zarr.storage._utils import _normalize_byte_range_index

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Final, Self

    from zarr.core.common import JSON
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

MAX_UINT_64 = 2**64 - 1
ShardMapping = Mapping[tuple[int, ...], Buffer | None]
ShardMutableMapping = MutableMapping[tuple[int, ...], Buffer | None]


class ShardingCodecIndexLocation(Enum):
    """
    Enum for index location used by the sharding codec.
    """

    start = "start"
    end = "end"


SubchunkWriteOrder = Literal["morton", "unordered", "lexicographic", "colexicographic"]
SUBCHUNK_WRITE_ORDER: Final[tuple[str, str, str, str]] = (
    "morton",
    "unordered",
    "lexicographic",
    "colexicographic",
)


def parse_index_location(data: object) -> ShardingCodecIndexLocation:
    return parse_enum(data, ShardingCodecIndexLocation)


@dataclass(frozen=True)
class _ShardingByteGetter(ByteGetter):
    shard_dict: ShardMapping
    chunk_coords: tuple[int, ...]

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        assert prototype == default_buffer_prototype(), (
            f"prototype is not supported within shards currently. diff: {prototype} != {default_buffer_prototype()}"
        )
        value = self.shard_dict.get(self.chunk_coords)
        if value is None:
            return None
        if byte_range is None:
            return value
        start, stop = _normalize_byte_range_index(value, byte_range)
        return value[start:stop]


@dataclass(frozen=True)
class _ShardingByteSetter(_ShardingByteGetter, ByteSetter):
    shard_dict: ShardMutableMapping

    async def set(self, value: Buffer, byte_range: ByteRequest | None = None) -> None:
        assert byte_range is None, "byte_range is not supported within shards"
        self.shard_dict[self.chunk_coords] = value

    async def delete(self) -> None:
        del self.shard_dict[self.chunk_coords]

    async def set_if_not_exists(self, default: Buffer) -> None:
        self.shard_dict.setdefault(self.chunk_coords, default)


class _ShardIndex(NamedTuple):
    # the chunk grid shape of a single shard
    chunks_per_shard: tuple[int, ...]
    # dtype uint64, shape (chunks_per_shard_0, chunks_per_shard_1, ..., 2)
    offsets_and_lengths: npt.NDArray[np.uint64]

    def _localize_chunk(self, chunk_coords: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            chunk_i % shard_i
            for chunk_i, shard_i in zip(chunk_coords, self.chunks_per_shard, strict=False)
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
        # Handle 0-dimensional arrays (n_dims == 0): the shard holds a single
        # chunk, so every coordinate maps to the same flat entry.
        if chunk_coords_array.shape[1] == 0:
            offsets_and_lengths = self.offsets_and_lengths.reshape(-1, 2)
            offsets_and_lengths = np.broadcast_to(
                offsets_and_lengths, (chunk_coords_array.shape[0], 2)
            )
        else:
            # Localize coordinates via modulo (vectorized)
            shard_shape = np.array(self.chunks_per_shard, dtype=np.uint64)
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

    @classmethod
    def create_empty(cls, chunks_per_shard: tuple[int, ...]) -> _ShardIndex:
        offsets_and_lengths = np.zeros(chunks_per_shard + (2,), dtype="<u8", order="C")
        offsets_and_lengths.fill(MAX_UINT_64)
        return cls(chunks_per_shard, offsets_and_lengths)


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
        return iter(lexicographic_order_coords(self.index.chunks_per_shard))

    def to_dict_vectorized(self) -> dict[tuple[int, ...], Buffer | None]:
        """Build a dict of chunk coordinates to buffers using vectorized lookup.

        The full per-shard chunk coordinate grid (both the array used for the
        vectorized index lookup and the plain tuples used as dict keys) is
        cached on `chunks_per_shard`, so neither is rebuilt on every call. For a
        shard with tens of thousands of chunks this avoids reconstructing that
        many tuples on every partial write.

        Returns
        -------
        dict mapping chunk coordinate tuples to Buffer or None
        """
        chunks_per_shard = self.index.chunks_per_shard
        # The same chunk-grid coordinates are needed in two forms, and neither can
        # stand in for the other:
        #   - `chunk_coords_array`: an (n_chunks, n_dims) numpy array, fed to the
        #     vectorized index lookup, which does modulo + advanced indexing on it.
        #     A list of tuples can't be used for that without first being arrayified.
        #   - `chunk_coords_keys`: the same coordinates as hashable Python tuples,
        #     used as the result dict's keys. numpy array rows are unhashable
        #     (mutable), so they can't key a dict.
        # Both are cached per shape (see indexing.py), so neither is rebuilt here;
        # row i of the array and key i refer to the same chunk.
        chunk_coords_array = _lexicographic_order(chunks_per_shard)
        chunk_coords_keys = lexicographic_order_coords(chunks_per_shard)
        starts, ends, valid = self.index.get_chunk_slices_vectorized(chunk_coords_array)

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
    """Sharding codec.

    `subchunk_write_order` controls the physical order of subchunks within a shard. It is a
    write-time setting only: it is not stored in array metadata, so reopening a sharded array
    does not recover it (the setting reverts to the `morton` default per codec instance).
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end
    subchunk_write_order: SubchunkWriteOrder = "morton"

    def __init__(
        self,
        *,
        chunk_shape: ShapeLike,
        codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(),),
        index_codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(), Crc32cCodec()),
        index_location: ShardingCodecIndexLocation | str = ShardingCodecIndexLocation.end,
        subchunk_write_order: SubchunkWriteOrder = "morton",
    ) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        codecs_parsed = parse_codecs(codecs)
        index_codecs_parsed = parse_codecs(index_codecs)
        index_location_parsed = parse_index_location(index_location)
        if subchunk_write_order not in SUBCHUNK_WRITE_ORDER:
            raise ValueError(
                f"Unrecognized subchunk write order: {subchunk_write_order}. Only {SUBCHUNK_WRITE_ORDER} are allowed."
            )

        object.__setattr__(self, "chunk_shape", chunk_shape_parsed)
        object.__setattr__(self, "codecs", codecs_parsed)
        object.__setattr__(self, "index_codecs", index_codecs_parsed)
        object.__setattr__(self, "index_location", index_location_parsed)
        object.__setattr__(self, "subchunk_write_order", subchunk_write_order)

        # Use instance-local lru_cache to avoid memory leaks

        # numpy void scalars are not hashable, which means an array spec with a fill value that is
        # a numpy void scalar will break the lru_cache. This is commented for now but should be
        # fixed. See https://github.com/zarr-developers/zarr-python/issues/3054
        # object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))

    # todo: typedict return type
    def __getstate__(self) -> dict[str, Any]:
        # `subchunk_write_order` is not part of codec metadata (`to_dict`), so carry it
        # explicitly to survive a pickle round-trip (otherwise it reverts to `morton`).
        return {"subchunk_write_order": self.subchunk_write_order, **self.to_dict()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        config = state["configuration"]
        object.__setattr__(self, "chunk_shape", parse_shapelike(config["chunk_shape"]))
        object.__setattr__(self, "codecs", parse_codecs(config["codecs"]))
        object.__setattr__(self, "index_codecs", parse_codecs(config["index_codecs"]))
        object.__setattr__(self, "index_location", parse_index_location(config["index_location"]))
        object.__setattr__(self, "subchunk_write_order", state["subchunk_write_order"])

        # Use instance-local lru_cache to avoid memory leaks
        # object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "sharding_indexed")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    @property
    def codec_pipeline(self) -> CodecPipeline:
        return get_pipeline_class().from_codecs(self.codecs)

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
        chunk_grid: ChunkGridMetadata,
    ) -> None:
        if len(self.chunk_shape) != len(shape):
            raise ValueError(
                "The shard's `chunk_shape` and array's `shape` need to have the same number of dimensions."
            )
        if isinstance(chunk_grid, RegularChunkGridMetadata):
            edges_per_dim: tuple[tuple[int, ...], ...] = tuple((s,) for s in chunk_grid.chunk_shape)
        elif isinstance(chunk_grid, RectilinearChunkGridMetadata):
            edges_per_dim = tuple(
                (s,) if isinstance(s, int) else s for s in chunk_grid.chunk_shapes
            )
        else:
            raise TypeError(
                f"Sharding is only compatible with regular and rectilinear chunk grids, "
                f"got {type(chunk_grid)}"
            )
        for i, (edges, inner) in enumerate(zip(edges_per_dim, self.chunk_shape, strict=False)):
            for edge in set(edges):
                if edge % inner != 0:
                    raise ValueError(
                        f"Chunk edge length {edge} in dimension {i} is not "
                        f"divisible by the shard's inner chunk size {inner}."
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
            chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
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

        # decoding chunks and writing them into the output buffer
        await self.codec_pipeline.read(
            [
                (
                    _ShardingByteGetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
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
            chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
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
        shard_dict_maybe: ShardMapping | None
        if self._is_total_shard(all_chunk_coords, chunks_per_shard):
            # read entire shard
            shard_dict_maybe = await self._load_full_shard_maybe(
                byte_getter=byte_getter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
        else:
            # read some chunks within the shard
            shard_dict_maybe = await self._load_partial_shard_maybe(
                byte_getter,
                chunk_spec.prototype,
                chunks_per_shard,
                all_chunk_coords,
            )

        if shard_dict_maybe is None:
            return None
        shard_dict = shard_dict_maybe

        # decoding chunks and writing them into the output buffer
        await self.codec_pipeline.read(
            [
                (
                    _ShardingByteGetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            out,
        )

        if hasattr(indexer, "sel_shape"):
            return out.reshape(indexer.sel_shape)
        else:
            return out

    def _subchunk_order_iter(
        self, chunks_per_shard: tuple[int, ...], subchunk_write_order: SubchunkWriteOrder
    ) -> Iterable[tuple[int, ...]]:
        subchunk_iter: Iterable[tuple[int, ...]]
        match subchunk_write_order:
            case "morton":
                subchunk_iter = morton_order_coords(chunks_per_shard)
            case "lexicographic":
                subchunk_iter = lexicographic_order_coords(chunks_per_shard)
            case "colexicographic":
                subchunk_iter = colexicographic_order_coords(chunks_per_shard)
            case "unordered":
                # "unordered" promises no particular layout; today it happens to be
                # lexicographic, but callers must not rely on that.
                subchunk_iter = np.ndindex(chunks_per_shard)
            case _:
                raise ValueError(f"Unrecognized subchunk write order: {subchunk_write_order!r}.")
        return subchunk_iter

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
                chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
            )
        )
        shard_builder = dict.fromkeys(lexicographic_order_coords(chunks_per_shard))

        await self.codec_pipeline.write(
            [
                (
                    _ShardingByteSetter(shard_builder, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            shard_array,
        )

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

        indexer = list(
            get_indexer(
                selection,
                shape=shard_shape,
                chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
            )
        )

        if self._is_complete_shard_write(indexer, chunks_per_shard):
            shard_dict = dict.fromkeys(lexicographic_order_coords(chunks_per_shard))
        else:
            shard_reader = await self._load_full_shard_maybe(
                byte_getter=byte_setter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
            shard_reader = shard_reader or _ShardReader.create_empty(chunks_per_shard)
            # Use vectorized lookup for better performance. The lexicographic
            # coordinate array and keys are cached, so neither is rebuilt on
            # every write.
            shard_dict = shard_reader.to_dict_vectorized()

        await self.codec_pipeline.write(
            [
                (
                    _ShardingByteSetter(shard_dict, chunk_coords),
                    chunk_spec,
                    chunk_selection,
                    out_selection,
                    is_complete_shard,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_shard in indexer
            ],
            shard_array,
        )
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
        for chunk_coords in self._subchunk_order_iter(chunks_per_shard, self.subchunk_write_order):
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
        # `all_chunk_coords` comes from an indexer over this shard's chunk grid, so
        # it is always a subset of that grid (`validate` requires the shard shape to
        # be divisible by the inner chunk shape, so the indexer cannot produce an
        # out-of-grid coordinate). A subset whose size equals the grid's is the
        # whole grid, so the count check alone proves totality — no need to build
        # and membership-test the full coordinate set on this hot path.
        return len(all_chunk_coords) == product(chunks_per_shard)

    def _is_complete_shard_write(
        self,
        indexed_chunks: Sequence[ChunkProjection],
        chunks_per_shard: tuple[int, ...],
    ) -> bool:
        all_chunk_coords = {chunk_coords for chunk_coords, *_ in indexed_chunks}
        return self._is_total_shard(all_chunk_coords, chunks_per_shard) and all(
            is_complete_chunk for *_, is_complete_chunk in indexed_chunks
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
        return _ShardIndex(chunks_per_shard, index_array.as_numpy_array())

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

    async def _load_partial_shard_maybe(
        self,
        byte_getter: ByteGetter,
        prototype: BufferPrototype,
        chunks_per_shard: tuple[int, ...],
        all_chunk_coords: set[tuple[int, ...]],
    ) -> ShardMapping | None:
        """
        Read chunks from `byte_getter` for the case where the read is less than a full shard.
        Returns a mapping of chunk coordinates to bytes or None.
        """
        shard_index = await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
        if shard_index is None:
            return None

        # Pair up chunks and their byte ranges as list[tuple[chunk_coord, byte_range]]
        chunk_coord_byte_ranges: list[tuple[tuple[int, ...], RangeByteRequest]] = []
        for chunk_coord in all_chunk_coords:
            chunk_byte_slice = shard_index.get_chunk_slice(chunk_coord)
            if chunk_byte_slice is not None:
                chunk_coord_byte_ranges.append(
                    (chunk_coord, RangeByteRequest(chunk_byte_slice[0], chunk_byte_slice[1]))
                )

        if not chunk_coord_byte_ranges:
            return {}

        shard_dict: ShardMutableMapping = {}
        if isinstance(byte_getter, StorePath):
            # External store: use Store.get_ranges for coalescing + concurrency.
            byte_ranges = [byte_range for _, byte_range in chunk_coord_byte_ranges]
            try:
                async for group in byte_getter.store.get_ranges(
                    byte_getter.path, byte_ranges, prototype=prototype
                ):
                    for idx, buf in group:
                        if buf is not None:
                            chunk_coord, _ = chunk_coord_byte_ranges[idx]
                            shard_dict[chunk_coord] = buf
            except BaseExceptionGroup as eg:
                # `Store.get_ranges` raises FileNotFoundError (wrapped in a
                # BaseExceptionGroup) if any underlying fetch indicates the key is
                # absent. The shard index loaded above, so this typically means a
                # race where the shard was deleted mid-read; treat it as "shard
                # gone" to match the index-missing branch (return None). Anything
                # else in the group (e.g. IO errors) is re-raised.
                _, rest = eg.split(FileNotFoundError)
                if rest is not None:
                    raise rest from None
                return None
        else:
            # Any other ByteGetter. In practice only `_ShardingByteGetter` for
            # nested sharding, which slices an in-memory buffer (no I/O to coalesce).
            for chunk_coord, byte_range in chunk_coord_byte_ranges:
                buf = await byte_getter.get(prototype, byte_range)
                if buf is not None:
                    shard_dict[chunk_coord] = buf

        return shard_dict

    def compute_encoded_size(self, input_byte_length: int, shard_spec: ArraySpec) -> int:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        return input_byte_length + self._shard_index_size(chunks_per_shard)
