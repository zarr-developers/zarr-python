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
    Store,
    SuffixByteRequest,
    SupportsGetSync,
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
from zarr.core.dtype.common import HasEndianness
from zarr.core.dtype.npy.int import UInt64
from zarr.core.indexing import (
    BasicIndexer,
    ChunkProjection,
    SelectorTuple,
    c_order_iter,
    get_indexer,
    morton_order_iter,
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

    def is_dense(self, chunk_byte_length: int) -> bool:
        """True when every chunk is present, fixed-length, and uniquely placed.

        Used to gate the vectorized whole-shard decode: a dense fixed-size shard
        is a regular grid of equal-length payloads, so it can be reshaped/scattered
        in bulk rather than decoded chunk-by-chunk.
        """
        offsets = self.offsets_and_lengths[..., 0].reshape(-1)
        lengths = self.offsets_and_lengths[..., 1].reshape(-1)
        # all present
        if bool(np.any(offsets == MAX_UINT_64)):
            return False
        # all the same fixed length
        if not bool(np.all(lengths == chunk_byte_length)):
            return False
        # offsets unique (no two chunks share a slot)
        return int(np.unique(offsets).size) == int(offsets.size)

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
        return c_order_iter(self.index.chunks_per_shard)

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

        result: dict[tuple[int, ...], Buffer | None] = {}
        for i, coords in enumerate(chunk_coords_array):
            if valid[i]:
                result[tuple(coords.ravel())] = self.buf[int(starts[i]) : int(ends[i])]
            else:
                result[tuple(coords.ravel())] = None

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

    is_fixed_size = False

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
        object.__setattr__(
            self, "_get_inner_chunk_transform", lru_cache()(self._get_inner_chunk_transform)
        )
        object.__setattr__(
            self, "_get_index_chunk_transform", lru_cache()(self._get_index_chunk_transform)
        )

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
        object.__setattr__(
            self, "_get_inner_chunk_transform", lru_cache()(self._get_inner_chunk_transform)
        )
        object.__setattr__(
            self, "_get_index_chunk_transform", lru_cache()(self._get_index_chunk_transform)
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "sharding_indexed")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    @property
    def codec_pipeline(self) -> CodecPipeline:
        # Resolve against the configured pipeline (registry default), matching the
        # rest of this module's use of get_pipeline_class — NOT a hard-coded
        # BatchedCodecPipeline. This restores main's behavior (#2179) that the
        # branch had reverted: the inner sub-chunk pipeline follows the same
        # codec_pipeline.path config as the outer array.
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

    def _get_inner_chunk_transform(self, shard_spec: ArraySpec) -> Any:
        """Build a ChunkTransform for the inner codec chain.

        The cache key is the shard_spec because evolved codecs may
        depend on it. The runtime chunk_spec is supplied per call.
        """
        from zarr.core.codec_pipeline import ChunkTransform

        chunk_spec = self._get_chunk_spec(shard_spec)
        evolved = tuple(c.evolve_from_array_spec(array_spec=chunk_spec) for c in self.codecs)
        return ChunkTransform(codecs=evolved)

    def _get_index_chunk_transform(self, chunks_per_shard: tuple[int, ...]) -> Any:
        """Build a ChunkTransform for the index codec chain."""
        from zarr.core.codec_pipeline import ChunkTransform

        index_spec = self._get_index_chunk_spec(chunks_per_shard)
        evolved = tuple(c.evolve_from_array_spec(array_spec=index_spec) for c in self.index_codecs)
        return ChunkTransform(codecs=evolved)

    def _decode_shard_index_sync(
        self, index_bytes: Buffer, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex:
        """Decode shard index synchronously using ChunkTransform."""
        index_transform = self._get_index_chunk_transform(chunks_per_shard)
        index_spec = self._get_index_chunk_spec(chunks_per_shard)
        index_array = index_transform.decode_chunk(index_bytes, index_spec)
        return _ShardIndex(chunks_per_shard, index_array.as_numpy_array())

    def _encode_shard_index_sync(self, index: _ShardIndex) -> Buffer:
        """Encode shard index synchronously using ChunkTransform."""
        index_transform = self._get_index_chunk_transform(index.chunks_per_shard)
        index_spec = self._get_index_chunk_spec(index.chunks_per_shard)
        index_nd = get_ndbuffer_class().from_numpy_array(index.offsets_and_lengths)
        result: Buffer | None = index_transform.encode_chunk(index_nd, index_spec)
        assert result is not None
        return result

    def _shard_reader_from_bytes_sync(
        self, buf: Buffer, chunks_per_shard: tuple[int, ...]
    ) -> _ShardReader:
        """Sync version of _ShardReader.from_bytes."""
        shard_index_size = self._shard_index_size(chunks_per_shard)
        if self.index_location == ShardingCodecIndexLocation.start:
            shard_index_bytes = buf[:shard_index_size]
        else:
            shard_index_bytes = buf[-shard_index_size:]
        index = self._decode_shard_index_sync(shard_index_bytes, chunks_per_shard)
        reader = _ShardReader()
        reader.buf = buf
        reader.index = index
        return reader

    def _decode_sync(
        self,
        shard_bytes: Buffer,
        shard_spec: ArraySpec,
    ) -> NDBuffer:
        """Decode a full shard synchronously.

        Sync counterpart to `_decode_single`. Same semantics (decode every
        inner chunk and assemble the full shard array) but routes through
        `ChunkTransform` instead of the async codec pipeline, so it can
        run on the sync codec-pipeline fast path without an event loop.

        For a partial read where the caller only needs a slice of the shard,
        use `_decode_partial_sync` instead — it fetches only the byte
        ranges that overlap the selection.

        This method does not parallelize decompression, but should.
        See TODO: make issue for handling subchunk parallelism
        """
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)
        inner_transform = self._get_inner_chunk_transform(shard_spec)

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
        )

        out = chunk_spec.prototype.nd_buffer.empty(
            shape=shard_shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        shard_dict = self._shard_reader_from_bytes_sync(shard_bytes, chunks_per_shard)

        if shard_dict.index.is_all_empty():
            out.fill(shard_spec.fill_value)
            return out

        for chunk_coords, chunk_selection, out_selection, _ in indexer:
            try:
                chunk_bytes = shard_dict[chunk_coords]
            except KeyError:
                out[out_selection] = shard_spec.fill_value
                continue
            chunk_array = inner_transform.decode_chunk(chunk_bytes, chunk_spec)
            out[out_selection] = chunk_array[chunk_selection]

        return out

    def _encode_sync(
        self,
        shard_array: NDBuffer,
        shard_spec: ArraySpec,
    ) -> Buffer | None:
        """Encode a full shard synchronously.

        Sync counterpart to `_encode_single`. This is reached when a
        `ShardingCodec` is an *inner* codec of another sharding codec (nested
        sharding): the outer codec encodes each inner chunk through its
        `ChunkTransform`, which calls this method on the inner `ShardingCodec`.

        Each inner chunk is encoded through the inner `ChunkTransform` and
        collected into an intermediate `dict`. The dict's key order is
        immaterial — the physical on-disk layout is decided downstream by the
        `subchunk_write_order` loop in `_encode_shard_dict_sync` (this method
        does NOT impose a layout). Empty inner chunks become `None` entries when
        `write_empty_chunks` is False, signalling `_encode_shard_dict_sync` to
        elide them from the data section and mark them empty in the shard index.

        Returns `None` if every inner chunk was elided (an all-empty shard) —
        callers treat that as "delete the shard key".

        This method does not parallelize compression, but should.
        See TODO: make issue for handling subchunk parallelism

        For a partial write that only touches some inner chunks, use
        `_encode_partial_sync` instead.
        """
        shard_shape = shard_spec.shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)
        inner_transform = self._get_inner_chunk_transform(shard_spec)

        indexer = BasicIndexer(
            tuple(slice(0, s) for s in shard_shape),
            shape=shard_shape,
            chunk_grid=ChunkGrid.from_sizes(shard_shape, self.chunk_shape),
        )

        # Key order here is immaterial; _encode_shard_dict_sync lays the present
        # chunks out in subchunk_write_order.
        shard_builder: dict[tuple[int, ...], Buffer | None] = dict.fromkeys(
            morton_order_iter(chunks_per_shard)
        )

        skip_empty = not shard_spec.config.write_empty_chunks
        fill_value = shard_spec.fill_value
        if fill_value is None:
            fill_value = shard_spec.dtype.default_scalar()

        for chunk_coords, _chunk_selection, out_selection, _ in indexer:
            chunk_array = shard_array[out_selection]
            if skip_empty and chunk_array.all_equal(fill_value):
                shard_builder[chunk_coords] = None
            else:
                encoded = inner_transform.encode_chunk(chunk_array, chunk_spec)
                shard_builder[chunk_coords] = encoded

        return self._encode_shard_dict_sync(
            shard_builder,
            chunks_per_shard=chunks_per_shard,
            buffer_prototype=default_buffer_prototype(),
        )

    def _encode_partial_sync(
        self,
        byte_setter: Any,
        value: NDBuffer,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> None:
        """Sync equivalent of `_encode_partial_single`.

        Receives the source data for the written region (not a pre-merged
        shard array) and the selection within the shard, matching the
        calling convention of the async partial-encode path used by
        `BatchedCodecPipeline`.

        This method does not parallelize compression, but should.
        See TODO: make issue for handling subchunk parallelism

        Loads the existing shard, merges the written region into the affected
        inner chunks, and rewrites the whole shard.
        """
        shard_shape = shard_spec.shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)
        inner_transform = self._get_inner_chunk_transform(shard_spec)

        indexer = list(
            get_indexer(
                selection,
                shape=shard_shape,
                chunk_grid=ChunkGrid.from_sizes(shard_shape, self.chunk_shape),
            )
        )

        is_complete = self._is_complete_shard_write(indexer, chunks_per_shard)

        skip_empty = not shard_spec.config.write_empty_chunks
        fill_value = shard_spec.fill_value
        if fill_value is None:
            fill_value = shard_spec.dtype.default_scalar()

        is_scalar = len(value.shape) == 0

        # Load existing inner-chunk bytes into a dict (same structure as
        # the async path's shard_dict).
        if is_complete:
            shard_dict: dict[tuple[int, ...], Buffer | None] = dict.fromkeys(
                morton_order_iter(chunks_per_shard)
            )
        else:
            existing_bytes = byte_setter.get_sync(prototype=shard_spec.prototype)
            if existing_bytes is not None:
                shard_reader_fb = self._shard_reader_from_bytes_sync(
                    existing_bytes, chunks_per_shard
                )
                # Build the dict with one vectorized index lookup over all chunks,
                # matching the async _encode_partial_single path. A per-coordinate
                # __getitem__ loop here is O(n_chunks) Python overhead that dominates
                # partial writes into shards with many inner chunks.
                shard_dict = shard_reader_fb.to_dict_vectorized(
                    np.array(list(np.ndindex(chunks_per_shard)))
                )
            else:
                shard_dict = dict.fromkeys(morton_order_iter(chunks_per_shard))

        # Merge, encode, and store each affected inner chunk into shard_dict.
        #
        # Scalar fast path: when the written value is a scalar broadcast, every
        # *complete* inner chunk is byte-for-byte identical — same fill, same
        # empty-check, same encoded bytes. Compute that outcome once and reuse it
        # for all complete chunks instead of re-merging, re-checking, and
        # re-encoding tens of thousands of identical chunks. Incomplete (edge)
        # chunks still merge against their own existing data individually.
        # `_sentinel` distinguishes "not computed yet" from a memoized `None`
        # (an empty chunk).
        _sentinel = object()
        scalar_complete_result: Buffer | None | object = _sentinel

        for chunk_coords, chunk_sel, out_sel, is_complete_chunk in indexer:
            if is_scalar and is_complete_chunk:
                if scalar_complete_result is _sentinel:
                    chunk_array = chunk_spec.prototype.nd_buffer.create(
                        shape=self.chunk_shape,
                        dtype=shard_spec.dtype.to_native_dtype(),
                        order=shard_spec.order,
                        fill_value=fill_value,
                    )
                    chunk_array[chunk_sel] = value
                    if skip_empty and chunk_array.all_equal(fill_value):
                        scalar_complete_result = None
                    else:
                        scalar_complete_result = inner_transform.encode_chunk(
                            chunk_array, chunk_spec
                        )
                shard_dict[chunk_coords] = scalar_complete_result  # type: ignore[assignment]
                continue

            chunk_value = value if is_scalar else value[out_sel]

            if is_complete_chunk and not is_scalar:
                chunk_array = chunk_value
            else:
                existing_raw = shard_dict.get(chunk_coords)
                if existing_raw is not None:
                    chunk_array = inner_transform.decode_chunk(existing_raw, chunk_spec).copy()
                else:
                    chunk_array = chunk_spec.prototype.nd_buffer.create(
                        shape=self.chunk_shape,
                        dtype=shard_spec.dtype.to_native_dtype(),
                        order=shard_spec.order,
                        fill_value=fill_value,
                    )
                chunk_array[chunk_sel] = chunk_value

            if skip_empty and chunk_array.all_equal(fill_value):
                shard_dict[chunk_coords] = None
            else:
                shard_dict[chunk_coords] = inner_transform.encode_chunk(chunk_array, chunk_spec)

        blob = self._encode_shard_dict_sync(
            shard_dict,
            chunks_per_shard=chunks_per_shard,
            buffer_prototype=default_buffer_prototype(),
        )
        if blob is None:
            byte_setter.delete_sync()
        else:
            byte_setter.set_sync(blob)

    def _build_shard_layout(
        self,
        shard_dict: ShardMapping,
        chunks_per_shard: tuple[int, ...],
    ) -> tuple[_ShardIndex, list[Buffer]] | None:
        """Lay out the present inner chunks of a shard. Pure compute, no IO.

        Packs the encoded inner chunks (in the codec's `subchunk_write_order`)
        into a contiguous data section and builds a shard index pointing each
        present chunk at its ABSOLUTE byte offset within the final blob: when
        the index is stored at the start, offsets are pre-shifted by the index
        size (known without encoding — the index codecs are fixed-size, which
        every index read path already relies on), so the index can be encoded
        exactly once by the caller.

        Returns `(index, data_buffers)`, or `None` for an all-empty shard (no
        chunks present). Shared by the sync and async `_encode_shard_dict*` so
        the layout/offset logic cannot drift between them.
        """
        index = _ShardIndex.create_empty(chunks_per_shard)
        buffers: list[Buffer] = []
        chunk_start = (
            self._shard_index_size(chunks_per_shard)
            if self.index_location == ShardingCodecIndexLocation.start
            else 0
        )

        for chunk_coords in self._subchunk_order_iter(chunks_per_shard, self.subchunk_write_order):
            value = shard_dict.get(chunk_coords)
            if value is None or len(value) == 0:
                continue
            chunk_length = len(value)
            buffers.append(value)
            index.set_chunk_slice(chunk_coords, slice(chunk_start, chunk_start + chunk_length))
            chunk_start += chunk_length

        if len(buffers) == 0:
            return None
        return index, buffers

    def _assemble_shard(
        self, index_bytes: Buffer, buffers: list[Buffer], buffer_prototype: BufferPrototype
    ) -> Buffer:
        """Concatenate the encoded index and data buffers into the shard blob.

        The layout from `_build_shard_layout` already assumes the index size, so
        the encoded index length must match `_shard_index_size` exactly — guard
        that assumption rather than silently corrupt offsets.
        """
        if self.index_location == ShardingCodecIndexLocation.start:
            buffers.insert(0, index_bytes)
        else:
            buffers.append(index_bytes)
        template = buffer_prototype.buffer.create_zero_length()
        return template.combine(buffers)

    def _encode_shard_dict_sync(
        self,
        shard_dict: ShardMapping,
        chunks_per_shard: tuple[int, ...],
        buffer_prototype: BufferPrototype,
    ) -> Buffer | None:
        """Sync version of _encode_shard_dict.

        Layout via the shared `_build_shard_layout` (offsets already absolute),
        then a single index encode and concatenation.

        Returns `None` for an all-empty shard (no chunks present).
        """
        layout = self._build_shard_layout(shard_dict, chunks_per_shard)
        if layout is None:
            return None
        index, buffers = layout
        index_bytes = self._encode_shard_index_sync(index)
        if len(index_bytes) != self._shard_index_size(chunks_per_shard):
            raise RuntimeError(
                "encoded shard index size does not match _shard_index_size; "
                "variable-size index codecs are not supported"
            )
        return self._assemble_shard(index_bytes, buffers, buffer_prototype)

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
        match subchunk_write_order:
            case "morton":
                subchunk_iter = morton_order_iter(chunks_per_shard)
            case "lexicographic":
                subchunk_iter = np.ndindex(chunks_per_shard)
            case "colexicographic":
                subchunk_iter = (c[::-1] for c in np.ndindex(chunks_per_shard[::-1]))
            case "unordered":
                # "unordered" promises no particular layout; today it happens to be
                # lexicographic, but callers must not rely on that.
                subchunk_iter = np.ndindex(chunks_per_shard)
            case _:
                raise ValueError(f"Unrecognized subchunk write order: {subchunk_write_order!r}.")
        return subchunk_iter

    def _decode_full_shard_bulk(
        self,
        shard_bytes: Buffer,
        shard_spec: ArraySpec,
        indexer: Any,
    ) -> NDBuffer | None:
        """Vectorized whole-shard decode for dense, fixed-size, uncompressed shards.

        Returns the assembled shard array, or None if the fast path does not
        apply (so the caller falls back to the per-chunk loop). Conditions:
        - inner codec chain is fixed-size (no compression / variable-length);
        - the inner codec chain is exactly a single BytesCodec — decode is a
          dtype/endian view with no reordering. A trailing crc32c is NOT accepted
          (the bulk path can't verify per-chunk checksums, so crc shards keep the
          per-chunk path's corruption detection);
        - the stored index is dense (every chunk present, equal fixed length,
          contiguous) so the data section is a regular grid of chunk payloads.

        Chunk positions are read from the stored index, so this is correct for
        any `subchunk_write_order` (morton / lexicographic / colexicographic /
        unordered). The on-disk byte order is taken from the BytesCodec's
        `endian`, so big- and little-endian shards both decode correctly.
        """
        # --- gate on a trivial, fixed-size inner codec chain ---
        if not self._inner_codecs_fixed_size:
            return None
        # The inner chain must be exactly a single BytesCodec (a dtype/endian
        # view, no reordering). A trailing Crc32cCodec is excluded on purpose:
        # the bulk path would have to strip-and-discard the per-chunk checksum
        # bytes, silently dropping the corruption detection the per-chunk path
        # enforces (Crc32cCodec._decode_sync raises on mismatch). crc-protected
        # shards therefore fall through to the per-chunk path.
        if len(self.codecs) != 1 or not isinstance(self.codecs[0], BytesCodec):
            return None
        ab_codec = self.codecs[0]

        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)
        n_chunks = product(chunks_per_shard)
        if n_chunks == 0:
            return None

        # Only valid when the read is the ENTIRE shard contiguously (output shape
        # equals the shard shape). A strided/fancy read may touch all chunks but
        # not want the whole grid laid out densely — those must use the per-chunk
        # path so chunk_selection / out_selection are honored.
        if tuple(indexer.shape) != tuple(shard_spec.shape):
            return None
        chunk_byte_length = self._inner_chunk_byte_length(chunk_spec)

        shard_index_size = self._shard_index_size(chunks_per_shard)
        if len(shard_bytes) != n_chunks * chunk_byte_length + shard_index_size:
            return None  # not a dense fixed-size shard

        # --- decode the index; require dense layout ---
        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = shard_bytes[:shard_index_size]
        else:
            index_bytes = shard_bytes[-shard_index_size:]
        index = self._decode_shard_index_sync(index_bytes, chunks_per_shard)
        if not index.is_dense(chunk_byte_length):
            return None

        # --- bulk reconstruct ---
        # The index gives each chunk's absolute byte offset within the blob; with
        # a dense, crc-free, fixed-size layout the payload length is exactly the
        # encoded item-bytes of one chunk.
        native_dtype = shard_spec.dtype.to_native_dtype()
        raw = shard_bytes.as_numpy_array().view(np.uint8)
        payload = chunk_byte_length
        cs = self.chunk_shape

        # On-disk byte order is carried by the BytesCodec's `endian`, NOT by the
        # data type (zarr v3). Build the read-view dtype from the codec's endian
        # exactly as BytesCodec._decode_sync does, so a big-endian shard read on a
        # little-endian host (or vice versa) is interpreted correctly. Assigning
        # into the native-dtype `out` then performs any needed byteswap.
        endian_str = ab_codec.endian.value if ab_codec.endian is not None else None
        if isinstance(chunk_spec.dtype, HasEndianness):
            stored_dtype = replace(chunk_spec.dtype, endianness=endian_str).to_native_dtype()  # type: ignore[call-arg]
        else:
            stored_dtype = chunk_spec.dtype.to_native_dtype()

        offsets = index.offsets_and_lengths[..., 0].reshape(-1)  # localized coords, C-order
        coords_c = list(np.ndindex(chunks_per_shard))
        out = shard_spec.prototype.nd_buffer.empty(
            shape=indexer.shape, dtype=native_dtype, order=shard_spec.order
        )
        for flat, coord in enumerate(coords_c):
            start = int(offsets[flat])
            chunk = raw[start : start + payload].view(stored_dtype).reshape(cs)
            sel = tuple(slice(c * s, c * s + s) for c, s in zip(coord, cs, strict=True))
            out[sel] = chunk
        return out

    def _decode_partial_sync(
        self,
        byte_getter: Any,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Sync equivalent of `_decode_partial_single`.

        Reads only the inner-chunk byte ranges that overlap `selection`
        (plus the shard index) and decodes them through the inner codec
        chain.  The store must support `get_sync` with byte ranges.

        This method does not parallelize decompression, but should.
        See TODO: make issue for handling subchunk parallelism

        Two sub-paths:
        - If `selection` covers the entire shard, just fetch the whole
          blob — that's strictly cheaper than two round trips (index, then
          data) plus the per-chunk overhead of partial fetches.
        - Otherwise fetch the index alone, look up only the byte slices of
          the inner chunks the selection touches, fetch those, and decode.
        """
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)
        inner_transform = self._get_inner_chunk_transform(shard_spec)

        indexer = get_indexer(
            selection,
            shape=shard_shape,
            chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
        )

        out = shard_spec.prototype.nd_buffer.empty(
            shape=indexer.shape,
            dtype=shard_spec.dtype.to_native_dtype(),
            order=shard_spec.order,
        )

        indexed_chunks = list(indexer)
        all_chunk_coords = {chunk_coords for chunk_coords, *_ in indexed_chunks}

        # Read just the inner chunks we need.
        if self._is_total_shard(all_chunk_coords, chunks_per_shard):
            shard_bytes = byte_getter.get_sync(prototype=chunk_spec.prototype)
            if shard_bytes is None:
                return None
            # Bulk fast path: a whole-shard read of a dense, fixed-size shard
            # (no compression/filters) is just the data section reshaped and
            # reordered into the grid -- no per-chunk decode/scatter loop.
            # Order-agnostic: chunk positions come from the stored index, so it
            # is correct for any subchunk_write_order. Falls through on any
            # mismatch (compression, partial shard, non-trivial inner codec).
            bulk = self._decode_full_shard_bulk(shard_bytes, shard_spec, indexer)
            if bulk is not None:
                if hasattr(indexer, "sel_shape"):
                    return bulk.reshape(indexer.sel_shape)
                return bulk
            shard_reader = self._shard_reader_from_bytes_sync(shard_bytes, chunks_per_shard)
            shard_dict: ShardMapping = shard_reader
        else:
            # Partial read: fetch only the touched inner chunks, coalescing
            # adjacent byte ranges (mirrors the async _load_partial_shard_maybe
            # / #3004). Returns None if the shard is absent.
            partial = self._load_partial_shard_maybe_sync(
                byte_getter, chunk_spec.prototype, chunks_per_shard, all_chunk_coords
            )
            if partial is None:
                return None
            shard_dict = partial

        # Decode each needed inner chunk and scatter into out.
        fill_value = shard_spec.fill_value
        if fill_value is None:
            fill_value = shard_spec.dtype.default_scalar()
        for chunk_coords, chunk_selection, out_selection, _ in indexed_chunks:
            try:
                chunk_bytes = shard_dict[chunk_coords]
            except KeyError:
                chunk_bytes = None
            if chunk_bytes is None:
                out[out_selection] = fill_value
                continue
            chunk_array = inner_transform.decode_chunk(chunk_bytes, chunk_spec)
            out[out_selection] = chunk_array[chunk_selection]

        if hasattr(indexer, "sel_shape"):
            return out.reshape(indexer.sel_shape)
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
                chunk_grid=ChunkGrid.from_sizes(shard_shape, chunk_shape),
            )
        )
        # The key order of this intermediate dict is immaterial; the physical layout is
        # decided later by the `subchunk_write_order` loop in `_encode_shard_dict`.
        shard_builder = dict.fromkeys(np.ndindex(chunks_per_shard))

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
            # Intermediate key order is immaterial (see `_encode_single`).
            shard_dict = dict.fromkeys(np.ndindex(chunks_per_shard))
        else:
            shard_reader = await self._load_full_shard_maybe(
                byte_getter=byte_setter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
            shard_reader = shard_reader or _ShardReader.create_empty(chunks_per_shard)
            # Use vectorized lookup for better performance
            shard_dict = shard_reader.to_dict_vectorized(
                np.array(list(np.ndindex(chunks_per_shard)))
            )

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
        """Layout via the shared `_build_shard_layout` (offsets already
        absolute), then a single index encode and concatenation. Async twin of
        `_encode_shard_dict_sync`."""
        layout = self._build_shard_layout(map, chunks_per_shard)
        if layout is None:
            return None
        index, buffers = layout
        index_bytes = await self._encode_shard_index(index)
        if len(index_bytes) != self._shard_index_size(chunks_per_shard):
            raise RuntimeError(
                "encoded shard index size does not match _shard_index_size; "
                "variable-size index codecs are not supported"
            )
        return self._assemble_shard(index_bytes, buffers, buffer_prototype)

    def _is_total_shard(
        self, all_chunk_coords: set[tuple[int, ...]], chunks_per_shard: tuple[int, ...]
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords for chunk_coords in c_order_iter(chunks_per_shard)
        )

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

    async def _load_full_shard_maybe(
        self, byte_getter: ByteGetter, prototype: BufferPrototype, chunks_per_shard: tuple[int, ...]
    ) -> _ShardReader | None:
        shard_bytes = await byte_getter.get(prototype=prototype)

        return (
            await _ShardReader.from_bytes(shard_bytes, self, chunks_per_shard)
            if shard_bytes
            else None
        )

    @property
    def _inner_codecs_fixed_size(self) -> bool:
        """True when all inner codecs produce fixed-size output (no compression)."""
        return all(c.is_fixed_size for c in self.codecs)

    def _inner_chunk_byte_length(self, chunk_spec: ArraySpec) -> int:
        """Encoded byte length of a single inner chunk. Only valid when _inner_codecs_fixed_size."""
        raw_byte_length = 1
        for s in self.chunk_shape:
            raw_byte_length *= s
        raw_byte_length *= chunk_spec.dtype.item_size  # type: ignore[attr-defined]
        return int(self.codec_pipeline.compute_encoded_size(raw_byte_length, chunk_spec))

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

    def _load_shard_index_maybe_sync(
        self, byte_getter: Any, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex | None:
        """Sync counterpart of `_load_shard_index_maybe`."""
        shard_index_size = self._shard_index_size(chunks_per_shard)
        if self.index_location == ShardingCodecIndexLocation.start:
            index_bytes = byte_getter.get_sync(
                prototype=numpy_buffer_prototype(),
                byte_range=RangeByteRequest(0, shard_index_size),
            )
        else:
            index_bytes = byte_getter.get_sync(
                prototype=numpy_buffer_prototype(),
                byte_range=SuffixByteRequest(shard_index_size),
            )
        if index_bytes is not None:
            return self._decode_shard_index_sync(index_bytes, chunks_per_shard)
        return None

    def _load_partial_shard_maybe_sync(
        self,
        byte_getter: Any,
        prototype: BufferPrototype,
        chunks_per_shard: tuple[int, ...],
        all_chunk_coords: set[tuple[int, ...]],
    ) -> ShardMapping | None:
        """Sync counterpart of `_load_partial_shard_maybe` (the #3004 read path).

        Reads the shard index, then fetches only the touched inner chunks via the
        store's coalescing `get_ranges_sync` (merging adjacent ranges into fewer
        reads), matching the async path's IO shape without an event loop.
        """
        shard_index = self._load_shard_index_maybe_sync(byte_getter, chunks_per_shard)
        if shard_index is None:
            return None

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
        store = byte_getter.store if hasattr(byte_getter, "store") else None
        if isinstance(store, Store) and isinstance(store, SupportsGetSync):
            # External store: coalesce via get_ranges_sync (mirrors get_ranges).
            byte_ranges = [byte_range for _, byte_range in chunk_coord_byte_ranges]
            try:
                for idx, buf in store.get_ranges_sync(
                    byte_getter.path, byte_ranges, prototype=prototype
                ):
                    if buf is not None:
                        chunk_coord, _ = chunk_coord_byte_ranges[idx]
                        shard_dict[chunk_coord] = buf
            except BaseExceptionGroup as eg:
                # Mirror the async path: a FileNotFoundError means the shard was
                # deleted mid-read -> treat as "gone" (None). Re-raise anything else.
                _, rest = eg.split(FileNotFoundError)
                if rest is not None:
                    raise rest from None
                return None
        else:
            # Nested sharding: an in-memory _ShardingByteGetter, no IO to coalesce.
            for chunk_coord, byte_range in chunk_coord_byte_ranges:
                buf = byte_getter.get_sync(prototype=prototype, byte_range=byte_range)
                if buf is not None:
                    shard_dict[chunk_coord] = buf

        return shard_dict

    def compute_encoded_size(self, input_byte_length: int, shard_spec: ArraySpec) -> int:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        return input_byte_length + self._shard_index_size(chunks_per_shard)
