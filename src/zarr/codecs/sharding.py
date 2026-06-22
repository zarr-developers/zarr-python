from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import (
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    Codec,
    CodecPipeline,
    SupportsSyncCodec,
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
from zarr.codecs._deprecated_enum import _coerce_enum_input, _DeprecatedStrEnumMeta
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
from zarr.core.chunk_utils import (
    ChunkTransform,
    decode_and_scatter_chunk,
    encode_or_elide_chunk,
    evolve_codecs,
    merge_and_encode_chunk,
)
from zarr.core.common import (
    ShapeLike,
    parse_named_configuration,
    parse_shapelike,
    product,
)
from zarr.core.config import config as zarr_config
from zarr.core.dtype.common import HasEndianness
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
    from typing import Self

    from zarr.core.common import JSON
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

MAX_UINT_64 = 2**64 - 1
ShardMapping = Mapping[tuple[int, ...], Buffer | None]
ShardMutableMapping = MutableMapping[tuple[int, ...], Buffer | None]


IndexLocation = Literal["start", "end"]
"""Position of the shard index within the encoded shard."""

INDEX_LOCATION: Final = ("start", "end")


class ShardingCodecIndexLocation(metaclass=_DeprecatedStrEnumMeta):
    """
    Deprecated. Pass a literal string (`"start"` or `"end"`) directly to
    `ShardingCodec` instead.
    """

    _members: ClassVar[dict[str, str]] = {"start": "start", "end": "end"}


SubchunkWriteOrder = Literal["morton", "unordered", "lexicographic", "colexicographic"]
SUBCHUNK_WRITE_ORDER: Final[tuple[str, str, str, str]] = (
    "morton",
    "unordered",
    "lexicographic",
    "colexicographic",
)


def _parse_index_location(data: object) -> IndexLocation:
    if isinstance(data, str) and data in INDEX_LOCATION:
        return data  # type: ignore[return-value]
    raise ValueError(f"index_location must be one of {list(INDEX_LOCATION)!r}. Got {data!r}.")


@dataclass(frozen=True)
class _ShardingByteGetter(ByteGetter):
    """In-memory byte getter for one inner chunk of a shard.

    Implements `SyncByteGetter` (dict access needs no event loop), so the
    synchronous codec pipeline takes its sync fast path on inner chunks
    instead of scheduling one coroutine per chunk; the async `get` simply
    delegates to `get_sync`.
    """

    shard_dict: ShardMapping
    chunk_coords: tuple[int, ...]

    def get_sync(
        self, prototype: BufferPrototype | None = None, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        assert prototype is None or prototype == default_buffer_prototype(), (
            f"prototype is not supported within shards currently. diff: {prototype} != {default_buffer_prototype()}"
        )
        value = self.shard_dict.get(self.chunk_coords)
        if value is None:
            return None
        if byte_range is None:
            return value
        start, stop = _normalize_byte_range_index(value, byte_range)
        return value[start:stop]

    async def get(
        self, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        return self.get_sync(prototype, byte_range)


@dataclass(frozen=True)
class _ShardingByteSetter(_ShardingByteGetter, ByteSetter):
    """In-memory byte setter for one inner chunk of a shard.

    Implements `SyncByteSetter`; the async methods delegate to the sync ones.
    """

    shard_dict: ShardMutableMapping

    def set_sync(self, value: Buffer) -> None:
        self.shard_dict[self.chunk_coords] = value

    def delete_sync(self) -> None:
        del self.shard_dict[self.chunk_coords]

    async def set(self, value: Buffer, byte_range: ByteRequest | None = None) -> None:
        assert byte_range is None, "byte_range is not supported within shards"
        self.set_sync(value)

    async def delete(self) -> None:
        self.delete_sync()

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
        if codec.index_location == "start":
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

    is_fixed_size = False

    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    index_location: IndexLocation = "end"
    subchunk_write_order: SubchunkWriteOrder = "morton"

    def __init__(
        self,
        *,
        chunk_shape: ShapeLike,
        codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(),),
        index_codecs: Iterable[Codec | dict[str, JSON]] = (BytesCodec(), Crc32cCodec()),
        index_location: ShardingCodecIndexLocation | IndexLocation = "end",
        subchunk_write_order: SubchunkWriteOrder = "morton",
    ) -> None:
        chunk_shape_parsed = parse_shapelike(chunk_shape)
        codecs_parsed = parse_codecs(codecs)
        index_codecs_parsed = parse_codecs(index_codecs)
        index_location_coerced = _coerce_enum_input(
            index_location, "index_location", "ShardingCodec"
        )
        index_location_parsed = _parse_index_location(index_location_coerced)
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
        object.__setattr__(self, "_shard_index_size", lru_cache()(self._shard_index_size))
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
        object.__setattr__(self, "index_location", _parse_index_location(config["index_location"]))
        object.__setattr__(self, "subchunk_write_order", state["subchunk_write_order"])

        # Use instance-local lru_cache to avoid memory leaks
        # object.__setattr__(self, "_get_chunk_spec", lru_cache()(self._get_chunk_spec))
        object.__setattr__(self, "_get_index_chunk_spec", lru_cache()(self._get_index_chunk_spec))
        object.__setattr__(self, "_get_chunks_per_shard", lru_cache()(self._get_chunks_per_shard))
        object.__setattr__(self, "_shard_index_size", lru_cache()(self._shard_index_size))
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

    def _get_inner_pipeline(self, shard_spec: ArraySpec) -> CodecPipeline:
        """The nested pipeline for inner-chunk IO, evolved against the inner
        chunk spec.

        Evolving matters for two reasons: it threads the spec through the inner
        codec chain (spec-changing codecs see the spec they will actually
        operate on), and — for a synchronous pipeline — it builds the sync
        transform, so inner-chunk IO over the (sync-capable) sharding byte
        getters takes the pipeline's sync fast path instead of scheduling one
        coroutine per inner chunk. The bare `codec_pipeline` property returns an
        unevolved pipeline, which a synchronous pipeline can only run through
        its async fallback.

        Memoized per (pipeline class, batch size, shard_spec): evolving builds
        a ChunkTransform, which is wasteful to redo on every shard operation.
        The pipeline class and batch size participate in the key so the
        `codec_pipeline.path` and `codec_pipeline.batch_size` configs are still
        honored after the first use (`from_codecs` captures batch_size at
        construction). A benign construction race between threads is possible
        (last writer wins) — same as the other caches here.
        """
        cache: dict[tuple[type[CodecPipeline], int, ArraySpec], CodecPipeline] | None = getattr(
            self, "_inner_pipeline_cache", None
        )
        if cache is None:
            cache = {}
            object.__setattr__(self, "_inner_pipeline_cache", cache)
        key = (get_pipeline_class(), zarr_config.get("codec_pipeline.batch_size"), shard_spec)
        pipeline = cache.get(key)
        if pipeline is None:
            chunk_spec = self._get_chunk_spec(shard_spec)
            pipeline = self.codec_pipeline.evolve_from_array_spec(chunk_spec)
            cache[key] = pipeline
        return pipeline

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": self.chunk_shape,
                "codecs": tuple(s.to_dict() for s in self.codecs),
                "index_codecs": tuple(s.to_dict() for s in self.index_codecs),
                "index_location": self.index_location,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """Thread the spec through the inner chain.

        Each codec is evolved against the spec produced by the previous one.
        Evolving every codec against the same unthreaded spec is the bug shape that
        strips `BytesCodec.endian` behind a dtype-changing codec — and this
        method runs on the real array-creation path, baking the damaged chain
        into the evolved instance before the transform builders ever run.

        Parameters
        ----------
        array_spec
            The base spec to be evolved as we thread it.

        Returns
        -------
            This codec with the evolved code chain.
        """
        from zarr.core.chunk_utils import evolve_codecs

        shard_spec = self._get_chunk_spec(array_spec)
        evolved_codecs = evolve_codecs(self.codecs, shard_spec)
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
        """The synchronous transform for the inner codec chain.

        Memoized by the instance-local `lru_cache` wrapping installed in
        `__init__`/`__setstate__` (the single cache mechanism for these
        builders — do not add another layer inside the body).

        Codecs are evolved with the spec THREADED forward (`evolve_codecs`):
        each inner codec is evolved against the spec produced by the previous
        one, not the original chunk spec. Evolving every codec against the same
        unthreaded spec is the bug shape that stripped `BytesCodec.endian` at
        the pipeline level (see `evolve_codecs`) — the inner chain must use the
        same single source of truth.
        """

        chunk_spec = self._get_chunk_spec(shard_spec)
        return ChunkTransform(codecs=evolve_codecs(self.codecs, chunk_spec))

    def _get_index_chunk_transform(self, chunks_per_shard: tuple[int, ...]) -> Any:
        """The synchronous transform for the index codec chain.

        Memoized via instance-local `lru_cache`.
        """

        index_spec = self._get_index_chunk_spec(chunks_per_shard)
        return ChunkTransform(codecs=evolve_codecs(self.index_codecs, index_spec))

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
        if self.index_location == "start":
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
            # the GetResult status is discarded: missing INNER chunks of a
            # present shard always fill (read_missing_chunks is a store-key
            # level promise, applied to top-level statuses at the array layer)
            decode_and_scatter_chunk(
                shard_dict.get(chunk_coords),
                out,
                chunk_spec=chunk_spec,
                chunk_selection=chunk_selection,
                out_selection=out_selection,
                drop_axes=(),
                decode=inner_transform.decode_chunk,
            )

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
            lexicographic_order_coords(chunks_per_shard)
        )

        for chunk_coords, _chunk_selection, out_selection, _ in indexer:
            # None = chunk normalized to missing (see encode_or_elide_chunk)
            shard_builder[chunk_coords] = encode_or_elide_chunk(
                shard_array[out_selection], chunk_spec, inner_transform.encode_chunk
            )

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

        is_scalar = len(value.shape) == 0

        # Load existing inner-chunk bytes into a dict (same structure as
        # the async path's shard_dict).
        if is_complete:
            shard_dict: dict[tuple[int, ...], Buffer | None] = dict.fromkeys(
                lexicographic_order_coords(chunks_per_shard)
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
                # partial writes into shards with many inner chunks. The coordinate
                # array and keys are cached on the reader, so neither is rebuilt here.
                shard_dict = shard_reader_fb.to_dict_vectorized()
            else:
                shard_dict = dict.fromkeys(lexicographic_order_coords(chunks_per_shard))

        # Merge, encode, and store each affected inner chunk into shard_dict via
        # the canonical merge_and_encode_chunk (None = normalized to missing).
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
                    scalar_complete_result = merge_and_encode_chunk(
                        None,
                        value,
                        chunk_spec=chunk_spec,
                        chunk_selection=chunk_sel,
                        out_selection=out_sel,
                        is_complete=is_complete_chunk,
                        drop_axes=(),
                        decode=inner_transform.decode_chunk,
                        encode=inner_transform.encode_chunk,
                    )
                shard_dict[chunk_coords] = scalar_complete_result  # type: ignore[assignment]
                continue

            # A complete chunk fully overwrites: skip decoding what it replaces.
            existing_raw = None if is_complete_chunk else shard_dict.get(chunk_coords)
            shard_dict[chunk_coords] = merge_and_encode_chunk(
                existing_raw,
                value,
                chunk_spec=chunk_spec,
                chunk_selection=chunk_sel,
                out_selection=out_sel,
                is_complete=is_complete_chunk,
                drop_axes=(),
                decode=inner_transform.decode_chunk,
                encode=inner_transform.encode_chunk,
            )

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
            self._shard_index_size(chunks_per_shard) if self.index_location == "start" else 0
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
        self,
        index_bytes: Buffer,
        buffers: list[Buffer],
        buffer_prototype: BufferPrototype,
        *,
        chunks_per_shard: tuple[int, ...],
    ) -> Buffer:
        """Concatenate the encoded index and data buffers into the shard blob.

        The layout from `_build_shard_layout` already assumes the index size, so
        the encoded index length must match `_shard_index_size` exactly — guard
        that assumption rather than silently corrupt offsets. The guard lives
        here, once, for both the sync and async encode paths.
        """
        if len(index_bytes) != self._shard_index_size(chunks_per_shard):
            raise RuntimeError(
                "encoded shard index size does not match _shard_index_size; "
                "variable-size index codecs are not supported"
            )
        if self.index_location == "start":
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
        return self._assemble_shard(
            index_bytes, buffers, buffer_prototype, chunks_per_shard=chunks_per_shard
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
        await self._get_inner_pipeline(shard_spec).read(
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
                max_gap_bytes=shard_spec.config.sharding_coalesce_max_gap_bytes,
                max_coalesced_bytes=shard_spec.config.sharding_coalesce_max_bytes,
            )

        if shard_dict_maybe is None:
            return None
        shard_dict = shard_dict_maybe

        # decoding chunks and writing them into the output buffer
        await self._get_inner_pipeline(shard_spec).read(
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

    def _decode_full_shard_bulk_if_uncompressed(
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

        # Only valid for a plain contiguous full-shard read, where each chunk
        # lands at its natural grid position. The `sel_shape` check is
        # load-bearing: a gather indexer (CoordinateIndexer, from vindex / an
        # oindex with an integer-array selection) reorders points and exposes
        # `sel_shape`, but its `.shape` is the FLATTENED point count, which can
        # equal the shard shape by coincidence (trivially in 1-D). Gating on
        # shape alone lets such a selection through, and the bulk path then
        # returns the shard in natural order, silently dropping the reordering.
        # A contiguous full read (BasicIndexer, or a non-gathering
        # OrthogonalIndexer from `arr[:]`) has no `sel_shape` and is served here.
        # Anything that gathers must fall through to the per-chunk path so
        # chunk_selection / out_selection are honored.
        if getattr(indexer, "sel_shape", None) is not None:
            return None
        if tuple(indexer.shape) != tuple(shard_spec.shape):
            return None
        chunk_byte_length = self._inner_chunk_byte_length(chunk_spec)

        shard_index_size = self._shard_index_size(chunks_per_shard)
        if len(shard_bytes) != n_chunks * chunk_byte_length + shard_index_size:
            return None  # not a dense fixed-size shard

        # --- decode the index; require dense layout ---
        if self.index_location == "start":
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
        # into the native-dtype `out` then performs any needed byteswap. `endian`
        # is now a plain Literal['little', 'big'] | None string (no longer an enum).
        endian_str = ab_codec.endian
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
            bulk = self._decode_full_shard_bulk_if_uncompressed(shard_bytes, shard_spec, indexer)
            if bulk is not None:
                # The bulk path only fires for a contiguous full-shard read (it
                # returns None for any gather indexer that exposes `sel_shape`),
                # so the result is already shard-shaped — no reshape needed.
                return bulk
            shard_reader = self._shard_reader_from_bytes_sync(shard_bytes, chunks_per_shard)
            shard_dict: ShardMapping = shard_reader
        else:
            # Partial read: fetch only the touched inner chunks, coalescing
            # adjacent byte ranges (mirrors the async _load_partial_shard_maybe
            # / #3004). Returns None if the shard is absent.
            partial = self._load_partial_shard_maybe_sync(
                byte_getter,
                chunk_spec.prototype,
                chunks_per_shard,
                all_chunk_coords,
                max_gap_bytes=shard_spec.config.sharding_coalesce_max_gap_bytes,
                max_coalesced_bytes=shard_spec.config.sharding_coalesce_max_bytes,
            )
            if partial is None:
                return None
            shard_dict = partial

        # Decode each needed inner chunk and scatter into out (statuses
        # discarded: missing inner chunks fill, see _decode_sync).
        for chunk_coords, chunk_selection, out_selection, _ in indexed_chunks:
            decode_and_scatter_chunk(
                shard_dict.get(chunk_coords),
                out,
                chunk_spec=chunk_spec,
                chunk_selection=chunk_selection,
                out_selection=out_selection,
                drop_axes=(),
                decode=inner_transform.decode_chunk,
            )

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
        shard_builder = dict.fromkeys(lexicographic_order_coords(chunks_per_shard))

        await self._get_inner_pipeline(shard_spec).write(
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

        await self._get_inner_pipeline(shard_spec).write(
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
        return self._assemble_shard(
            index_bytes, buffers, buffer_prototype, chunks_per_shard=chunks_per_shard
        )

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

    def _index_codecs_sync_capable(self) -> bool:
        return all(isinstance(c, SupportsSyncCodec) for c in self.index_codecs)

    async def _decode_shard_index(
        self, index_bytes: Buffer, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex:
        # Pure compute (the bytes are already in hand): delegate to the sync
        # implementation instead of spinning up a pipeline + per-call
        # AsyncChunkTransform for a tiny fixed-size decode. The default
        # (bytes + crc32c) index chain is sync-capable; an async-only
        # third-party index codec falls back to the full async pipeline, which
        # the synchronous read paths cannot use but this async path still can.
        if self._index_codecs_sync_capable():
            return self._decode_shard_index_sync(index_bytes, chunks_per_shard)
        index_array = next(
            iter(
                await get_pipeline_class()
                .from_codecs(self.index_codecs)
                .decode([(index_bytes, self._get_index_chunk_spec(chunks_per_shard))])
            )
        )
        assert index_array is not None  # the bytes are already in hand
        return _ShardIndex(chunks_per_shard, index_array.as_numpy_array())

    async def _encode_shard_index(self, index: _ShardIndex) -> Buffer:
        # Pure compute: delegate to the sync implementation, with the same
        # async-pipeline fallback as _decode_shard_index.
        if self._index_codecs_sync_capable():
            return self._encode_shard_index_sync(index)
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
                    ]
                )
            )
        )
        assert index_bytes is not None
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

    def _shard_index_byte_range(
        self, chunks_per_shard: tuple[int, ...]
    ) -> RangeByteRequest | SuffixByteRequest:
        """Byte range of the shard index within the shard blob.

        Single source of truth for the index-location arithmetic, shared by the
        sync and async index loaders so they cannot drift.
        """
        shard_index_size = self._shard_index_size(chunks_per_shard)
        if self.index_location == "start":
            return RangeByteRequest(0, shard_index_size)
        return SuffixByteRequest(shard_index_size)

    @staticmethod
    def _pair_chunks_with_byte_ranges(
        shard_index: _ShardIndex, all_chunk_coords: set[tuple[int, ...]]
    ) -> list[tuple[tuple[int, ...], RangeByteRequest]]:
        """Pair each requested chunk coord with its byte range in the shard.

        Coords whose chunk is absent from the index are omitted. Shared by the
        sync and async partial-shard loaders.
        """
        chunk_coord_byte_ranges: list[tuple[tuple[int, ...], RangeByteRequest]] = []
        for chunk_coord in all_chunk_coords:
            chunk_byte_slice = shard_index.get_chunk_slice(chunk_coord)
            if chunk_byte_slice is not None:
                chunk_coord_byte_ranges.append(
                    (chunk_coord, RangeByteRequest(chunk_byte_slice[0], chunk_byte_slice[1]))
                )
        return chunk_coord_byte_ranges

    async def _load_shard_index_maybe(
        self, byte_getter: ByteGetter, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex | None:
        index_bytes = await byte_getter.get(
            prototype=numpy_buffer_prototype(),
            byte_range=self._shard_index_byte_range(chunks_per_shard),
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
        max_gap_bytes: int,
        max_coalesced_bytes: int,
    ) -> ShardMapping | None:
        """
        Read chunks from `byte_getter` for the case where the read is less than a full shard.
        Returns a mapping of chunk coordinates to bytes or None.

        `max_gap_bytes` and `max_coalesced_bytes` are forwarded to
        `Store.get_ranges` to control byte-range coalescing across the requested
        chunks.
        """
        shard_index = await self._load_shard_index_maybe(byte_getter, chunks_per_shard)
        if shard_index is None:
            return None

        chunk_coord_byte_ranges = self._pair_chunks_with_byte_ranges(shard_index, all_chunk_coords)

        if not chunk_coord_byte_ranges:
            return {}

        shard_dict: ShardMutableMapping = {}
        if isinstance(byte_getter, StorePath):
            # External store: use Store.get_ranges for coalescing + concurrency.
            byte_ranges = [byte_range for _, byte_range in chunk_coord_byte_ranges]
            try:
                async for group in byte_getter.store.get_ranges(
                    byte_getter.path,
                    byte_ranges,
                    prototype=prototype,
                    max_gap_bytes=max_gap_bytes,
                    max_coalesced_bytes=max_coalesced_bytes,
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
        index_bytes = byte_getter.get_sync(
            prototype=numpy_buffer_prototype(),
            byte_range=self._shard_index_byte_range(chunks_per_shard),
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
        *,
        max_gap_bytes: int,
        max_coalesced_bytes: int,
    ) -> ShardMapping | None:
        """Sync counterpart of `_load_partial_shard_maybe` (the #3004 read path).

        Reads the shard index, then fetches only the touched inner chunks via the
        store's coalescing `get_ranges_sync` (merging adjacent ranges into fewer
        reads), matching the async path's IO shape without an event loop.
        `max_gap_bytes` and `max_coalesced_bytes` control the coalescing, forwarded
        from the array's `sharding_coalesce_*` config exactly as the async path.
        """
        shard_index = self._load_shard_index_maybe_sync(byte_getter, chunks_per_shard)
        if shard_index is None:
            return None

        chunk_coord_byte_ranges = self._pair_chunks_with_byte_ranges(shard_index, all_chunk_coords)

        if not chunk_coord_byte_ranges:
            return {}

        shard_dict: ShardMutableMapping = {}
        store = byte_getter.store if hasattr(byte_getter, "store") else None
        if isinstance(store, Store) and isinstance(store, SupportsGetSync):
            # External store: coalesce via get_ranges_sync (mirrors get_ranges).
            byte_ranges = [byte_range for _, byte_range in chunk_coord_byte_ranges]
            try:
                for idx, buf in store.get_ranges_sync(
                    byte_getter.path,
                    byte_ranges,
                    prototype=prototype,
                    max_gap_bytes=max_gap_bytes,
                    max_coalesced_bytes=max_coalesced_bytes,
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
