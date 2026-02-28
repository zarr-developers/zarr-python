from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from operator import itemgetter
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import (
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    Codec,
)
from zarr.abc.store import (
    ByteGetter,
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
from zarr.core.common import (
    ShapeLike,
    parse_enum,
    parse_named_configuration,
    parse_shapelike,
    product,
)
from zarr.core.dtype.common import HasItemSize
from zarr.core.dtype.npy.int import UInt64
from zarr.core.indexing import (
    BasicIndexer,
    ChunkProjection,
    SelectorTuple,
    _morton_order_keys,
    c_order_iter,
    get_indexer,
    morton_order_iter,
)
from zarr.core.metadata.v3 import parse_codecs
from zarr.registry import get_ndbuffer_class

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    from zarr.abc.codec import PreparedWrite
    from zarr.core.codec_pipeline import CodecChain
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
    def from_bytes_sync(
        cls, buf: Buffer, codec: ShardingCodec, chunks_per_shard: tuple[int, ...]
    ) -> _ShardReader:
        """Synchronous version of from_bytes — decodes the shard index inline."""
        shard_index_size = codec._shard_index_size(chunks_per_shard)
        obj = cls()
        obj.buf = buf
        if codec.index_location == ShardingCodecIndexLocation.start:
            shard_index_bytes = obj.buf[:shard_index_size]
        else:
            shard_index_bytes = obj.buf[-shard_index_size:]

        obj.index = codec._decode_shard_index_sync(shard_index_bytes, chunks_per_shard)
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
class ShardingCodec(ArrayBytesCodec, ArrayBytesCodecPartialDecodeMixin):
    """Sharding codec"""

    chunk_shape: tuple[int, ...]
    codecs: tuple[Codec, ...]
    index_codecs: tuple[Codec, ...]
    index_location: ShardingCodecIndexLocation = ShardingCodecIndexLocation.end
    _codec_chain: CodecChain = field(init=False, repr=False, compare=False)
    _index_codec_chain: CodecChain = field(init=False, repr=False, compare=False)

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

        # Cached CodecChain instances — computed once, used for all sync paths.
        from zarr.core.codec_pipeline import CodecChain

        object.__setattr__(self, "_codec_chain", CodecChain.from_codecs(codecs_parsed))
        object.__setattr__(self, "_index_codec_chain", CodecChain.from_codecs(index_codecs_parsed))

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

        from zarr.core.codec_pipeline import CodecChain

        object.__setattr__(
            self, "_codec_chain", CodecChain.from_codecs(parse_codecs(config["codecs"]))
        )
        object.__setattr__(
            self,
            "_index_codec_chain",
            CodecChain.from_codecs(parse_codecs(config["index_codecs"])),
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "sharding_indexed")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

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
        # _decode_single is pure compute (no IO), same as _decode_sync.
        return self._decode_sync(shard_bytes, shard_spec)

    def _decode_sync(
        self,
        shard_bytes: Buffer,
        shard_spec: ArraySpec,
    ) -> NDBuffer:
        """Synchronous full-shard decode.

        Uses deserialize() to unpack stored bytes into per-inner-chunk buffers,
        then decodes each inner chunk via CodecChain.decode_chunk (pure compute).
        """
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunk_spec = self._get_chunk_spec(shard_spec)

        chunk_dict = self.deserialize(shard_bytes, shard_spec)

        # Check if all chunks are empty
        if all(v is None for v in chunk_dict.values()):
            out = chunk_spec.prototype.nd_buffer.empty(
                shape=shard_shape,
                dtype=shard_spec.dtype.to_native_dtype(),
                order=shard_spec.order,
            )
            out.fill(shard_spec.fill_value)
            return out

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

        # Pre-resolve metadata chain once for all inner chunks.
        codec_chain = self._codec_chain
        aa_chain, ab_pair, bb_chain = codec_chain.resolve_metadata_chain(chunk_spec)

        # Decode each inner chunk directly via CodecChain (pure compute).
        from zarr.core.codec_pipeline import fill_value_or_default

        fill_value = fill_value_or_default(shard_spec)
        for chunk_coords, chunk_selection, out_selection, _is_complete in indexer:
            chunk_bytes = chunk_dict.get(chunk_coords)
            if chunk_bytes is not None:
                chunk_array = codec_chain.decode_chunk(
                    chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain
                )
                if chunk_array is not None:
                    out[out_selection] = chunk_array[chunk_selection]
                else:
                    out[out_selection] = fill_value
            else:
                out[out_selection] = fill_value

        return out

    def _encode_sync(
        self,
        shard_array: NDBuffer,
        shard_spec: ArraySpec,
    ) -> Buffer | None:
        """Synchronous full-shard encode.

        Encodes each inner chunk via CodecChain.encode_chunk (pure compute),
        then assembles the shard via serialize().
        """
        shard_shape = shard_spec.shape
        chunk_shape = self.chunk_shape
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        chunk_spec = self._get_chunk_spec(shard_spec)
        skip_empty = not chunk_spec.config.write_empty_chunks

        if skip_empty:
            from zarr.core.codec_pipeline import fill_value_or_default

            fill_value = fill_value_or_default(chunk_spec)

            # Quick check: if the entire shard equals fill value, return None.
            if shard_array.all_equal(fill_value):
                return None

        # Fast path: vectorized encoding for fixed-size inner codecs (no compression).
        # Reorders chunks from C-order to morton order using numpy operations,
        # avoiding per-chunk Python function calls entirely.
        if self._inner_codecs_fixed_size:
            result = self._encode_vectorized(
                shard_array, shard_spec, chunks_per_shard, chunk_shape, chunk_spec, skip_empty
            )
            if result is not None:
                return result
            # result is None means either:
            # 1. All chunks are fill-value (skip_empty=True) → return None
            # 2. Vectorized path not applicable → fall through to per-chunk loop
            if skip_empty:
                return None

        # Slow path: per-chunk encode loop.
        indexer = list(
            BasicIndexer(
                tuple(slice(0, s) for s in shard_shape),
                shape=shard_shape,
                chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            )
        )

        shard_builder: dict[tuple[int, ...], Buffer | None] = dict.fromkeys(
            morton_order_iter(chunks_per_shard)
        )

        codec_chain = self._codec_chain

        if skip_empty:
            from zarr.core.codec_pipeline import fill_value_or_default

            fill_value = fill_value_or_default(chunk_spec)

            # Vectorized per-chunk fill-value check.
            shard_np = shard_array.as_numpy_array()
            ndim = len(shard_shape)
            reshaped_dims = []
            for s_dim, c_dim in zip(chunks_per_shard, chunk_shape, strict=False):
                reshaped_dims.extend([s_dim, c_dim])
            try:
                shard_reshaped = shard_np.reshape(reshaped_dims)
                chunk_interior_axes = tuple(range(1, 2 * ndim, 2))
                if np.isnan(fill_value) if isinstance(fill_value, float) else False:
                    is_fill = np.all(np.isnan(shard_reshaped), axis=chunk_interior_axes)
                else:
                    is_fill = np.all(shard_reshaped == fill_value, axis=chunk_interior_axes)
            except (ValueError, AttributeError):
                is_fill = None

            for chunk_coords, _chunk_selection, out_selection, _is_complete in indexer:
                chunk_array = shard_array[out_selection]
                if chunk_array is not None:
                    if is_fill is not None:
                        chunk_is_fill = bool(is_fill[chunk_coords])
                    else:
                        chunk_is_fill = chunk_array.all_equal(fill_value)
                    if chunk_is_fill:
                        shard_builder[chunk_coords] = None
                    else:
                        shard_builder[chunk_coords] = codec_chain.encode_chunk(
                            chunk_array, chunk_spec
                        )
        else:
            for chunk_coords, _chunk_selection, out_selection, _is_complete in indexer:
                chunk_array = shard_array[out_selection]
                if chunk_array is not None:
                    shard_builder[chunk_coords] = codec_chain.encode_chunk(chunk_array, chunk_spec)

        return self.serialize(shard_builder, shard_spec)

    def _encode_vectorized(
        self,
        shard_array: NDBuffer,
        shard_spec: ArraySpec,
        chunks_per_shard: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        chunk_spec: ArraySpec,
        skip_empty: bool,
    ) -> Buffer | None:
        """Vectorized shard encoding for fixed-size inner codecs.

        Reorders chunks from C-order to morton order using numpy operations,
        building the entire shard blob without per-chunk Python function calls.
        Returns None if all chunks are fill-value, or if the fast path is not
        applicable (caller should fall through to per-chunk loop).
        """
        from zarr.core.indexing import _morton_order

        shard_np = shard_array.as_numpy_array()
        ndim = len(chunks_per_shard)
        total_chunks = product(chunks_per_shard)

        # Handle endianness at the shard level (BytesCodec normally does this per-chunk)
        ab_codec = self._codec_chain.array_bytes_codec
        if (
            isinstance(ab_codec, BytesCodec)
            and shard_np.dtype.itemsize > 1
            and ab_codec.endian is not None
            and ab_codec.endian != shard_array.byteorder
        ):
            new_dtype = shard_np.dtype.newbyteorder(ab_codec.endian.name)  # type: ignore[arg-type]
            shard_np = shard_np.astype(new_dtype)

        # Reshape: (shard_shape) → (cps[0], cs[0], cps[1], cs[1], ...)
        reshaped_dims: list[int] = []
        for cps, cs in zip(chunks_per_shard, chunk_shape, strict=False):
            reshaped_dims.extend([cps, cs])

        shard_reshaped = shard_np.reshape(reshaped_dims)

        if skip_empty:
            from zarr.core.codec_pipeline import fill_value_or_default

            fill_value = fill_value_or_default(chunk_spec)
            chunk_interior_axes = tuple(range(1, 2 * ndim, 2))
            if np.isnan(fill_value) if isinstance(fill_value, float) else False:
                is_fill = np.all(np.isnan(shard_reshaped), axis=chunk_interior_axes)
            else:
                is_fill = np.all(shard_reshaped == fill_value, axis=chunk_interior_axes)

            if np.all(is_fill):
                return None
            if np.any(is_fill):
                # Some chunks are fill-value, some are not.
                # Fall through to per-chunk loop for this mixed case.
                # Return a sentinel that the caller interprets as "not applicable".
                # We use a special approach: return _MIXED_FILL sentinel
                return self._encode_vectorized_sparse(
                    shard_np,
                    shard_spec,
                    chunks_per_shard,
                    chunk_shape,
                    chunk_spec,
                    shard_reshaped,
                    is_fill,
                )

        # Transpose to (cps[0], cps[1], ..., cs[0], cs[1], ...)
        chunk_grid_axes = tuple(range(0, 2 * ndim, 2))
        chunk_data_axes = tuple(range(1, 2 * ndim, 2))
        transposed = shard_reshaped.transpose(chunk_grid_axes + chunk_data_axes)

        # Reshape to (total_chunks, elements_per_chunk), then reorder to morton
        elements_per_chunk = product(chunk_shape)
        chunks_2d = transposed.reshape(total_chunks, elements_per_chunk)

        # Reorder from C-order to morton order
        morton_coords = _morton_order(chunks_per_shard)  # (total_chunks, ndim)
        c_order_linear = np.ravel_multi_index(
            tuple(morton_coords[:, i] for i in range(ndim)), chunks_per_shard
        )
        reordered = chunks_2d[c_order_linear]

        # Flatten to bytes
        chunk_data_bytes = reordered.ravel().view(np.uint8)

        # Build deterministic shard index (all chunks present, each chunk_byte_length)
        index = _ShardIndex.create_empty(chunks_per_shard)
        encoded_chunk_byte_length = self._inner_chunk_byte_length(chunk_spec)
        for rank in range(total_chunks):
            offset = rank * encoded_chunk_byte_length
            chunk_coords = tuple(int(x) for x in morton_coords[rank])
            index.set_chunk_slice(chunk_coords, slice(offset, offset + encoded_chunk_byte_length))

        index_bytes = self._encode_shard_index_sync(index)

        if self.index_location == ShardingCodecIndexLocation.start:
            # Shift non-empty offsets by index size
            non_empty = index.offsets_and_lengths[..., 0] != MAX_UINT_64
            index.offsets_and_lengths[non_empty, 0] += len(index_bytes)
            index_bytes = self._encode_shard_index_sync(index)
            shard_bytes_np = np.concatenate(
                [
                    np.frombuffer(index_bytes.as_buffer_like(), dtype=np.uint8),
                    chunk_data_bytes,
                ]
            )
        else:
            shard_bytes_np = np.concatenate(
                [
                    chunk_data_bytes,
                    np.frombuffer(index_bytes.as_buffer_like(), dtype=np.uint8),
                ]
            )

        return default_buffer_prototype().buffer.from_array_like(shard_bytes_np)

    def _encode_vectorized_sparse(
        self,
        shard_np: npt.NDArray[Any],
        shard_spec: ArraySpec,
        chunks_per_shard: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        chunk_spec: ArraySpec,
        shard_reshaped: npt.NDArray[Any],
        is_fill: npt.NDArray[np.bool_],
    ) -> Buffer | None:
        """Vectorized encoding when some chunks are fill-value (sparse shard).

        Builds the shard blob with only non-fill chunks present.
        """
        from zarr.core.indexing import _morton_order

        ndim = len(chunks_per_shard)
        total_chunks = product(chunks_per_shard)

        # Transpose to (cps[0], cps[1], ..., cs[0], cs[1], ...)
        chunk_grid_axes = tuple(range(0, 2 * ndim, 2))
        chunk_data_axes = tuple(range(1, 2 * ndim, 2))
        transposed = shard_reshaped.transpose(chunk_grid_axes + chunk_data_axes)

        elements_per_chunk = product(chunk_shape)
        chunks_2d = transposed.reshape(total_chunks, elements_per_chunk)

        # Reorder from C-order to morton order
        morton_coords = _morton_order(chunks_per_shard)  # (total_chunks, ndim)
        c_order_linear = np.ravel_multi_index(
            tuple(morton_coords[:, i] for i in range(ndim)), chunks_per_shard
        )
        reordered = chunks_2d[c_order_linear]

        # is_fill is in C-order grid shape, flatten to C-order linear
        is_fill_morton = is_fill.ravel()[c_order_linear]

        # Build index and collect non-fill chunk data
        index = _ShardIndex.create_empty(chunks_per_shard)
        encoded_chunk_byte_length = self._inner_chunk_byte_length(chunk_spec)

        # Select only non-fill chunks
        non_fill_mask = ~is_fill_morton
        non_fill_data = reordered[non_fill_mask]

        if len(non_fill_data) == 0:
            return None

        # Build index: set offsets for non-fill chunks using morton coordinates
        offset = 0
        for rank in range(total_chunks):
            if non_fill_mask[rank]:
                chunk_coords = tuple(int(x) for x in morton_coords[rank])
                index.set_chunk_slice(
                    chunk_coords, slice(offset, offset + encoded_chunk_byte_length)
                )
                offset += encoded_chunk_byte_length

        index_bytes = self._encode_shard_index_sync(index)

        chunk_data_bytes = non_fill_data.ravel().view(np.uint8)

        if self.index_location == ShardingCodecIndexLocation.start:
            non_empty = index.offsets_and_lengths[..., 0] != MAX_UINT_64
            index.offsets_and_lengths[non_empty, 0] += len(index_bytes)
            index_bytes = self._encode_shard_index_sync(index)
            shard_bytes_np = np.concatenate(
                [
                    np.frombuffer(index_bytes.as_buffer_like(), dtype=np.uint8),
                    chunk_data_bytes,
                ]
            )
        else:
            shard_bytes_np = np.concatenate(
                [
                    chunk_data_bytes,
                    np.frombuffer(index_bytes.as_buffer_like(), dtype=np.uint8),
                ]
            )

        return default_buffer_prototype().buffer.from_array_like(shard_bytes_np)

    def _encode_shard_dict_sync(
        self,
        map: ShardMapping,
        chunks_per_shard: tuple[int, ...],
        buffer_prototype: BufferPrototype,
    ) -> Buffer | None:
        """Serialize encoded chunks into a shard blob with index."""
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

        index_bytes = self._encode_shard_index_sync(index)
        if self.index_location == ShardingCodecIndexLocation.start:
            empty_chunks_mask = index.offsets_and_lengths[..., 0] == MAX_UINT_64
            index.offsets_and_lengths[~empty_chunks_mask, 0] += len(index_bytes)
            index_bytes = self._encode_shard_index_sync(
                index
            )  # encode again with corrected offsets
            buffers.insert(0, index_bytes)
        else:
            buffers.append(index_bytes)

        return template.combine(buffers)

    def _load_shard_index_maybe_sync(
        self, byte_getter: Any, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex | None:
        """Synchronous version of _load_shard_index_maybe.

        Reads the shard index bytes via byte_getter.get_sync (a sync byte-range
        read from the store), then decodes the index inline.
        """
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

    def _load_full_shard_maybe_sync(
        self,
        byte_getter: Any,
        prototype: BufferPrototype,
        chunks_per_shard: tuple[int, ...],
    ) -> _ShardReader | None:
        """Synchronous version of _load_full_shard_maybe."""
        shard_bytes = byte_getter.get_sync(prototype=prototype)
        return (
            _ShardReader.from_bytes_sync(shard_bytes, self, chunks_per_shard)
            if shard_bytes
            else None
        )

    def _decode_partial_sync(
        self,
        byte_getter: Any,
        selection: SelectorTuple,
        shard_spec: ArraySpec,
    ) -> NDBuffer | None:
        """Synchronous partial decode: fetch shard index + requested chunks
        via sync byte-range reads, then decode via CodecChain (pure compute).
        """
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
            shard_dict_maybe = self._load_full_shard_maybe_sync(
                byte_getter=byte_getter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
            if shard_dict_maybe is None:
                return None
            shard_dict = shard_dict_maybe
        else:
            # read some chunks within the shard
            shard_index = self._load_shard_index_maybe_sync(byte_getter, chunks_per_shard)
            if shard_index is None:
                return None
            shard_dict = {}
            for chunk_coords in all_chunk_coords:
                chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                if chunk_byte_slice:
                    chunk_bytes = byte_getter.get_sync(
                        prototype=chunk_spec.prototype,
                        byte_range=RangeByteRequest(chunk_byte_slice[0], chunk_byte_slice[1]),
                    )
                    if chunk_bytes:
                        shard_dict[chunk_coords] = chunk_bytes

        # Decode chunks directly via CodecChain (pure compute, no inner pipeline).
        codec_chain = self._codec_chain
        aa_chain, ab_pair, bb_chain = codec_chain.resolve_metadata_chain(chunk_spec)

        from zarr.core.codec_pipeline import fill_value_or_default

        fill_value = fill_value_or_default(shard_spec)
        for chunk_coords, chunk_selection, out_selection, _is_complete in indexed_chunks:
            chunk_bytes = shard_dict.get(chunk_coords)
            if chunk_bytes is not None:
                chunk_array = codec_chain.decode_chunk(
                    chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain
                )
                if chunk_array is not None:
                    out[out_selection] = chunk_array[chunk_selection]
                else:
                    out[out_selection] = fill_value
            else:
                out[out_selection] = fill_value

        if hasattr(indexer, "sel_shape"):
            return out.reshape(indexer.sel_shape)
        else:
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

        # Decode chunks directly via CodecChain (pure compute, no inner pipeline).
        codec_chain = self._codec_chain
        aa_chain, ab_pair, bb_chain = codec_chain.resolve_metadata_chain(chunk_spec)

        from zarr.core.codec_pipeline import fill_value_or_default

        fill_value = fill_value_or_default(shard_spec)
        for chunk_coords, chunk_selection, out_selection, _is_complete in indexed_chunks:
            chunk_bytes = shard_dict.get(chunk_coords)
            if chunk_bytes is not None:
                chunk_array = codec_chain.decode_chunk(
                    chunk_bytes, chunk_spec, aa_chain, ab_pair, bb_chain
                )
                if chunk_array is not None:
                    out[out_selection] = chunk_array[chunk_selection]
                else:
                    out[out_selection] = fill_value
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
        # _encode_single is pure compute (no IO), same as _encode_sync.
        return self._encode_sync(shard_array, shard_spec)

    def _is_total_shard(
        self, all_chunk_coords: set[tuple[int, ...]], chunks_per_shard: tuple[int, ...]
    ) -> bool:
        return len(all_chunk_coords) == product(chunks_per_shard) and all(
            chunk_coords in all_chunk_coords for chunk_coords in c_order_iter(chunks_per_shard)
        )

    def _decode_shard_index_sync(
        self, index_bytes: Buffer, chunks_per_shard: tuple[int, ...]
    ) -> _ShardIndex:
        """Decode shard index synchronously via the cached index CodecChain."""
        index_chunk_spec = self._get_index_chunk_spec(chunks_per_shard)
        index_array = self._index_codec_chain.decode_chunk(index_bytes, index_chunk_spec)
        assert index_array is not None
        return _ShardIndex(index_array.as_numpy_array())

    def _encode_shard_index_sync(self, index: _ShardIndex) -> Buffer:
        """Encode shard index synchronously via the cached index CodecChain."""
        index_chunk_spec = self._get_index_chunk_spec(index.chunks_per_shard)
        index_nd = get_ndbuffer_class().from_numpy_array(index.offsets_and_lengths)
        index_bytes = self._index_codec_chain.encode_chunk(index_nd, index_chunk_spec)
        assert index_bytes is not None
        assert isinstance(index_bytes, Buffer)
        return index_bytes

    def _shard_index_size(self, chunks_per_shard: tuple[int, ...]) -> int:
        return self._index_codec_chain.compute_encoded_size(
            16 * product(chunks_per_shard), self._get_index_chunk_spec(chunks_per_shard)
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
            return self._decode_shard_index_sync(index_bytes, chunks_per_shard)
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
            _ShardReader.from_bytes_sync(shard_bytes, self, chunks_per_shard)
            if shard_bytes
            else None
        )

    # -------------------------------------------------------------------
    # prepare_* overrides — composable building blocks for the pipeline
    # -------------------------------------------------------------------

    @property
    def inner_codec_chain(self) -> Any:
        return self._codec_chain

    def deserialize(
        self, raw: Buffer | None, shard_spec: ArraySpec
    ) -> dict[tuple[int, ...], Buffer | None]:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        if raw is None:
            return dict.fromkeys(morton_order_iter(chunks_per_shard))
        shard_reader = _ShardReader.from_bytes_sync(raw, self, chunks_per_shard)
        result: dict[tuple[int, ...], Buffer | None] = {}
        for coords in morton_order_iter(chunks_per_shard):
            chunk_byte_slice = shard_reader.index.get_chunk_slice(coords)
            if chunk_byte_slice:
                result[coords] = shard_reader.buf[chunk_byte_slice[0] : chunk_byte_slice[1]]
            else:
                result[coords] = None
        return result

    def serialize(
        self, chunk_dict: dict[tuple[int, ...], Buffer | None], shard_spec: ArraySpec
    ) -> Buffer | None:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        return self._encode_shard_dict_sync(
            chunk_dict, chunks_per_shard, default_buffer_prototype()
        )

    def prepare_read_sync(
        self,
        byte_getter: Any,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        codec_chain: Any,
        aa_chain: Any,
        ab_pair: Any,
        bb_chain: Any,
    ) -> NDBuffer | None:
        return self._decode_partial_sync(byte_getter, chunk_selection, chunk_spec)

    async def prepare_read(
        self,
        byte_getter: Any,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        codec_chain: Any,
        aa_chain: Any,
        ab_pair: Any,
        bb_chain: Any,
    ) -> NDBuffer | None:
        return await self._decode_partial_single(byte_getter, chunk_selection, chunk_spec)

    def _prepare_write_partial_fixed_sync(
        self,
        byte_setter: Any,
        chunk_spec: ArraySpec,
        chunks_per_shard: tuple[int, ...],
        chunk_spec_inner: ArraySpec,
        indexer: list[ChunkProjection],
    ) -> dict[tuple[int, ...], Buffer | None]:
        """For fixed-size partial writes: fetch only the inner chunks that need merging."""
        chunk_byte_length = self._inner_chunk_byte_length(chunk_spec_inner)
        chunk_dict: dict[tuple[int, ...], Buffer | None] = {}
        for coords, _, _, is_complete_inner in indexer:
            if is_complete_inner:
                chunk_dict[coords] = None
            else:
                offset = self._chunk_byte_offset(coords, chunks_per_shard, chunk_byte_length)
                chunk_dict[coords] = byte_setter.get_sync(
                    prototype=chunk_spec_inner.prototype,
                    byte_range=RangeByteRequest(offset, offset + chunk_byte_length),
                )
        return chunk_dict

    async def _prepare_write_partial_fixed(
        self,
        byte_setter: Any,
        chunk_spec: ArraySpec,
        chunks_per_shard: tuple[int, ...],
        chunk_spec_inner: ArraySpec,
        indexer: list[ChunkProjection],
    ) -> dict[tuple[int, ...], Buffer | None]:
        """Async version: fetch only the inner chunks that need merging."""
        chunk_byte_length = self._inner_chunk_byte_length(chunk_spec_inner)
        chunk_dict: dict[tuple[int, ...], Buffer | None] = {}
        for coords, _, _, is_complete_inner in indexer:
            if is_complete_inner:
                chunk_dict[coords] = None
            else:
                offset = self._chunk_byte_offset(coords, chunks_per_shard, chunk_byte_length)
                chunk_dict[coords] = await byte_setter.get(
                    prototype=chunk_spec_inner.prototype,
                    byte_range=RangeByteRequest(offset, offset + chunk_byte_length),
                )
        return chunk_dict

    def prepare_write_sync(
        self,
        byte_setter: Any,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        codec_chain: Any,
    ) -> PreparedWrite:
        from zarr.abc.codec import PreparedWrite, _is_complete_selection

        chunks_per_shard = self._get_chunks_per_shard(chunk_spec)
        chunk_spec_inner = self._get_chunk_spec(chunk_spec)

        # Build inner indexer first — needed for fixed-size targeted reads.
        indexer = list(
            get_indexer(
                chunk_selection,
                shape=chunk_spec.shape,
                chunk_grid=RegularChunkGrid(chunk_shape=self.chunk_shape),
            )
        )

        is_complete = _is_complete_selection(chunk_selection, chunk_spec.shape)
        if is_complete:
            # Complete selection: the pipeline will pass the shard value to
            # finalize_write which encodes the full shard in one shot.
            return PreparedWrite(
                chunk_dict={},
                inner_codec_chain=self._codec_chain,
                inner_chunk_spec=chunk_spec_inner,
                indexer=[],
                value_selection=out_selection,
                write_full_shard=True,
                is_complete_shard=True,
            )
        elif self._inner_codecs_fixed_size:
            # Fixed-size partial write: only fetch inner chunks that need merging.
            # Check if shard exists first — if not, all chunks are None.
            probe = byte_setter.get_sync(
                prototype=chunk_spec_inner.prototype,
                byte_range=RangeByteRequest(0, 1),
            )
            chunk_dict: dict[tuple[int, ...], Buffer | None]
            if probe is None:
                write_full_shard = True
                chunk_dict = {coords: None for coords, _, _, _ in indexer}
            else:
                write_full_shard = False
                chunk_dict = self._prepare_write_partial_fixed_sync(
                    byte_setter,
                    chunk_spec,
                    chunks_per_shard,
                    chunk_spec_inner,
                    indexer,
                )
        else:
            # Variable-size: must fetch entire shard.
            shard_reader = self._load_full_shard_maybe_sync(
                byte_setter, chunk_spec_inner.prototype, chunks_per_shard
            )
            write_full_shard = shard_reader is None
            shard_reader = shard_reader or _ShardReader.create_empty(chunks_per_shard)
            chunk_dict = {k: shard_reader.get(k) for k in morton_order_iter(chunks_per_shard)}

        return PreparedWrite(
            chunk_dict=chunk_dict,
            inner_codec_chain=self._codec_chain,
            inner_chunk_spec=chunk_spec_inner,
            indexer=indexer,
            value_selection=out_selection,
            write_full_shard=write_full_shard,
        )

    async def prepare_write(
        self,
        byte_setter: Any,
        chunk_spec: ArraySpec,
        chunk_selection: SelectorTuple,
        out_selection: SelectorTuple,
        codec_chain: Any,
    ) -> PreparedWrite:
        from zarr.abc.codec import PreparedWrite, _is_complete_selection

        chunks_per_shard = self._get_chunks_per_shard(chunk_spec)
        chunk_spec_inner = self._get_chunk_spec(chunk_spec)

        indexer = list(
            get_indexer(
                chunk_selection,
                shape=chunk_spec.shape,
                chunk_grid=RegularChunkGrid(chunk_shape=self.chunk_shape),
            )
        )

        is_complete = _is_complete_selection(chunk_selection, chunk_spec.shape)
        if is_complete:
            return PreparedWrite(
                chunk_dict={},
                inner_codec_chain=self._codec_chain,
                inner_chunk_spec=chunk_spec_inner,
                indexer=[],
                value_selection=out_selection,
                write_full_shard=True,
                is_complete_shard=True,
            )
        elif self._inner_codecs_fixed_size:
            probe = await byte_setter.get(
                prototype=chunk_spec_inner.prototype,
                byte_range=RangeByteRequest(0, 1),
            )
            chunk_dict: dict[tuple[int, ...], Buffer | None]
            if probe is None:
                write_full_shard = True
                chunk_dict = {coords: None for coords, _, _, _ in indexer}
            else:
                write_full_shard = False
                chunk_dict = await self._prepare_write_partial_fixed(
                    byte_setter,
                    chunk_spec,
                    chunks_per_shard,
                    chunk_spec_inner,
                    indexer,
                )
        else:
            shard_reader = await self._load_full_shard_maybe(
                byte_setter, chunk_spec_inner.prototype, chunks_per_shard
            )
            write_full_shard = shard_reader is None
            shard_reader = shard_reader or _ShardReader.create_empty(chunks_per_shard)
            chunk_dict = {k: shard_reader.get(k) for k in morton_order_iter(chunks_per_shard)}

        return PreparedWrite(
            chunk_dict=chunk_dict,
            inner_codec_chain=self._codec_chain,
            inner_chunk_spec=chunk_spec_inner,
            indexer=indexer,
            value_selection=out_selection,
            write_full_shard=write_full_shard,
        )

    @property
    def _inner_codecs_fixed_size(self) -> bool:
        return all(c.is_fixed_size for c in self._codec_chain)

    @property
    def is_fixed_size(self) -> bool:  # type: ignore[override]
        # ShardingCodec output varies when write_empty_chunks=False causes
        # fill-value sub-chunks to be omitted, so it is not fixed-size in general.
        return False

    def _inner_chunk_byte_length(self, chunk_spec: ArraySpec) -> int:
        """Encoded byte length of a single inner chunk (only valid when _inner_codecs_fixed_size)."""
        assert isinstance(chunk_spec.dtype, HasItemSize)
        raw_byte_length = product(self.chunk_shape) * chunk_spec.dtype.item_size
        return int(self._codec_chain.compute_encoded_size(raw_byte_length, chunk_spec))

    @staticmethod
    @lru_cache(maxsize=16)
    def _morton_rank_map(chunks_per_shard: tuple[int, ...]) -> dict[tuple[int, ...], int]:
        """Return a dict mapping morton-order coords → rank (0-based).

        Cached because the same chunks_per_shard is used repeatedly for reads and writes.
        """
        return {coords: rank for rank, coords in enumerate(morton_order_iter(chunks_per_shard))}

    def _chunk_byte_offset(
        self,
        chunk_coords: tuple[int, ...],
        chunks_per_shard: tuple[int, ...],
        chunk_byte_length: int,
    ) -> int:
        """Byte offset of an inner chunk within a dense shard blob.

        Assumes all chunks are present and laid out in morton order.
        """
        rank = self._morton_rank_map(chunks_per_shard)[chunk_coords]
        offset = rank * chunk_byte_length
        if self.index_location == ShardingCodecIndexLocation.start:
            offset += self._shard_index_size(chunks_per_shard)
        return offset

    def _shard_index_byte_offset(
        self, chunks_per_shard: tuple[int, ...], chunk_byte_length: int
    ) -> int:
        """Byte offset of the shard index within a dense shard blob."""
        n_chunks = product(chunks_per_shard)
        if self.index_location == ShardingCodecIndexLocation.start:
            return 0
        return n_chunks * chunk_byte_length

    def _build_dense_shard_index(
        self, chunks_per_shard: tuple[int, ...], chunk_byte_length: int
    ) -> _ShardIndex:
        """Build a shard index for a fully-dense shard with fixed-size chunks."""
        index = _ShardIndex.create_empty(chunks_per_shard)
        data_offset = (
            self._shard_index_size(chunks_per_shard)
            if self.index_location == ShardingCodecIndexLocation.start
            else 0
        )
        for rank, coords in enumerate(morton_order_iter(chunks_per_shard)):
            chunk_start = data_offset + rank * chunk_byte_length
            index.set_chunk_slice(coords, slice(chunk_start, chunk_start + chunk_byte_length))
        return index

    def _build_dense_shard_blob(
        self,
        chunk_dict: dict[tuple[int, ...], Buffer | None],
        chunks_per_shard: tuple[int, ...],
        chunk_byte_length: int,
    ) -> Buffer:
        """Build a dense shard blob with fixed-size chunks at deterministic offsets.

        Unlike ``_encode_shard_dict_sync`` (used by ``serialize``), this places each
        chunk at ``rank * chunk_byte_length`` (plus index offset for start-indexed shards),
        producing a layout compatible with ``_chunk_byte_offset`` / ``set_range``.
        """
        index = self._build_dense_shard_index(chunks_per_shard, chunk_byte_length)
        index_bytes = self._encode_shard_index_sync(index)

        # Allocate the full blob as a flat numpy array
        n_chunks = product(chunks_per_shard)
        data_size = n_chunks * chunk_byte_length
        total_size = data_size + len(index_bytes)
        blob_array = np.zeros(total_size, dtype=np.uint8)

        data_offset = (
            len(index_bytes) if self.index_location == ShardingCodecIndexLocation.start else 0
        )
        index_offset = 0 if self.index_location == ShardingCodecIndexLocation.start else data_size

        # Place each chunk at its deterministic offset
        for rank, coords in enumerate(morton_order_iter(chunks_per_shard)):
            chunk_bytes = chunk_dict.get(coords)
            if chunk_bytes is not None:
                start = data_offset + rank * chunk_byte_length
                blob_array[start : start + len(chunk_bytes)] = chunk_bytes.as_numpy_array()

        # Place the index
        blob_array[index_offset : index_offset + len(index_bytes)] = index_bytes.as_numpy_array()

        return default_buffer_prototype().buffer.from_bytes(blob_array.tobytes())

    def finalize_write_sync(self, prepared: Any, chunk_spec: ArraySpec, byte_setter: Any) -> None:
        from zarr.abc.codec import PreparedWrite

        assert isinstance(prepared, PreparedWrite)

        # Complete shard: encode the entire shard value in one shot.
        if prepared.is_complete_shard:
            assert prepared.shard_data is not None
            shard_data = prepared.shard_data
            # Expand scalar/broadcast value to shard shape.
            if shard_data.shape != chunk_spec.shape:
                expanded = chunk_spec.prototype.nd_buffer.create(
                    shape=chunk_spec.shape,
                    dtype=chunk_spec.dtype.to_native_dtype(),
                    order=chunk_spec.order,
                    fill_value=0,
                )
                expanded[()] = shard_data
                shard_data = expanded
            blob = self._encode_sync(shard_data, chunk_spec)
            if blob is None:
                byte_setter.delete_sync()
            else:
                byte_setter.set_sync(blob)
            return

        chunks_per_shard = self._get_chunks_per_shard(chunk_spec)
        chunk_spec_inner = self._get_chunk_spec(chunk_spec)

        if not self._inner_codecs_fixed_size:
            # Fall back to full serialize + set
            blob = self.serialize(prepared.chunk_dict, chunk_spec)
            if blob is None:
                byte_setter.delete_sync()
            else:
                byte_setter.set_sync(blob)
            return

        chunk_byte_length = self._inner_chunk_byte_length(chunk_spec_inner)
        # Spec with write_empty_chunks=True — needed to encode fill-value chunks
        # into actual bytes for the dense shard layout.
        dense_spec = replace(
            chunk_spec_inner,
            config=ArrayConfig(
                order=chunk_spec_inner.config.order,
                write_empty_chunks=True,
            ),
        )

        if prepared.write_full_shard:
            # Full shard write: create a fully-dense blob and write it all at once.
            # If all chunks are fill-value (None), delete the shard if it exists.
            if all(v is None for v in prepared.chunk_dict.values()):
                byte_setter.delete_sync()
                return
            # Encode fill-value chunks for any coords that are None (either
            # unmodified coords or modified coords that equal the fill value).
            for coords in morton_order_iter(chunks_per_shard):
                if prepared.chunk_dict.get(coords) is None:
                    fill_chunk = chunk_spec_inner.prototype.nd_buffer.create(
                        shape=chunk_spec_inner.shape,
                        dtype=chunk_spec_inner.dtype.to_native_dtype(),
                        order=chunk_spec_inner.order,
                        fill_value=chunk_spec_inner.fill_value,
                    )
                    prepared.chunk_dict[coords] = prepared.inner_codec_chain.encode_chunk(
                        fill_chunk, dense_spec
                    )
            blob = self._build_dense_shard_blob(
                prepared.chunk_dict, chunks_per_shard, chunk_byte_length
            )
            byte_setter.set_sync(blob)
            return

        # Existing shard with fixed-size chunks: write only modified chunks via set_range.
        # If any modified chunk became fill-value (None), fall back to full read-modify-write
        # so that shard deletion works correctly.
        has_fill_chunks = any(
            prepared.chunk_dict.get(coords) is None for coords, _, _, _ in prepared.indexer
        )
        if has_fill_chunks:
            # Need full read-modify-write for correct shard deletion behavior.
            shard_reader = self._load_full_shard_maybe_sync(
                byte_setter, chunk_spec_inner.prototype, chunks_per_shard
            )
            if shard_reader is not None:
                full_dict: dict[tuple[int, ...], Buffer | None] = {
                    k: shard_reader.get(k) for k in morton_order_iter(chunks_per_shard)
                }
                # Merge modified chunks into the full dict.
                for coords, _, _, _ in prepared.indexer:
                    full_dict[coords] = prepared.chunk_dict.get(coords)
                blob = self.serialize(full_dict, chunk_spec)
                if blob is None:
                    byte_setter.delete_sync()
                else:
                    byte_setter.set_sync(blob)
            return

        try:
            for coords, _, _, _ in prepared.indexer:
                chunk_bytes = prepared.chunk_dict.get(coords)
                if chunk_bytes is not None:
                    offset = self._chunk_byte_offset(coords, chunks_per_shard, chunk_byte_length)
                    byte_setter.set_range_sync(chunk_bytes, offset)

            # Update the shard index (unchanged for dense layout but must be rewritten
            # because the index is part of the blob).
            index = self._build_dense_shard_index(chunks_per_shard, chunk_byte_length)
            index_bytes = self._encode_shard_index_sync(index)
            index_offset = self._shard_index_byte_offset(chunks_per_shard, chunk_byte_length)
            byte_setter.set_range_sync(index_bytes, index_offset)
        except NotImplementedError:
            # Store doesn't support set_range — fall back to full serialize + set.
            blob = self.serialize(prepared.chunk_dict, chunk_spec)
            if blob is None:
                byte_setter.delete_sync()
            else:
                byte_setter.set_sync(blob)

    async def finalize_write(self, prepared: Any, chunk_spec: ArraySpec, byte_setter: Any) -> None:
        from zarr.abc.codec import PreparedWrite

        assert isinstance(prepared, PreparedWrite)

        # Complete shard: encode the entire shard value in one shot.
        if prepared.is_complete_shard:
            assert prepared.shard_data is not None
            shard_data = prepared.shard_data
            if shard_data.shape != chunk_spec.shape:
                expanded = chunk_spec.prototype.nd_buffer.create(
                    shape=chunk_spec.shape,
                    dtype=chunk_spec.dtype.to_native_dtype(),
                    order=chunk_spec.order,
                    fill_value=0,
                )
                expanded[()] = shard_data
                shard_data = expanded
            blob = self._encode_sync(shard_data, chunk_spec)
            if blob is None:
                await byte_setter.delete()
            else:
                await byte_setter.set(blob)
            return

        chunks_per_shard = self._get_chunks_per_shard(chunk_spec)
        chunk_spec_inner = self._get_chunk_spec(chunk_spec)

        if not self._inner_codecs_fixed_size:
            blob = self.serialize(prepared.chunk_dict, chunk_spec)
            if blob is None:
                await byte_setter.delete()
            else:
                await byte_setter.set(blob)
            return

        chunk_byte_length = self._inner_chunk_byte_length(chunk_spec_inner)

        if prepared.write_full_shard:
            # If all chunks are fill-value (None), delete the shard if it exists.
            if all(v is None for v in prepared.chunk_dict.values()):
                await byte_setter.delete()
                return
            dense_spec = replace(
                chunk_spec_inner,
                config=ArrayConfig(
                    order=chunk_spec_inner.config.order,
                    write_empty_chunks=True,
                ),
            )
            for coords in morton_order_iter(chunks_per_shard):
                if prepared.chunk_dict.get(coords) is None:
                    fill_chunk = chunk_spec_inner.prototype.nd_buffer.create(
                        shape=chunk_spec_inner.shape,
                        dtype=chunk_spec_inner.dtype.to_native_dtype(),
                        order=chunk_spec_inner.order,
                        fill_value=chunk_spec_inner.fill_value,
                    )
                    prepared.chunk_dict[coords] = prepared.inner_codec_chain.encode_chunk(
                        fill_chunk, dense_spec
                    )
            blob = self._build_dense_shard_blob(
                prepared.chunk_dict, chunks_per_shard, chunk_byte_length
            )
            await byte_setter.set(blob)
            return

        has_fill_chunks = any(
            prepared.chunk_dict.get(coords) is None for coords, _, _, _ in prepared.indexer
        )
        if has_fill_chunks:
            shard_reader = await self._load_full_shard_maybe(
                byte_setter, chunk_spec_inner.prototype, chunks_per_shard
            )
            if shard_reader is not None:
                full_dict: dict[tuple[int, ...], Buffer | None] = {
                    k: shard_reader.get(k) for k in morton_order_iter(chunks_per_shard)
                }
                for coords, _, _, _ in prepared.indexer:
                    full_dict[coords] = prepared.chunk_dict.get(coords)
                blob = self.serialize(full_dict, chunk_spec)
                if blob is None:
                    await byte_setter.delete()
                else:
                    await byte_setter.set(blob)
            return

        try:
            for coords, _, _, _ in prepared.indexer:
                chunk_bytes = prepared.chunk_dict.get(coords)
                if chunk_bytes is not None:
                    offset = self._chunk_byte_offset(coords, chunks_per_shard, chunk_byte_length)
                    await byte_setter.set_range(chunk_bytes, offset)

            index = self._build_dense_shard_index(chunks_per_shard, chunk_byte_length)
            index_bytes = self._encode_shard_index_sync(index)
            index_offset = self._shard_index_byte_offset(chunks_per_shard, chunk_byte_length)
            await byte_setter.set_range(index_bytes, index_offset)
        except NotImplementedError:
            blob = self.serialize(prepared.chunk_dict, chunk_spec)
            if blob is None:
                await byte_setter.delete()
            else:
                await byte_setter.set(blob)

    def compute_encoded_size(self, input_byte_length: int, shard_spec: ArraySpec) -> int:
        chunks_per_shard = self._get_chunks_per_shard(shard_spec)
        return input_byte_length + self._shard_index_size(chunks_per_shard)
