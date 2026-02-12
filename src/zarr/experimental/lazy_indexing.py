"""An experimental array that supports lazy indexing by explicitly tracking the
domain of the array.

This module implements TensorStore-inspired lazy indexing for Zarr arrays.
Key concepts:

- **IndexDomain**: Represents a rectangular region in index space with inclusive
  lower bounds and exclusive upper bounds. Unlike NumPy, domains preserve non-zero
  origins when slicing.

- **Lazy Indexing**: When you index an Array, instead of loading data, you get
  a new Array with a narrowed domain. Data is only loaded when you call `resolve()`.

- **Non-zero Origins**: Arrays can have domains that don't start at zero.
  For example, an array with domain [10, 20) has indices 10, 11, ..., 19.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product as itertools_product
from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.abc.numcodec import Numcodec
from zarr.core._info import ArrayInfo
from zarr.core.array import (
    _append,
    _get_coordinate_selection,
    _get_mask_selection,
    _get_orthogonal_selection,
    _getitem,
    _info_complete,
    _iter_chunk_coords,
    _iter_chunk_regions,
    _iter_shard_coords,
    _iter_shard_keys,
    _iter_shard_regions,
    _nbytes_stored,
    _nchunks_initialized,
    _nshards_initialized,
    _resize,
    _setitem,
    _update_attributes,
    create_codec_pipeline,
    get_array_metadata,
    parse_array_metadata,
)
from zarr.core.array_spec import ArrayConfig, ArrayConfigLike, parse_array_config
from zarr.core.buffer import (
    BufferPrototype,
    NDArrayLikeOrScalar,
    NDBuffer,
    default_buffer_prototype,
)
from zarr.core.common import (
    JSON,
    MemoryOrder,
    ShapeLike,
    ZarrFormat,
    ceildiv,
    product,
)
from zarr.core.indexing import (
    BasicSelection,
    CoordinateSelection,
    Fields,
    MaskSelection,
    OrthogonalSelection,
)
from zarr.core.metadata import (
    ArrayMetadata,
    ArrayMetadataDict,
    ArrayV2Metadata,
    ArrayV3Metadata,
)
from zarr.core.sync import sync
from zarr.storage._common import StorePath, make_store_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Self

    import numpy.typing as npt

    from zarr.abc.codec import CodecPipeline
    from zarr.abc.store import Store
    from zarr.storage import StoreLike


@dataclass(frozen=True)
class IndexDomain:
    """
    Represents a rectangular region in index space.

    An IndexDomain defines the valid indices for an array, with inclusive lower
    bounds and exclusive upper bounds for each dimension. This is inspired by
    TensorStore's IndexDomain concept.

    Unlike NumPy arrays which always have origins at zero, IndexDomain supports
    non-zero origins. For example, after slicing arr[5:10], the resulting array
    has a domain with origin 5 and shape 5, meaning valid indices are 5, 6, 7, 8, 9.

    Parameters
    ----------
    inclusive_min : tuple[int, ...]
        The inclusive lower bounds for each dimension (the first valid index).
    exclusive_max : tuple[int, ...]
        The exclusive upper bounds for each dimension (one past the last valid index).

    Examples
    --------
    >>> domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20))
    >>> domain.shape
    (10, 20)
    >>> domain.origin
    (0, 0)

    >>> # After slicing [5:8, 10:15]
    >>> sliced = IndexDomain(inclusive_min=(5, 10), exclusive_max=(8, 15))
    >>> sliced.shape
    (3, 5)
    >>> sliced.origin
    (5, 10)
    """

    inclusive_min: tuple[int, ...]
    exclusive_max: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.inclusive_min) != len(self.exclusive_max):
            raise ValueError(
                f"inclusive_min and exclusive_max must have the same length. "
                f"Got {len(self.inclusive_min)} and {len(self.exclusive_max)}."
            )
        for i, (lo, hi) in enumerate(zip(self.inclusive_min, self.exclusive_max, strict=True)):
            if lo > hi:
                raise ValueError(
                    f"inclusive_min must be <= exclusive_max for all dimensions. "
                    f"Dimension {i}: {lo} > {hi}"
                )

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> IndexDomain:
        """Create a domain with origin at zero and the given shape."""
        return cls(
            inclusive_min=(0,) * len(shape),
            exclusive_max=shape,
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.inclusive_min)

    @property
    def origin(self) -> tuple[int, ...]:
        """The origin (inclusive lower bounds) of the domain."""
        return self.inclusive_min

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the domain (exclusive_max - inclusive_min)."""
        return tuple(hi - lo for lo, hi in zip(self.inclusive_min, self.exclusive_max, strict=True))

    def contains(self, index: tuple[int, ...]) -> bool:
        """Check if an index is within this domain."""
        if len(index) != self.ndim:
            return False
        return all(
            lo <= idx < hi
            for lo, hi, idx in zip(self.inclusive_min, self.exclusive_max, index, strict=True)
        )

    def contains_domain(self, other: IndexDomain) -> bool:
        """Check if another domain is entirely contained within this domain."""
        if other.ndim != self.ndim:
            return False
        return all(
            self_lo <= other_lo and other_hi <= self_hi
            for self_lo, self_hi, other_lo, other_hi in zip(
                self.inclusive_min,
                self.exclusive_max,
                other.inclusive_min,
                other.exclusive_max,
                strict=True,
            )
        )

    def __repr__(self) -> str:
        ranges = ", ".join(
            f"[{lo}, {hi})" for lo, hi in zip(self.inclusive_min, self.exclusive_max, strict=True)
        )
        return f"IndexDomain({ranges})"

    def intersect(self, other: IndexDomain) -> IndexDomain | None:
        """
        Compute the intersection of this domain with another.

        Returns None if the domains do not overlap.

        Parameters
        ----------
        other : IndexDomain
            The other domain to intersect with.

        Returns
        -------
        IndexDomain | None
            The intersection domain, or None if they don't overlap.
        """
        if other.ndim != self.ndim:
            raise ValueError(
                f"Cannot intersect domains with different ranks: {self.ndim} vs {other.ndim}"
            )
        new_min = tuple(
            max(a, b) for a, b in zip(self.inclusive_min, other.inclusive_min, strict=True)
        )
        new_max = tuple(
            min(a, b) for a, b in zip(self.exclusive_max, other.exclusive_max, strict=True)
        )
        # Check if intersection is empty
        if any(lo >= hi for lo, hi in zip(new_min, new_max, strict=True)):
            return None
        return IndexDomain(inclusive_min=new_min, exclusive_max=new_max)

    def to_slices(self) -> tuple[slice, ...]:
        """Convert this domain to a tuple of slices.

        Returns
        -------
        tuple[slice, ...]
            One slice per dimension, from inclusive_min to exclusive_max.

        Examples
        --------
        >>> IndexDomain(inclusive_min=(5, 10), exclusive_max=(8, 15)).to_slices()
        (slice(5, 8), slice(10, 15))
        """
        return tuple(
            slice(lo, hi)
            for lo, hi in zip(self.inclusive_min, self.exclusive_max, strict=True)
        )

    def translate(self, offset: tuple[int, ...]) -> IndexDomain:
        """
        Translate (shift) this domain by an offset.

        Returns a new IndexDomain with bounds shifted by the given offset.
        This is useful for converting between coordinate systems.

        Parameters
        ----------
        offset : tuple[int, ...]
            The offset to add to each dimension's bounds. Positive values
            shift the domain in the positive direction.

        Returns
        -------
        IndexDomain
            A new domain with translated bounds.

        Examples
        --------
        >>> domain = IndexDomain(inclusive_min=(10, 20), exclusive_max=(30, 40))
        >>> domain.translate((-10, -20))
        IndexDomain([0, 10), [0, 20))

        >>> # Useful for converting domain coordinates to output coordinates
        >>> intersection = domain.intersect(other_domain)
        >>> output_domain = intersection.translate((-domain.inclusive_min[0], ...))
        """
        if len(offset) != self.ndim:
            raise ValueError(
                f"Offset must have same length as domain dimensions. "
                f"Domain has {self.ndim} dimensions, offset has {len(offset)}."
            )
        new_min = tuple(lo + off for lo, off in zip(self.inclusive_min, offset, strict=True))
        new_max = tuple(hi + off for hi, off in zip(self.exclusive_max, offset, strict=True))
        return IndexDomain(inclusive_min=new_min, exclusive_max=new_max)


@dataclass(frozen=True)
class ChunkLayout:
    """
    Describes the chunk grid for an array.

    A ChunkLayout defines how an array is partitioned into chunks. It consists of:
    - grid_origin: The coordinate where the chunk grid starts (where chunk (0,0,...) begins)
    - chunk_shape: The size of each chunk

    Key insight: Each chunk is conceptually a sub-array with its own domain. The chunk
    at coordinates (i, j, ...) has domain:
        [grid_origin[d] + i * chunk_shape[d], grid_origin[d] + (i+1) * chunk_shape[d])
    for each dimension d.

    This means a chunked array can be thought of as a concatenation of chunk sub-arrays,
    each with its own domain.

    Parameters
    ----------
    grid_origin : tuple[int, ...]
        The origin of the chunk grid (where chunk boundaries start).
    chunk_shape : tuple[int, ...]
        The shape of each chunk.

    Examples
    --------
    >>> layout = ChunkLayout(grid_origin=(0, 0), chunk_shape=(10, 10))
    >>> layout.chunk_domain((0, 0))
    IndexDomain([0, 10), [0, 10))
    >>> layout.chunk_domain((1, 2))
    IndexDomain([10, 20), [20, 30))

    >>> # With non-zero origin
    >>> layout = ChunkLayout(grid_origin=(5, 5), chunk_shape=(10, 10))
    >>> layout.chunk_domain((0, 0))
    IndexDomain([5, 15), [5, 15))
    >>> layout.is_aligned((5, 5))
    True
    >>> layout.is_aligned((7, 5))
    False
    """

    grid_origin: tuple[int, ...]
    chunk_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.grid_origin) != len(self.chunk_shape):
            raise ValueError(
                f"grid_origin and chunk_shape must have the same length. "
                f"Got {len(self.grid_origin)} and {len(self.chunk_shape)}."
            )
        if any(s <= 0 for s in self.chunk_shape):
            raise ValueError(
                f"chunk_shape must be positive in all dimensions. Got {self.chunk_shape}"
            )

    @classmethod
    def from_chunk_shape(cls, chunk_shape: tuple[int, ...]) -> ChunkLayout:
        """Create a ChunkLayout with grid origin at zero."""
        return cls(grid_origin=(0,) * len(chunk_shape), chunk_shape=chunk_shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.grid_origin)

    def is_aligned(self, coord: tuple[int, ...]) -> bool:
        """
        Check if a coordinate lies on a chunk boundary.

        A coordinate is chunk-aligned if (coord[d] - grid_origin[d]) is divisible
        by chunk_shape[d] for all dimensions d.

        Parameters
        ----------
        coord : tuple[int, ...]
            The coordinate to check.

        Returns
        -------
        bool
            True if the coordinate is on a chunk boundary in all dimensions.
        """
        if len(coord) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {len(coord)}")
        return all(
            (c - o) % s == 0
            for c, o, s in zip(coord, self.grid_origin, self.chunk_shape, strict=True)
        )

    def chunk_coords_for_point(self, point: tuple[int, ...]) -> tuple[int, ...]:
        """
        Get the chunk coordinates containing a given point.

        Parameters
        ----------
        point : tuple[int, ...]
            A point in the array's coordinate space.

        Returns
        -------
        tuple[int, ...]
            The chunk coordinates (i, j, ...) of the chunk containing this point.
        """
        if len(point) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {len(point)}")
        # Use floor division to handle negative coordinates correctly
        return tuple(
            (p - o) // s if (p - o) >= 0 else -ceildiv(o - p, s)
            for p, o, s in zip(point, self.grid_origin, self.chunk_shape, strict=True)
        )

    def chunk_domain(self, chunk_coords: tuple[int, ...]) -> IndexDomain:
        """
        Get the domain of a specific chunk.

        Each chunk is a sub-array with its own domain. This returns that domain.

        Parameters
        ----------
        chunk_coords : tuple[int, ...]
            The chunk coordinates (e.g., (0, 0) for the first chunk).

        Returns
        -------
        IndexDomain
            The domain of the chunk.
        """
        if len(chunk_coords) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {len(chunk_coords)}")
        inclusive_min = tuple(
            o + c * s
            for o, c, s in zip(self.grid_origin, chunk_coords, self.chunk_shape, strict=True)
        )
        exclusive_max = tuple(
            o + (c + 1) * s
            for o, c, s in zip(self.grid_origin, chunk_coords, self.chunk_shape, strict=True)
        )
        return IndexDomain(inclusive_min=inclusive_min, exclusive_max=exclusive_max)

    def iter_chunk_coords(self, domain: IndexDomain) -> Iterator[tuple[int, ...]]:
        """
        Iterate over all chunk coordinates that overlap with a domain.

        Parameters
        ----------
        domain : IndexDomain
            The domain to find overlapping chunks for.

        Yields
        ------
        tuple[int, ...]
            Chunk coordinates for each chunk that overlaps with the domain.
        """
        if domain.ndim != self.ndim:
            raise ValueError(f"Domain has {domain.ndim} dims, layout has {self.ndim} dims")

        # Find the range of chunk coordinates
        start_coords = self.chunk_coords_for_point(domain.inclusive_min)
        # For exclusive_max, we need the chunk containing (exclusive_max - 1)
        # But if exclusive_max is on a boundary, the last chunk is the previous one
        end_coords = tuple(
            self.chunk_coords_for_point(tuple(m - 1 for m in domain.exclusive_max))[d] + 1
            for d in range(self.ndim)
        )

        def iter_coords(
            starts: tuple[int, ...], ends: tuple[int, ...]
        ) -> Iterator[tuple[int, ...]]:
            if not starts:
                yield ()
                return
            for c in range(starts[0], ends[0]):
                for rest in iter_coords(starts[1:], ends[1:]):
                    yield (c,) + rest

        yield from iter_coords(start_coords, end_coords)

    def iter_chunk_domains(
        self, domain: IndexDomain
    ) -> Iterator[tuple[tuple[int, ...], IndexDomain]]:
        """
        Iterate over chunks that overlap with a domain, yielding their domains.

        This embodies the insight that chunks are sub-arrays with their own domains.
        For each chunk overlapping the given domain, yields the chunk coordinates
        and the intersection of the chunk's domain with the given domain.

        Parameters
        ----------
        domain : IndexDomain
            The domain to find overlapping chunks for.

        Yields
        ------
        tuple[tuple[int, ...], IndexDomain]
            Pairs of (chunk_coords, chunk_intersection_domain).
        """
        for chunk_coords in self.iter_chunk_coords(domain):
            chunk_dom = self.chunk_domain(chunk_coords)
            intersection = chunk_dom.intersect(domain)
            if intersection is not None:
                yield chunk_coords, intersection

    def aligned_domain(self, domain: IndexDomain) -> IndexDomain:
        """
        Return the largest chunk-aligned subdomain contained within the given domain.

        This rounds the lower bounds up and upper bounds down to chunk boundaries.

        Parameters
        ----------
        domain : IndexDomain
            The domain to align.

        Returns
        -------
        IndexDomain
            The largest aligned subdomain. May have zero size in some dimensions
            if the domain is smaller than a chunk.
        """
        if domain.ndim != self.ndim:
            raise ValueError(f"Domain has {domain.ndim} dims, layout has {self.ndim} dims")

        # Round lower bounds UP to next chunk boundary
        aligned_min = tuple(
            o + ceildiv(lo - o, s) * s
            for lo, o, s in zip(
                domain.inclusive_min, self.grid_origin, self.chunk_shape, strict=True
            )
        )
        # Round upper bounds DOWN to previous chunk boundary
        aligned_max = tuple(
            o + ((hi - o) // s) * s
            for hi, o, s in zip(
                domain.exclusive_max, self.grid_origin, self.chunk_shape, strict=True
            )
        )
        # Ensure we don't create an invalid domain (max < min)
        aligned_max = tuple(max(lo, hi) for lo, hi in zip(aligned_min, aligned_max, strict=True))
        return IndexDomain(inclusive_min=aligned_min, exclusive_max=aligned_max)

    def __repr__(self) -> str:
        return f"ChunkLayout(grid_origin={self.grid_origin}, chunk_shape={self.chunk_shape})"


@dataclass(frozen=True)
class StorageSource:
    """
    A source backed by Zarr storage.

    This encapsulates all the information needed to read data from a Zarr array
    stored on disk or in memory. It includes the store path, metadata, codec
    pipeline, and the index transform that maps domain coordinates to storage
    coordinates.

    Parameters
    ----------
    store_path : StorePath
        The path to the Zarr store.
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The metadata of the array.
    codec_pipeline : CodecPipeline
        The codec pipeline used for encoding and decoding chunks.
    config : ArrayConfig
        The runtime configuration of the array.
    index_transform : tuple[int, ...]
        The offset to subtract from domain coordinates to get storage coordinates.
    """

    store_path: StorePath
    metadata: ArrayV2Metadata | ArrayV3Metadata
    codec_pipeline: Any  # CodecPipeline - avoid forward reference issues
    config: ArrayConfig
    index_transform: tuple[int, ...]

    @property
    def storage_shape(self) -> tuple[int, ...]:
        """The shape of the underlying storage."""
        return self.metadata.shape

    @property
    def chunks(self) -> tuple[int, ...]:
        """The store-level chunk shape.

        For unsharded arrays, this is the chunk shape. For sharded arrays,
        this is the shard shape (chunk_grid.chunk_shape), since shards are
        the store-level unit that maps to store keys. The ShardingCodec
        handles inner chunk decoding internally.
        """
        from zarr.core.chunk_grids import RegularChunkGrid

        if isinstance(self.metadata.chunk_grid, RegularChunkGrid):
            return self.metadata.chunk_grid.chunk_shape
        return self.metadata.chunks

    @property
    def dtype(self) -> np.dtype[Any]:
        """The data type."""
        return (
            self.metadata.data_type.to_native_dtype()
            if hasattr(self.metadata, "data_type")
            else self.metadata.dtype.to_native_dtype()
        )

    @property
    def fill_value(self) -> Any:
        """The fill value."""
        return self.metadata.fill_value


def _get_storage_identity(arr: Array) -> tuple[Any, tuple[int, ...]] | None:
    """
    Get the storage identity for an Array if it's backed by a single storage source.

    Returns (store_path, index_transform) if the array has a single StorageSource,
    or None if it has multiple sources or Array sources.
    """
    if len(arr._sources) == 1 and isinstance(arr._sources[0], StorageSource):
        source = arr._sources[0]
        return (source.store_path, source.index_transform)
    return None


def _try_merge_to_single_source(
    arrays: list[Array],
    domain: IndexDomain,
) -> StorageSource | None:
    """
    Try to merge multiple Arrays into a single StorageSource.

    This succeeds when all input arrays:
    1. Are backed by the same storage (same store_path)
    2. Have the same index_transform (same coordinate mapping)
    3. Their combined domains fully cover the target domain (no gaps)

    In this case, we can represent the concatenation as a single StorageSource,
    since the storage already contains all the data we need.

    Returns the merged StorageSource, or None if merging isn't possible.
    """
    if not arrays:
        return None

    # Check if all arrays share the same storage identity
    first_identity = _get_storage_identity(arrays[0])
    if first_identity is None:
        return None

    for arr in arrays[1:]:
        identity = _get_storage_identity(arr)
        if identity != first_identity:
            return None

    # All arrays share the same storage identity.
    # Now check if the source domains fully cover the target domain.
    # We need to verify there are no gaps that would require fill_value.

    # For simplicity, check if the union of input domains equals the target domain.
    # This is a conservative check - we only merge when domains are exactly covering.

    # Compute the bounding box of all input domains
    ndim = domain.ndim
    input_min = tuple(min(arr.domain.inclusive_min[d] for arr in arrays) for d in range(ndim))
    input_max = tuple(max(arr.domain.exclusive_max[d] for arr in arrays) for d in range(ndim))
    input_bbox = IndexDomain(inclusive_min=input_min, exclusive_max=input_max)

    # For the merge to be valid without gaps, we need to check that the input arrays
    # completely cover their bounding box. This is complex to check in general.
    #
    # A simple conservative approach: only merge if there's a single input array
    # or if the input arrays' total volume equals the bounding box volume.
    # This works for non-overlapping, gap-free cases like arr[:10] and arr[10:].

    total_input_volume = sum(
        int(
            np.prod(
                [arr.domain.exclusive_max[d] - arr.domain.inclusive_min[d] for d in range(ndim)]
            )
        )
        for arr in arrays
    )
    bbox_volume = int(
        np.prod([input_bbox.exclusive_max[d] - input_bbox.inclusive_min[d] for d in range(ndim)])
    )

    # If total volume < bbox volume, there are gaps -> can't merge
    # If total volume > bbox volume, there are overlaps -> we can still merge
    #   (overlaps are fine, we just read from one source)
    # If total volume == bbox volume, perfect coverage -> can merge
    if total_input_volume < bbox_volume:
        return None

    # All arrays share the same storage and fully cover the domain
    first_source = arrays[0]._sources[0]
    assert isinstance(first_source, StorageSource)
    return first_source


@dataclass(frozen=True)
class ChunkCoordSlice:
    """
    Identifies a slice within a specific chunk.

    Parameters
    ----------
    chunk_coords : tuple[int, ...]
        The coordinates of the chunk in the chunk grid.
    selection : tuple[slice, ...]
        The slice within the chunk to read (in chunk-local coordinates, starting at 0).
    """

    chunk_coords: tuple[int, ...]
    selection: tuple[slice, ...]


def get_chunk_projections(
    storage_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    domain: IndexDomain,
    index_transform: tuple[int, ...] | None = None,
    output_origin: tuple[int, ...] | None = None,
) -> Iterator[tuple[tuple[slice, ...], ChunkCoordSlice]]:
    """
    Compute chunk projections for resolving data from a domain.

    This function maps domain coordinates to storage coordinates and determines
    which chunks need to be read and how to assemble them into an output array.

    The mapping from domain to storage coordinates is:
        storage_coord = domain_coord - index_transform

    Parameters
    ----------
    storage_shape : tuple[int, ...]
        The shape of the underlying storage array.
    chunk_shape : tuple[int, ...]
        The shape of each chunk.
    domain : IndexDomain
        The domain to resolve.
    index_transform : tuple[int, ...] | None
        The offset to subtract from domain coordinates to get storage coordinates.
        If None, defaults to (0, 0, ...), meaning domain coordinates equal storage
        coordinates.
    output_origin : tuple[int, ...] | None
        The origin of the output array's coordinate system. When provided,
        output_selection slices are relative to this origin (i.e., position 0 in
        the output array corresponds to output_origin in domain space).
        If None, defaults to the domain's origin.

    Yields
    ------
    tuple[tuple[slice, ...], ChunkCoordSlice]
        For each chunk that overlaps with the domain (after translation to storage
        coordinates), yields a tuple of:
        - output_selection: where to place the data in the output array
        - chunk_info: ChunkCoordSlice with chunk coords and slice within chunk

    Examples
    --------
    >>> # Storage is shape (100,) with chunks of size 10
    >>> # Domain is [25, 75), with default offset (0,) so storage coords are [25, 75)
    >>> storage_shape = (100,)
    >>> chunk_shape = (10,)
    >>> domain = IndexDomain(inclusive_min=(25,), exclusive_max=(75,))
    >>> projs = list(get_chunk_projections(storage_shape, chunk_shape, domain))
    >>> projs[0]  # First chunk: output_selection, chunk_info
    ((slice(0, 5),), ChunkCoordSlice(chunk_coords=(2,), selection=(slice(5, 10),)))

    >>> # With index_transform=(10,), domain 10 maps to storage 0
    >>> domain = IndexDomain(inclusive_min=(10,), exclusive_max=(20,))
    >>> list(get_chunk_projections((10,), (5,), domain, index_transform=(10,)))
    [((slice(0, 5),), ChunkCoordSlice(chunk_coords=(0,), selection=(slice(0, 5),))),
     ((slice(5, 10),), ChunkCoordSlice(chunk_coords=(1,), selection=(slice(0, 5),)))]
    """
    ndim = len(storage_shape)
    if len(chunk_shape) != ndim or domain.ndim != ndim:
        raise ValueError(
            f"Dimension mismatch: storage_shape has {ndim} dims, "
            f"chunk_shape has {len(chunk_shape)} dims, domain has {domain.ndim} dims"
        )

    if index_transform is None:
        index_transform = (0,) * ndim

    if len(index_transform) != ndim:
        raise ValueError(f"index_transform has {len(index_transform)} dims, expected {ndim}")

    if output_origin is None:
        output_origin = domain.inclusive_min

    # Translate domain to storage coordinates: storage_coord = domain_coord - index_transform
    # Then intersect with valid storage bounds [0, storage_dim)
    # All done with raw tuples to avoid IndexDomain allocation overhead in the hot loop.
    valid_lo = tuple(
        max(d_lo - it, 0)
        for d_lo, it in zip(domain.inclusive_min, index_transform, strict=True)
    )
    valid_hi = tuple(
        min(d_hi - it, s)
        for d_hi, it, s in zip(domain.exclusive_max, index_transform, storage_shape, strict=True)
    )

    # Check if there's any valid intersection
    if any(lo >= hi for lo, hi in zip(valid_lo, valid_hi, strict=True)):
        return  # No chunks to read

    # Precompute the offset from storage coordinates to output coordinates.
    # domain_coord = storage_coord + index_transform
    # output_index = domain_coord - output_origin = storage_coord + index_transform - output_origin
    storage_to_output = tuple(
        it - oo for it, oo in zip(index_transform, output_origin, strict=True)
    )

    # Compute the range of chunk coordinates that overlap with the valid storage region
    chunk_start = tuple(
        lo // c for lo, c in zip(valid_lo, chunk_shape, strict=True)
    )
    chunk_end = tuple(
        ceildiv(hi, c) for hi, c in zip(valid_hi, chunk_shape, strict=True)
    )

    # Iterate over all chunks using itertools.product (replaces recursive generator)
    for chunk_coords in itertools_product(*(range(s, e) for s, e in zip(chunk_start, chunk_end, strict=True))):
        # Compute the storage region covered by this chunk
        chunk_storage_start = tuple(c * cs for c, cs in zip(chunk_coords, chunk_shape, strict=True))
        chunk_storage_end = tuple(
            min((c + 1) * cs, dim)
            for c, cs, dim in zip(chunk_coords, chunk_shape, storage_shape, strict=True)
        )

        # Intersect with valid storage region — inline arithmetic, no IndexDomain
        inter_lo = tuple(max(a, b) for a, b in zip(chunk_storage_start, valid_lo, strict=True))
        inter_hi = tuple(min(a, b) for a, b in zip(chunk_storage_end, valid_hi, strict=True))
        if any(lo >= hi for lo, hi in zip(inter_lo, inter_hi, strict=True)):
            continue

        # chunk_selection: translate to chunk-local coords and make slices directly
        chunk_selection = tuple(
            slice(lo - cs, hi - cs)
            for lo, hi, cs in zip(inter_lo, inter_hi, chunk_storage_start, strict=True)
        )
        # output_selection: translate to output coords and make slices directly
        output_selection = tuple(
            slice(lo + off, hi + off)
            for lo, hi, off in zip(inter_lo, inter_hi, storage_to_output, strict=True)
        )

        yield (
            output_selection,
            ChunkCoordSlice(chunk_coords=chunk_coords, selection=chunk_selection),
        )


# ---------------------------------------------------------------------------
# Fast codec classes — bypass the overhead in the standard decode/encode path.
#
# The standard inner-shard path goes through:
#   codec.decode() → _batching_helper → concurrent_map → _noop_for_none
#   → BytesCodec._decode_single (with isinstance(x, NDArrayLike) Protocol check)
#
# On the write side, morton_order_iter recomputes from scratch every call
# (~250 decode_morton calls with O(n) list scans), totaling 35% of write time.
#
# These fast codecs eliminate that overhead while preserving correctness.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _cached_morton_order(chunks_per_shard: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Compute and cache the morton ordering for a given shard shape.

    The standard ``morton_order_iter`` recomputes the ordering from scratch on
    every call: ~250 ``decode_morton`` invocations with O(n) list-membership
    checks per call.  For (5,5,5) shards this adds up to 35% of write time.
    Caching the result eliminates this entirely after the first call.
    """
    from zarr.core.indexing import morton_order_iter

    return tuple(morton_order_iter(chunks_per_shard))


class _FastBytesCodec:
    """Drop-in replacement for BytesCodec._decode_single that avoids:
    1. isinstance(x, NDArrayLike) Protocol check (~16 getattr_static calls)
    2. _batching_helper/concurrent_map/Semaphore overhead
    3. _noop_for_none wrapper and its typing introspection

    This is not a full Codec subclass — it only implements the decode path
    needed by the inner shard pipeline. It stores the endian configuration
    from the original BytesCodec.
    """

    __slots__ = ("_endian_str", "_has_endianness_type")

    def __init__(self, endian_str: str | None) -> None:
        self._endian_str = endian_str
        # Cache the import so we don't re-import in the hot loop
        from zarr.core.dtype.common import HasEndianness

        self._has_endianness_type = HasEndianness

    def decode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        """Synchronous fast decode — no async overhead needed for in-memory work."""
        from dataclasses import replace as dataclass_replace

        if isinstance(chunk_spec.dtype, self._has_endianness_type):
            dtype = dataclass_replace(
                chunk_spec.dtype, endianness=self._endian_str
            ).to_native_dtype()
        else:
            dtype = chunk_spec.dtype.to_native_dtype()

        as_nd_array_like = np.asanyarray(chunk_bytes.as_array_like())
        chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
            as_nd_array_like.view(dtype=dtype)
        )
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(chunk_spec.shape)
        return chunk_array

    def encode_single(self, chunk_array: Any, chunk_spec: Any) -> Any:
        """Synchronous fast encode — mirrors BytesCodec._encode_single."""
        from zarr.codecs.bytes import Endian

        if (
            chunk_array.dtype.itemsize > 1
            and self._endian_str is not None
            and chunk_array.byteorder != Endian(self._endian_str)
        ):
            new_dtype = chunk_array.dtype.newbyteorder(Endian(self._endian_str).name)
            chunk_array = chunk_array.astype(new_dtype)

        nd_array = chunk_array.as_ndarray_like()
        nd_array = nd_array.ravel().view(dtype="B")
        return chunk_spec.prototype.buffer.from_array_like(nd_array)


def _make_fast_sharding_codec(
    original: Any,
) -> Any:
    """Create a FastShardingCodec from an existing ShardingCodec.

    Returns a new ShardingCodec subclass instance whose codec_pipeline property
    returns a pipeline that uses _FastBytesCodec for the inner decode path,
    bypassing all the standard overhead.

    Only transforms the decode path — encoding is unchanged (uses the standard
    codec pipeline).
    """
    from zarr.codecs.bytes import BytesCodec as _BytesCodec
    from zarr.codecs.sharding import ShardingCodec

    if not isinstance(original, ShardingCodec):
        return original

    # Find the BytesCodec in the inner codecs and extract its endian config
    endian_str = None
    for codec in original.codecs:
        if isinstance(codec, _BytesCodec):
            endian_str = codec.endian.value if codec.endian is not None else None
            break

    fast_bytes = _FastBytesCodec(endian_str)

    # Create a subclass that overrides _decode_partial_single with a fast path
    class _FastShardingCodec(ShardingCodec):
        """ShardingCodec with a fast inner decode path.

        Overrides _decode_partial_single to decode inner chunks directly using
        _FastBytesCodec instead of going through codec_pipeline.read() which
        adds per-chunk overhead from:
        - BasicIndexer iteration + morton ordering
        - concurrent_map + Semaphore per batch
        - _batching_helper + _noop_for_none per codec
        - isinstance(x, NDArrayLike) Protocol check per chunk
        """

        _fast_bytes: _FastBytesCodec

        async def _decode_partial_single(
            self,
            byte_getter: Any,
            selection: Any,
            shard_spec: Any,
        ) -> NDBuffer | None:
            from zarr.abc.store import RangeByteRequest
            from zarr.core.chunk_grids import RegularChunkGrid
            from zarr.core.indexing import get_indexer

            shard_shape = shard_spec.shape
            chunk_shape = self.chunk_shape
            chunks_per_shard = self._get_chunks_per_shard(shard_spec)
            chunk_spec = self._get_chunk_spec(shard_spec)

            indexer = get_indexer(
                selection,
                shape=shard_shape,
                chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            )

            # Setup output array
            out = shard_spec.prototype.nd_buffer.empty(
                shape=indexer.shape,
                dtype=shard_spec.dtype.to_native_dtype(),
                order=shard_spec.order,
            )

            indexed_chunks = list(indexer)
            all_chunk_coords = {chunk_coords for chunk_coords, *_ in indexed_chunks}

            # Read bytes of all requested chunks (same logic as standard ShardingCodec)
            shard_dict: dict[tuple[int, ...], Any] = {}
            if self._is_total_shard(all_chunk_coords, chunks_per_shard):
                shard_dict_maybe = await self._load_full_shard_maybe(
                    byte_getter=byte_getter,
                    prototype=chunk_spec.prototype,
                    chunks_per_shard=chunks_per_shard,
                )
                if shard_dict_maybe is None:
                    return None
                shard_dict = shard_dict_maybe  # type: ignore[assignment]
            else:
                shard_index = await self._load_shard_index_maybe(
                    byte_getter, chunks_per_shard
                )
                if shard_index is None:
                    return None
                for chunk_coords in all_chunk_coords:
                    chunk_byte_slice = shard_index.get_chunk_slice(chunk_coords)
                    if chunk_byte_slice:
                        chunk_bytes = await byte_getter.get(
                            prototype=chunk_spec.prototype,
                            byte_range=RangeByteRequest(
                                chunk_byte_slice[0], chunk_byte_slice[1]
                            ),
                        )
                        if chunk_bytes:
                            shard_dict[chunk_coords] = chunk_bytes

            # Fast inner decode: directly decode + slice each chunk without
            # going through codec_pipeline.read() and all its overhead.
            fill_value = shard_spec.fill_value
            fb = self._fast_bytes
            for chunk_coords, chunk_selection, out_selection, _is_complete in indexed_chunks:
                chunk_bytes = shard_dict.get(chunk_coords)
                if chunk_bytes is not None:
                    chunk_array = fb.decode_single(chunk_bytes, chunk_spec)
                    out[out_selection] = chunk_array[chunk_selection]
                else:
                    out[out_selection] = fill_value

            if hasattr(indexer, "sel_shape"):
                return out.reshape(indexer.sel_shape)
            return out

        async def _decode_single(
            self,
            shard_bytes: Any,
            shard_spec: Any,
        ) -> NDBuffer:
            """Fast full-shard decode that bypasses codec_pipeline.read()."""
            from zarr.codecs.sharding import _ShardReader, _ShardingByteGetter
            from zarr.core.chunk_grids import RegularChunkGrid
            from zarr.core.indexing import BasicIndexer

            shard_shape = shard_spec.shape
            chunk_shape = self.chunk_shape
            chunks_per_shard = self._get_chunks_per_shard(shard_spec)
            chunk_spec = self._get_chunk_spec(shard_spec)

            indexer = BasicIndexer(
                tuple(slice(0, s) for s in shard_shape),
                shape=shard_shape,
                chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            )

            out = chunk_spec.prototype.nd_buffer.empty(
                shape=shard_shape,
                dtype=shard_spec.dtype.to_native_dtype(),
                order=shard_spec.order,
            )
            shard_dict = await _ShardReader.from_bytes(shard_bytes, self, chunks_per_shard)

            if shard_dict.index.is_all_empty():
                out.fill(shard_spec.fill_value)
                return out

            fill_value = shard_spec.fill_value
            fb = self._fast_bytes
            for chunk_coords, chunk_selection, out_selection, _is_complete in indexer:
                try:
                    chunk_bytes = shard_dict[chunk_coords]
                except KeyError:
                    out[out_selection] = fill_value
                    continue
                chunk_array = fb.decode_single(chunk_bytes, chunk_spec)
                out[out_selection] = chunk_array[chunk_selection]

            return out

        async def _encode_partial_single(
            self,
            byte_setter: Any,
            shard_array: Any,
            selection: Any,
            shard_spec: Any,
        ) -> None:
            """Fast partial encode that bypasses the inner codec_pipeline.write().

            The standard path:
              codec_pipeline.write() → write_batch (not partial) →
              decode_batch → _merge_chunk_array → all_equal → encode_batch
            Each step goes through _batching_helper → concurrent_map overhead.

            This override:
            - Uses cached morton ordering (eliminates 35% of write time)
            - Decodes/encodes inline with _FastBytesCodec (no async overhead)
            - Merges data directly (no _merge_chunk_array/all_equal overhead)
            """
            from zarr.codecs.sharding import _ShardReader
            from zarr.core.buffer import default_buffer_prototype
            from zarr.core.chunk_grids import RegularChunkGrid
            from zarr.core.indexing import get_indexer

            shard_shape = shard_spec.shape
            chunk_shape = self.chunk_shape
            chunks_per_shard = self._get_chunks_per_shard(shard_spec)
            chunk_spec = self._get_chunk_spec(shard_spec)

            # Load existing shard data (same as parent)
            shard_reader = await self._load_full_shard_maybe(
                byte_getter=byte_setter,
                prototype=chunk_spec.prototype,
                chunks_per_shard=chunks_per_shard,
            )
            shard_reader = shard_reader or _ShardReader.create_empty(chunks_per_shard)

            # Build shard_dict from existing data using CACHED morton order
            morton = _cached_morton_order(chunks_per_shard)
            shard_dict: dict[tuple[int, ...], Any] = {
                k: shard_reader.get(k) for k in morton
            }

            # Get the indexer for the selection being written
            indexer = get_indexer(
                selection,
                shape=shard_shape,
                chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            )

            # Fast inner write: decode existing → merge → encode, all inline.
            fb = self._fast_bytes
            fill_value = shard_spec.fill_value
            write_empty = chunk_spec.config.write_empty_chunks

            for chunk_coords, chunk_selection, out_selection, is_complete in indexer:
                existing_bytes = shard_dict.get(chunk_coords)

                if (
                    is_complete
                    and shard_array.shape == chunk_spec.shape
                    and shard_array[out_selection].shape == chunk_spec.shape
                ):
                    # Complete overwrite — encode the new data directly
                    chunk_array = shard_array
                else:
                    # Partial write — merge with existing data
                    if existing_bytes is not None:
                        chunk_array = fb.decode_single(existing_bytes, chunk_spec)
                        chunk_array = chunk_array.copy()
                    else:
                        chunk_array = chunk_spec.prototype.nd_buffer.create(
                            shape=chunk_spec.shape,
                            dtype=chunk_spec.dtype.to_native_dtype(),
                            order=chunk_spec.order,
                            fill_value=fill_value,
                        )
                    chunk_array[chunk_selection] = shard_array[out_selection]

                if not write_empty and chunk_array.all_equal(fill_value):
                    shard_dict[chunk_coords] = None
                else:
                    shard_dict[chunk_coords] = fb.encode_single(chunk_array, chunk_spec)

            # Encode the shard dict using CACHED morton order
            buf = await self._fast_encode_shard_dict(
                shard_dict, chunks_per_shard, default_buffer_prototype()
            )

            if buf is None:
                await byte_setter.delete()
            else:
                await byte_setter.set(buf)

        async def _encode_single(
            self,
            shard_array: Any,
            shard_spec: Any,
        ) -> Any:
            """Fast full-shard encode that bypasses codec_pipeline.write()."""
            from zarr.core.buffer import default_buffer_prototype
            from zarr.core.chunk_grids import RegularChunkGrid
            from zarr.core.indexing import BasicIndexer

            shard_shape = shard_spec.shape
            chunk_shape = self.chunk_shape
            chunks_per_shard = self._get_chunks_per_shard(shard_spec)
            chunk_spec = self._get_chunk_spec(shard_spec)

            indexer = BasicIndexer(
                tuple(slice(0, s) for s in shard_shape),
                shape=shard_shape,
                chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            )

            morton = _cached_morton_order(chunks_per_shard)
            shard_builder: dict[tuple[int, ...], Any] = dict.fromkeys(morton)

            fb = self._fast_bytes
            fill_value = shard_spec.fill_value
            write_empty = chunk_spec.config.write_empty_chunks

            for chunk_coords, chunk_selection, out_selection, is_complete in indexer:
                if (
                    is_complete
                    and shard_array.shape == chunk_spec.shape
                    and shard_array[out_selection].shape == chunk_spec.shape
                ):
                    chunk_array = shard_array
                else:
                    chunk_array = shard_array[out_selection]
                    if chunk_array.shape != chunk_spec.shape:
                        full = chunk_spec.prototype.nd_buffer.create(
                            shape=chunk_spec.shape,
                            dtype=chunk_spec.dtype.to_native_dtype(),
                            order=chunk_spec.order,
                            fill_value=fill_value,
                        )
                        full[chunk_selection] = chunk_array
                        chunk_array = full

                if not write_empty and chunk_array.all_equal(fill_value):
                    shard_builder[chunk_coords] = None
                else:
                    shard_builder[chunk_coords] = fb.encode_single(chunk_array, chunk_spec)

            return await self._fast_encode_shard_dict(
                shard_builder, chunks_per_shard, default_buffer_prototype()
            )

        async def _fast_encode_shard_dict(
            self,
            map: Any,
            chunks_per_shard: tuple[int, ...],
            buffer_prototype: Any,
        ) -> Any:
            """Encode a shard dict using cached morton order."""
            from zarr.codecs.sharding import (
                MAX_UINT_64,
                ShardingCodecIndexLocation,
                _ShardIndex,
            )

            index = _ShardIndex.create_empty(chunks_per_shard)
            buffers = []
            template = buffer_prototype.buffer.create_zero_length()
            chunk_start = 0

            for chunk_coords in _cached_morton_order(chunks_per_shard):
                value = map.get(chunk_coords)
                if value is None:
                    continue
                if len(value) == 0:
                    continue
                chunk_length = len(value)
                buffers.append(value)
                index.set_chunk_slice(
                    chunk_coords, slice(chunk_start, chunk_start + chunk_length)
                )
                chunk_start += chunk_length

            if len(buffers) == 0:
                return None

            index_bytes = await self._encode_shard_index(index)
            if self.index_location == ShardingCodecIndexLocation.start:
                empty_chunks_mask = index.offsets_and_lengths[..., 0] == MAX_UINT_64
                index.offsets_and_lengths[~empty_chunks_mask, 0] += len(index_bytes)
                index_bytes = await self._encode_shard_index(index)
                buffers.insert(0, index_bytes)
            else:
                buffers.append(index_bytes)

            return template.combine(buffers)

    # Construct the fast sharding codec instance by bypassing __init__
    # (which would re-parse all the codecs through parse_codecs).
    fast_codec = object.__new__(_FastShardingCodec)
    object.__setattr__(fast_codec, "chunk_shape", original.chunk_shape)
    object.__setattr__(fast_codec, "codecs", original.codecs)
    object.__setattr__(fast_codec, "index_codecs", original.index_codecs)
    object.__setattr__(fast_codec, "index_location", original.index_location)
    object.__setattr__(fast_codec, "_fast_bytes", fast_bytes)

    # Copy the lru_cached methods from the original
    from functools import lru_cache

    object.__setattr__(fast_codec, "_get_index_chunk_spec", lru_cache()(fast_codec._get_index_chunk_spec))
    object.__setattr__(fast_codec, "_get_chunks_per_shard", lru_cache()(fast_codec._get_chunks_per_shard))

    return fast_codec


def _make_fast_codec_pipeline(pipeline: Any) -> Any:
    """Replace standard codecs in a pipeline with fast versions.

    This function takes a BatchedCodecPipeline and returns a new one where:
    - ShardingCodec is replaced with _FastShardingCodec (fast inner decode)
    - BytesCodec as the outer array-bytes codec is left alone (we already
      use _decode_bytes_fast in _decode_chunks_with_selection)

    Only modifies the codec if it's a ShardingCodec — other codecs pass through.
    """
    from zarr.codecs.sharding import ShardingCodec
    from zarr.core.codec_pipeline import BatchedCodecPipeline

    if not isinstance(pipeline, BatchedCodecPipeline):
        return pipeline

    ab_codec = pipeline.array_bytes_codec
    if isinstance(ab_codec, ShardingCodec):
        fast_ab = _make_fast_sharding_codec(ab_codec)
        return BatchedCodecPipeline(
            array_array_codecs=pipeline.array_array_codecs,
            array_bytes_codec=fast_ab,
            bytes_bytes_codecs=pipeline.bytes_bytes_codecs,
            batch_size=pipeline.batch_size,
        )

    return pipeline


class _StoreByteGetter:
    """Lightweight ByteGetter that wraps a store + path string.

    Avoids StorePath construction overhead (normalize_path with regex,
    string splitting, and validation) by storing the path directly.
    Satisfies the ByteGetter protocol: async def get(prototype, byte_range=None).
    """

    __slots__ = ("_store", "_path")

    def __init__(self, store: Any, path: str) -> None:
        self._store = store
        self._path = path

    async def get(
        self, prototype: BufferPrototype, byte_range: Any = None
    ) -> Buffer | None:
        return await self._store.get(self._path, prototype=prototype, byte_range=byte_range)


def _decode_bytes_fast(
    endian_str: str | None,
    chunk_bytes: Buffer,
    chunk_spec: Any,
) -> NDBuffer:
    """Fast-path for BytesCodec._decode_single that avoids the expensive
    isinstance(x, NDArrayLike) Protocol check.

    The standard BytesCodec._decode_single checks isinstance(x, NDArrayLike)
    where NDArrayLike is a @runtime_checkable Protocol with 16 members. Each
    check triggers ~16 inspect.getattr_static() calls via typing.__instancecheck__.
    This adds up to ~213k getattr_static calls for a typical workload.

    Instead, we call np.asanyarray() unconditionally, which is a no-op for numpy
    arrays (returns the same object) and correctly handles other array types.
    """
    from dataclasses import replace as dataclass_replace

    from zarr.core.dtype.common import HasEndianness

    if isinstance(chunk_spec.dtype, HasEndianness):
        dtype = dataclass_replace(chunk_spec.dtype, endianness=endian_str).to_native_dtype()
    else:
        dtype = chunk_spec.dtype.to_native_dtype()

    # Skip the expensive isinstance(x, NDArrayLike) Protocol check.
    # np.asanyarray is a no-op on numpy arrays, so this is always safe.
    as_nd_array_like = np.asanyarray(chunk_bytes.as_array_like())

    chunk_array = chunk_spec.prototype.nd_buffer.from_ndarray_like(
        as_nd_array_like.view(dtype=dtype)
    )

    if chunk_array.shape != chunk_spec.shape:
        chunk_array = chunk_array.reshape(chunk_spec.shape)
    return chunk_array


async def _decode_chunks_with_selection(
    codec_pipeline: Any,
    store: Any,
    chunk_paths: list[str],
    chunk_spec: Any,
    chunk_selections: list[tuple[slice, ...]],
    prototype: BufferPrototype,
) -> list[NDBuffer | None]:
    """Decode chunks and apply selections in storage-coordinate space.

    Unlike the standard codec pipeline read path which applies selections
    after the full decode pipeline (in decoded/user space), this function
    applies chunk_selections directly to the output of the array-bytes codec
    (in storage/encoded space), before array-array codecs run. This is
    correct because chunk_selections from get_chunk_projections are already
    expressed in storage coordinates, which is the same coordinate space as
    the array-bytes codec output.

    This avoids the need to reverse-transform selections through array-array
    codecs (e.g. transpose), which the standard path would require.

    Key optimizations vs standard path:
    - Takes store + path strings directly, avoiding StorePath creation and
      normalize_path (regex, string splitting) per chunk.
    - Takes a single chunk_spec instead of per-chunk specs, since all chunks
      in a regular chunk grid have the same spec. Metadata is resolved once
      through the codec chain, not per-chunk.
    - For codecs that support partial decoding (e.g. ShardingCodec), uses
      _decode_partial_single to read only the needed inner chunks.
    - For BytesCodec, uses _decode_bytes_fast to avoid the expensive
      isinstance(x, NDArrayLike) Protocol check.
    - Other codec decode calls bypass the batching machinery and call
      _decode_single directly.
    """
    from zarr.abc.codec import ArrayBytesCodecPartialDecodeMixin
    from zarr.codecs.bytes import BytesCodec as _BytesCodec
    from zarr.core.codec_pipeline import BatchedCodecPipeline
    from zarr.core.common import concurrent_map
    from zarr.core.config import config

    assert isinstance(codec_pipeline, BatchedCodecPipeline)

    ab_codec = codec_pipeline.array_bytes_codec

    # Check if we can use partial decoding. This requires:
    # 1. No bytes-bytes codecs (they scramble byte ranges)
    # 2. No array-array codecs (they transform array structure)
    # 3. Array-bytes codec supports partial decode (e.g. ShardingCodec)
    use_partial_decode = (
        len(codec_pipeline.bytes_bytes_codecs) == 0
        and len(codec_pipeline.array_array_codecs) == 0
        and isinstance(ab_codec, ArrayBytesCodecPartialDecodeMixin)
    )

    if use_partial_decode:
        # Partial decode path: pass ByteGetter + selection to the codec.
        # The codec reads only the needed inner chunks (for ShardingCodec)
        # and returns an NDBuffer already shaped to the selection.
        # No post-decode slicing needed.

        # Resolve spec through the codec chain (just the ab codec, no aa/bb)
        ab_spec = ab_codec.resolve_metadata(chunk_spec)

        # Use concurrent_map so shard decodes can overlap (important for
        # remote stores where each shard decode does I/O).
        chunk_array_batch: list[NDBuffer | None] = await concurrent_map(
            [
                (_StoreByteGetter(store, path), sel, ab_spec)
                for path, sel in zip(chunk_paths, chunk_selections, strict=False)
            ],
            ab_codec._decode_partial_single,
            config.get("async.concurrency"),
        )

        return chunk_array_batch

    # Full decode path: fetch all bytes, decode, then apply selection.

    # Fetch chunk bytes from store directly (bypass StorePath construction).
    chunk_bytes_batch = await concurrent_map(
        [(path,) for path in chunk_paths],
        lambda path: store.get(path, prototype=prototype),
        config.get("async.concurrency"),
    )

    # Resolve metadata through codec chain once (all chunks share the same spec).
    spec = chunk_spec
    aa_codecs_with_spec = []
    for aa_codec in codec_pipeline.array_array_codecs:
        aa_codecs_with_spec.append((aa_codec, spec))
        spec = aa_codec.resolve_metadata(spec)

    ab_codec_spec = (ab_codec, spec)
    spec = ab_codec.resolve_metadata(spec)

    bb_codecs_with_spec = []
    for bb_codec in codec_pipeline.bytes_bytes_codecs:
        bb_codecs_with_spec.append((bb_codec, spec))
        spec = bb_codec.resolve_metadata(spec)

    # Decode: bytes-bytes codecs (reverse order) — direct _decode_single calls
    for bb_codec, bb_spec in bb_codecs_with_spec[::-1]:
        chunk_bytes_batch = [
            (await bb_codec._decode_single(chunk, bb_spec)) if chunk is not None else None
            for chunk in chunk_bytes_batch
        ]

    # Decode: array-bytes codec — use fast path for BytesCodec to avoid
    # the expensive isinstance(x, NDArrayLike) Protocol check.
    ab_codec_resolved, ab_spec = ab_codec_spec
    if isinstance(ab_codec_resolved, _BytesCodec):
        endian_str = ab_codec_resolved.endian.value if ab_codec_resolved.endian is not None else None
        chunk_array_batch = [
            _decode_bytes_fast(endian_str, chunk, ab_spec) if chunk is not None else None
            for chunk in chunk_bytes_batch
        ]
    else:
        chunk_array_batch = [
            (await ab_codec_resolved._decode_single(chunk, ab_spec)) if chunk is not None else None
            for chunk in chunk_bytes_batch
        ]

    # Apply chunk_selections in storage-coordinate space (array-bytes output space).
    chunk_array_batch = [
        chunk_array[sel] if chunk_array is not None else None
        for chunk_array, sel in zip(chunk_array_batch, chunk_selections, strict=False)
    ]

    # Decode: array-array codecs (reverse order) — direct _decode_single calls
    for aa_codec, aa_spec in aa_codecs_with_spec[::-1]:
        chunk_array_batch = [
            (await aa_codec._decode_single(chunk, aa_spec)) if chunk is not None else None
            for chunk in chunk_array_batch
        ]

    return chunk_array_batch


class Array:
    """
    A Zarr array class that supports lazy indexing with explicit domain tracking.

    This class extends standard Zarr array functionality with TensorStore-inspired
    lazy indexing. When you index an array using `__getitem__`, instead of loading
    data immediately, you get a new Array with a narrowed domain. Data is only
    loaded when you explicitly call `resolve()`.

    An Array can be backed by:
    - A single storage source (when opened from a store)
    - Multiple sources (when created via merge)
    - Other Arrays as sources (enabling nested composition)

    Key concepts:
    - **Domain**: Each array has an IndexDomain that defines its valid index range.
      The domain has an origin (inclusive lower bounds) and a shape.
    - **Lazy Indexing**: `arr[5:10]` returns a new Array with domain [5, 10), not data.
    - **Data Resolution**: Call `resolve()` to actually load the data as a numpy array.
    - **Non-zero Origins**: Arrays can have domains that don't start at zero.
    - **Merging**: `merge([a, b])` returns an Array combining multiple sources.

    Examples
    --------
    >>> arr = Array.open("path/to/array")
    >>> arr.domain
    IndexDomain([0, 100))

    >>> # Lazy indexing - returns a new Array, not data
    >>> sliced = arr[20:30]
    >>> sliced.domain
    IndexDomain([20, 30))

    >>> # Actually load the data
    >>> data = sliced.resolve()
    >>> data.shape
    (10,)

    >>> # Merging returns the same type
    >>> combined = merge([arr[0:30], arr[70:100]])
    >>> isinstance(combined, Array)
    True
    """

    _domain: IndexDomain
    _sources: tuple[StorageSource | Array, ...]
    _dtype: np.dtype[Any]
    _fill_value: Any

    # For storage-backed arrays, keep references to these for compatibility
    # These are None for multi-source arrays
    _metadata: ArrayV2Metadata | ArrayV3Metadata | None
    _store_path: StorePath | None
    _codec_pipeline: Any | None  # CodecPipeline
    _config: ArrayConfig | None

    def __init__(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata | ArrayMetadataDict,
        *,
        domain: IndexDomain | None = None,
        index_transform: tuple[int, ...] | None = None,
        codec_pipeline: CodecPipeline | None = None,
        config: ArrayConfigLike | None = None,
    ) -> None:
        """Create an Array from storage (single source)."""
        metadata_parsed = parse_array_metadata(metadata)
        config_parsed = parse_array_config(config)

        if codec_pipeline is None:
            codec_pipeline = create_codec_pipeline(metadata=metadata_parsed, store=store_path.store)

        # Replace standard codecs with fast versions for the decode path.
        # This replaces ShardingCodec with _FastShardingCodec which bypasses
        # the inner codec_pipeline.read() overhead (concurrent_map, Semaphore,
        # _batching_helper, _noop_for_none, isinstance Protocol checks).
        codec_pipeline = _make_fast_codec_pipeline(codec_pipeline)

        # Default domain is origin at zero with shape from metadata
        if domain is None:
            domain = IndexDomain.from_shape(metadata_parsed.shape)

        # Default storage transform offset is zero (domain coords = storage coords)
        if index_transform is None:
            index_transform = (0,) * domain.ndim

        # Create a single storage source
        source = StorageSource(
            store_path=store_path,
            metadata=metadata_parsed,
            codec_pipeline=codec_pipeline,
            config=config_parsed,
            index_transform=index_transform,
        )

        self._domain = domain
        self._sources = (source,)
        self._dtype = (
            metadata_parsed.data_type.to_native_dtype()
            if hasattr(metadata_parsed, "data_type")
            else metadata_parsed.dtype.to_native_dtype()
        )
        self._fill_value = metadata_parsed.fill_value

        # Keep references for backward compatibility
        self._metadata = metadata_parsed
        self._store_path = store_path
        self._codec_pipeline = codec_pipeline
        self._config = config_parsed

    @classmethod
    def _from_sources(
        cls,
        sources: Sequence[StorageSource | Array],
        *,
        domain: IndexDomain,
        dtype: np.dtype[Any],
        fill_value: Any,
    ) -> Array:
        """Create an Array from multiple sources (internal constructor)."""
        arr = object.__new__(cls)
        arr._domain = domain
        arr._sources = tuple(sources)
        arr._dtype = dtype
        arr._fill_value = fill_value

        # For single StorageSource, preserve the storage references for compatibility
        if len(sources) == 1 and isinstance(sources[0], StorageSource):
            source = sources[0]
            arr._metadata = source.metadata
            arr._store_path = source.store_path
            arr._codec_pipeline = source.codec_pipeline
            arr._config = source.config
        else:
            # Multi-source arrays don't have single storage references
            arr._metadata = None
            arr._store_path = None
            arr._codec_pipeline = None
            arr._config = None

        return arr

    # -------------------------------------------------------------------------
    # Class methods: open
    # -------------------------------------------------------------------------

    @classmethod
    async def open_async(
        cls,
        store: StoreLike,
        *,
        domain: IndexDomain | None = None,
        config: ArrayConfigLike | None = None,
        codec_pipeline: CodecPipeline | None = None,
        zarr_format: ZarrFormat | None = 3,
    ) -> Array:
        """
        Async method to open an existing Zarr array from a given store.

        Parameters
        ----------
        store : StoreLike
            The store containing the Zarr array.
        domain : IndexDomain | None, optional
            The domain for this array view. If None, defaults to a domain with
            origin at zero and shape from the array metadata.
        zarr_format : ZarrFormat | None, optional
            The Zarr format version (default is 3).

        Returns
        -------
        Array
            The opened Zarr array.
        """
        store_path = await make_store_path(store)
        metadata_dict = await get_array_metadata(store_path, zarr_format=zarr_format)
        return cls(
            store_path=store_path,
            metadata=metadata_dict,
            domain=domain,
            codec_pipeline=codec_pipeline,
            config=config,
        )

    @classmethod
    def open(
        cls,
        store: StoreLike,
        *,
        domain: IndexDomain | None = None,
        config: ArrayConfigLike | None = None,
        codec_pipeline: CodecPipeline | None = None,
        zarr_format: ZarrFormat | None = 3,
    ) -> Array:
        """
        Open an existing Zarr array from a given store.

        Parameters
        ----------
        store : StoreLike
            The store containing the Zarr array.
        domain : IndexDomain | None, optional
            The domain for this array view. If None, defaults to a domain with
            origin at zero and shape from the array metadata.
        zarr_format : ZarrFormat | None, optional
            The Zarr format version (default is 3).

        Returns
        -------
        Array
            The opened Zarr array.
        """
        return sync(
            cls.open_async(
                store,
                config=config,
                codec_pipeline=codec_pipeline,
                domain=domain,
                zarr_format=zarr_format,
            )
        )

    # -------------------------------------------------------------------------
    # Properties (all synchronous, derived from internal state)
    # -------------------------------------------------------------------------

    @property
    def domain(self) -> IndexDomain:
        """The domain defining valid indices for this array view."""
        return self._domain

    @property
    def sources(self) -> tuple[StorageSource | Array, ...]:
        """The sources backing this array."""
        return self._sources

    @property
    def store(self) -> Store | None:
        """The store containing the array data, or None for multi-source arrays."""
        if self._store_path is not None:
            return self._store_path.store
        return None

    @property
    def store_path(self) -> StorePath | None:
        """The store path, or None for multi-source arrays."""
        return self._store_path

    @property
    def metadata(self) -> ArrayV2Metadata | ArrayV3Metadata | None:
        """The metadata, or None for multi-source arrays."""
        return self._metadata

    @property
    def codec_pipeline(self) -> Any | None:
        """The codec pipeline, or None for multi-source arrays."""
        return self._codec_pipeline

    @property
    def config(self) -> ArrayConfig | None:
        """The config, or None for multi-source arrays."""
        return self._config

    @property
    def origin(self) -> tuple[int, ...]:
        """The origin (inclusive lower bounds) of this array's domain."""
        return self._domain.origin

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions in the Array."""
        return self._domain.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the Array (from the domain, not metadata)."""
        return self._domain.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        """Returns the data type of the array."""
        return self._dtype

    @property
    def fill_value(self) -> Any:
        """Returns the fill value of the array."""
        return self._fill_value

    @property
    def index_transform(self) -> tuple[int, ...] | None:
        """The index transform for single-source storage arrays, or None."""
        if len(self._sources) == 1 and isinstance(self._sources[0], StorageSource):
            return self._sources[0].index_transform
        return None

    @property
    def chunks(self) -> tuple[int, ...] | None:
        """Returns the chunk shape of the Array, or None for multi-source arrays."""
        if self._metadata is not None:
            return self._metadata.chunks
        return None

    @property
    def chunk_layout(self) -> ChunkLayout | None:
        """
        Returns the chunk layout describing how this array is partitioned.

        For multi-source arrays, returns None as there is no single chunk layout.

        The chunk layout defines the chunk grid in domain coordinates. It accounts
        for the index_transform, so the grid_origin is expressed in the array's
        current coordinate system.

        Each chunk can be thought of as a sub-array with its own domain. Use
        `chunk_layout.chunk_domain(coords)` to get the domain of a specific chunk,
        or `chunk_layout.iter_chunk_domains(domain)` to iterate over chunks
        overlapping a region.

        Returns
        -------
        ChunkLayout | None
            The chunk layout with grid_origin in domain coordinates, or None.

        Examples
        --------
        >>> arr = Array.open("path/to/array")  # shape (100,), chunks (10,)
        >>> arr.chunk_layout
        ChunkLayout(grid_origin=(0,), chunk_shape=(10,))
        >>> arr.chunk_layout.is_aligned((0,))
        True
        >>> arr.chunk_layout.is_aligned((5,))
        False

        >>> # After slicing, layout is in the sliced domain's coordinates
        >>> sliced = arr[25:75]
        >>> sliced.chunk_layout.is_aligned((30,))  # 30 is a chunk boundary
        True
        """
        if self.index_transform is not None and self.chunks is not None:
            return ChunkLayout(grid_origin=self.index_transform, chunk_shape=self.chunks)
        return None

    @property
    def shards(self) -> tuple[int, ...] | None:
        """Returns the shard shape of the Array, or None if sharding is not used."""
        if self._metadata is not None:
            return self._metadata.shards
        return None

    @property
    def size(self) -> int:
        """Returns the total number of elements in the array."""
        return product(self.shape)

    @property
    def filters(self) -> tuple[Numcodec, ...] | tuple[ArrayArrayCodec, ...] | None:
        """Filters applied to each chunk before serialization."""
        if self._metadata is None:
            return None
        if self._metadata.zarr_format == 2:
            filters = self._metadata.filters
            if filters is None:
                return ()
            return filters
        return tuple(
            codec for codec in self._metadata.inner_codecs if isinstance(codec, ArrayArrayCodec)
        )

    @property
    def serializer(self) -> ArrayBytesCodec | None:
        """Array-to-bytes codec for serializing chunks."""
        if self._metadata is None:
            return None
        if self._metadata.zarr_format == 2:
            return None
        return next(
            codec for codec in self._metadata.inner_codecs if isinstance(codec, ArrayBytesCodec)
        )

    @property
    def compressors(self) -> tuple[Numcodec, ...] | tuple[BytesBytesCodec, ...] | None:
        """Compressors applied to each chunk after serialization."""
        if self._metadata is None:
            return None
        if self._metadata.zarr_format == 2:
            if self._metadata.compressor is not None:
                return (self._metadata.compressor,)
            return ()
        return tuple(
            codec for codec in self._metadata.inner_codecs if isinstance(codec, BytesBytesCodec)
        )

    @property
    def _zdtype(self) -> Any:
        """The zarr-specific representation of the array data type."""
        if self._metadata is None:
            return None
        if self._metadata.zarr_format == 2:
            return self._metadata.dtype
        else:
            return self._metadata.data_type

    @property
    def order(self) -> MemoryOrder | None:
        """Returns the memory order of the array."""
        if self._metadata is None or self._config is None:
            return None
        if self._metadata.zarr_format == 2:
            return self._metadata.order
        else:
            return self._config.order

    @property
    def attrs(self) -> dict[str, JSON] | None:
        """Returns the attributes of the array."""
        if self._metadata is None:
            return None
        return self._metadata.attributes

    @property
    def read_only(self) -> bool | None:
        """Returns True if the array is read-only, or None for multi-source arrays."""
        if self._store_path is not None:
            return self._store_path.read_only
        return None

    @property
    def path(self) -> str | None:
        """Storage path, or None for multi-source arrays."""
        if self._store_path is not None:
            return self._store_path.path
        return None

    @property
    def name(self) -> str | None:
        """Array name following h5py convention, or None for multi-source arrays."""
        if self.path is None:
            return None
        name = self.path
        if not name.startswith("/"):
            name = "/" + name
        return name

    @property
    def basename(self) -> str | None:
        """Final component of name, or None for multi-source arrays."""
        if self.name is None:
            return None
        return self.name.split("/")[-1]

    @property
    def cdata_shape(self) -> tuple[int, ...] | None:
        """The shape of the chunk grid for this array."""
        return self._chunk_grid_shape

    @property
    def _chunk_grid_shape(self) -> tuple[int, ...] | None:
        """The shape of the chunk grid for this array."""
        if self.chunks is None:
            return None
        return tuple(starmap(ceildiv, zip(self.shape, self.chunks, strict=True)))

    @property
    def _shard_grid_shape(self) -> tuple[int, ...] | None:
        """The shape of the shard grid for this array."""
        if self.chunks is None:
            return None
        if self.shards is None:
            shard_shape = self.chunks
        else:
            shard_shape = self.shards
        return tuple(starmap(ceildiv, zip(self.shape, shard_shape, strict=True)))

    @property
    def nchunks(self) -> int | None:
        """The number of chunks in this array."""
        if self._chunk_grid_shape is None:
            return None
        return product(self._chunk_grid_shape)

    @property
    def _nshards(self) -> int | None:
        """The number of shards in this array."""
        if self._shard_grid_shape is None:
            return None
        return product(self._shard_grid_shape)

    @property
    def nbytes(self) -> int:
        """The total number of bytes that would be stored if all chunks were initialized."""
        return self.size * self._dtype.itemsize

    @property
    def info(self) -> ArrayInfo | None:
        """Return the statically known information for an array, or None for multi-source."""
        if self._metadata is None:
            return None
        return self._info()

    def _info(
        self, count_chunks_initialized: int | None = None, count_bytes_stored: int | None = None
    ) -> ArrayInfo | None:
        if self._metadata is None or self._store_path is None:
            return None
        return ArrayInfo(
            _zarr_format=self._metadata.zarr_format,
            _data_type=self._zdtype,
            _fill_value=self._fill_value,
            _shape=self.shape,
            _order=self.order,
            _shard_shape=self.shards,
            _chunk_shape=self.chunks,
            _read_only=self.read_only,
            _compressors=self.compressors,
            _filters=self.filters,
            _serializer=self.serializer,
            _store_type=type(self._store_path.store).__name__,
            _count_bytes=self.nbytes,
            _count_bytes_stored=count_bytes_stored,
            _count_chunks_initialized=count_chunks_initialized,
        )

    # -------------------------------------------------------------------------
    # Iteration methods (synchronous)
    # -------------------------------------------------------------------------

    def _iter_chunk_coords(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[int, ...]]:
        """Iterate over chunk coordinates in chunk grid space."""
        return _iter_chunk_coords(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_shard_coords(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[int, ...]]:
        """Iterate over shard coordinates in shard grid space."""
        return _iter_shard_coords(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_shard_keys(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[str]:
        """Iterate over the keys of stored objects supporting this array."""
        return _iter_shard_keys(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_chunk_regions(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[slice, ...]]:
        """Iterate over chunk regions in array index space."""
        return _iter_chunk_regions(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_shard_regions(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[slice, ...]]:
        """Iterate over shard regions in array index space."""
        return _iter_shard_regions(array=self, origin=origin, selection_shape=selection_shape)

    # -------------------------------------------------------------------------
    # nchunks_initialized: async and sync
    # -------------------------------------------------------------------------

    async def nchunks_initialized_async(self) -> int:
        """
        Asynchronously calculate the number of chunks that have been initialized.

        Returns
        -------
        int
            The number of chunks that have been initialized.
        """
        return await _nchunks_initialized(self)

    def nchunks_initialized(self) -> int:
        """
        Calculate the number of chunks that have been initialized.

        Returns
        -------
        int
            The number of chunks that have been initialized.
        """
        return sync(self.nchunks_initialized_async())

    # -------------------------------------------------------------------------
    # _nshards_initialized: async and sync
    # -------------------------------------------------------------------------

    async def _nshards_initialized_async(self) -> int:
        """
        Asynchronously calculate the number of shards that have been initialized.

        Returns
        -------
        int
            The number of shards that have been initialized.
        """
        return await _nshards_initialized(self)

    def _nshards_initialized(self) -> int:
        """
        Calculate the number of shards that have been initialized.

        Returns
        -------
        int
            The number of shards that have been initialized.
        """
        return sync(self._nshards_initialized_async())

    # -------------------------------------------------------------------------
    # nbytes_stored: async and sync
    # -------------------------------------------------------------------------

    async def nbytes_stored_async(self) -> int:
        """
        Asynchronously calculate the number of bytes stored for this array.

        Returns
        -------
        int
            The number of bytes stored.
        """
        return await _nbytes_stored(self.store_path)

    def nbytes_stored(self) -> int:
        """
        Calculate the number of bytes stored for this array.

        Returns
        -------
        int
            The number of bytes stored.
        """
        return sync(self.nbytes_stored_async())

    # -------------------------------------------------------------------------
    # getitem: async and sync
    # -------------------------------------------------------------------------

    async def getitem_async(
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously retrieve a subset of the array's data based on the provided selection.

        Parameters
        ----------
        selection : BasicSelection
            A selection object specifying the subset of data to retrieve.
        prototype : BufferPrototype, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The retrieved subset of the array's data.
        """
        return await _getitem(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            prototype=prototype,
        )

    def getitem(
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Retrieve a subset of the array's data based on the provided selection.

        Parameters
        ----------
        selection : BasicSelection
            A selection object specifying the subset of data to retrieve.
        prototype : BufferPrototype, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The retrieved subset of the array's data.
        """
        return sync(self.getitem_async(selection, prototype=prototype))

    def __getitem__(self, selection: BasicSelection) -> Self:
        """
        Lazy indexing: returns a new Array with a narrowed domain.

        Unlike standard Zarr arrays which load data immediately, this method
        returns a new Array view with an updated domain. No I/O is performed.
        To actually load data, call `resolve()` on the result.

        This follows TensorStore's design where:
        - Indexing operations create virtual views without loading data
        - Indices are ABSOLUTE coordinates in the domain's index space
        - Negative indices refer to actual negative coordinates, NOT "from the end"

        This is different from NumPy where arr[-1] means "last element". Here,
        arr[-1] means "coordinate -1" which is only valid if -1 is within the
        array's domain.

        Parameters
        ----------
        selection : BasicSelection
            The selection (int, slice, or tuple of ints/slices). These are
            absolute coordinates in the domain's index space.

        Returns
        -------
        Array
            A new Array with a narrowed domain.

        Examples
        --------
        >>> arr = Array.open("path/to/array")  # shape (100,), domain [0, 100)
        >>> arr.domain
        IndexDomain([0, 100))

        >>> sliced = arr[20:30]  # No data loaded!
        >>> sliced.domain
        IndexDomain([20, 30))

        >>> # To get element at coordinate 25:
        >>> arr[25].domain
        IndexDomain([25, 26))

        >>> # With a shifted domain:
        >>> shifted = arr.with_domain(IndexDomain((-50,), (50,)))
        >>> shifted[-10:10].domain  # Coordinates -10 to 10
        IndexDomain([-10, 10))

        >>> data = sliced.resolve()  # Now data is loaded
        """
        new_domain = self._apply_selection_to_domain(selection)
        return self._with_domain(new_domain)

    def _normalize_selection(self, selection: BasicSelection) -> tuple[int | slice, ...]:
        """Normalize a selection to a tuple of ints/slices."""
        if not isinstance(selection, tuple):
            selection = (selection,)

        # Handle ellipsis
        result: list[int | slice] = []
        ellipsis_seen = False
        for sel in selection:
            if sel is Ellipsis:
                if ellipsis_seen:
                    raise IndexError("an index can only have a single ellipsis ('...')")
                ellipsis_seen = True
                # Insert enough slices to fill remaining dimensions
                num_missing = self.ndim - (len(selection) - 1)
                result.extend([slice(None)] * num_missing)
            else:
                result.append(sel)  # type: ignore[arg-type]

        # Pad with full slices if needed
        while len(result) < self.ndim:
            result.append(slice(None))

        if len(result) > self.ndim:
            raise IndexError(
                f"too many indices for array: array has {self.ndim} dimensions, "
                f"but {len(result)} were indexed"
            )

        return tuple(result)

    def _apply_selection_to_domain(self, selection: BasicSelection) -> IndexDomain:
        """
        Apply a selection to the current domain and return a new domain.

        Following TensorStore's design:
        - Indices are ABSOLUTE coordinates in the domain's index space
        - Negative indices refer to actual negative coordinates, NOT "from the end"
        - This is different from NumPy where -1 means "last element"

        For example, with domain [10, 20):
        - arr[15] selects coordinate 15 (valid)
        - arr[5] selects coordinate 5 (out of bounds - before domain start)
        - arr[-1] selects coordinate -1 (out of bounds - before domain start)

        This matches TensorStore's behavior where the index space has meaning
        independent of array bounds.
        """
        normalized = self._normalize_selection(selection)

        new_inclusive_min: list[int] = []
        new_exclusive_max: list[int] = []

        for dim_idx, (sel, dim_lo, dim_hi) in enumerate(
            zip(normalized, self.domain.inclusive_min, self.domain.exclusive_max, strict=True)
        ):
            if isinstance(sel, int):
                # In TensorStore style, the index IS the coordinate - no translation
                # Negative indices mean negative coordinates, not "from the end"
                abs_idx = sel

                # Bounds check against domain
                if abs_idx < dim_lo or abs_idx >= dim_hi:
                    raise IndexError(
                        f"index {sel} is out of bounds for dimension {dim_idx} "
                        f"with domain [{dim_lo}, {dim_hi})"
                    )

                # Integer indexing gives a length-1 slice in lazy indexing
                # (dimension is NOT dropped, unlike NumPy)
                new_inclusive_min.append(abs_idx)
                new_exclusive_max.append(abs_idx + 1)

            else:
                # sel is a slice
                # Slice bounds are also absolute coordinates
                start, stop, step = sel.start, sel.stop, sel.step

                if step is not None and step != 1:
                    raise IndexError(
                        "lazy indexing only supports step=1 slices. "
                        f"Got step={step}. Use resolve() first for strided access."
                    )

                # Handle None/default values - None means "to the edge of domain"
                if start is None:
                    abs_start = dim_lo
                else:
                    abs_start = start  # Absolute coordinate

                if stop is None:
                    abs_stop = dim_hi
                else:
                    abs_stop = stop  # Absolute coordinate

                # Clamp to domain bounds (like NumPy slice behavior - no error for OOB)
                abs_start = max(abs_start, dim_lo)
                abs_stop = min(abs_stop, dim_hi)
                abs_stop = max(abs_stop, abs_start)  # Ensure stop >= start

                new_inclusive_min.append(abs_start)
                new_exclusive_max.append(abs_stop)

        return IndexDomain(
            inclusive_min=tuple(new_inclusive_min),
            exclusive_max=tuple(new_exclusive_max),
        )

    def _with_domain(
        self,
        new_domain: IndexDomain,
        index_transform: tuple[int, ...] | None = None,
    ) -> Self:
        """Create a new Array with a different domain (internal helper).

        Parameters
        ----------
        new_domain : IndexDomain
            The new domain.
        index_transform : tuple[int, ...] | None
            The new storage transform offset. Only used for single-source storage arrays.
            If None, preserves the current offset.
        """
        # For single storage source, we can update the index_transform
        if len(self._sources) == 1 and isinstance(self._sources[0], StorageSource):
            source = self._sources[0]
            if index_transform is None:
                index_transform = source.index_transform
            new_source = StorageSource(
                store_path=source.store_path,
                metadata=source.metadata,
                codec_pipeline=source.codec_pipeline,
                config=source.config,
                index_transform=index_transform,
            )
            return self.__class__._from_sources(
                sources=[new_source],
                domain=new_domain,
                dtype=self._dtype,
                fill_value=self._fill_value,
            )
        else:
            # For multi-source arrays, just narrow the domain
            # Filter sources to only include those that overlap with new domain
            new_sources: list[StorageSource | Array] = []
            for source in self._sources:
                if isinstance(source, StorageSource):
                    # Keep the source as-is, resolve will handle the intersection
                    new_sources.append(source)
                else:
                    # It's an Array - slice it to the new domain
                    intersection = source.domain.intersect(new_domain)
                    if intersection is not None:
                        slices = tuple(
                            slice(
                                max(new_domain.inclusive_min[d], source.domain.inclusive_min[d]),
                                min(new_domain.exclusive_max[d], source.domain.exclusive_max[d]),
                            )
                            for d in range(self.ndim)
                        )
                        new_sources.append(source[slices])

            return self.__class__._from_sources(
                sources=new_sources if new_sources else list(self._sources),
                domain=new_domain,
                dtype=self._dtype,
                fill_value=self._fill_value,
            )

    def with_domain(self, new_domain: IndexDomain) -> Self:
        """
        Create a new Array view with a different domain.

        This allows creating views with arbitrary domains, including non-zero
        origins or even domains that extend beyond the underlying storage bounds.
        When resolving data from regions outside storage bounds, the array's
        fill_value is used (this is standard Zarr behavior for uninitialized chunks).

        The new domain's origin will map to storage coordinate 0. This means
        domain coordinate `new_domain.origin[i]` will read from storage coordinate 0
        in dimension i.

        Parameters
        ----------
        new_domain : IndexDomain
            The new domain for the array view.

        Returns
        -------
        Array
            A new Array with the specified domain.

        Examples
        --------
        >>> arr = Array.open("path/to/array")  # shape (10,), fill_value=0
        >>> arr.domain
        IndexDomain([0, 10))

        >>> # Create a view with a shifted origin - domain 10 maps to storage 0
        >>> shifted = arr.with_domain(IndexDomain(inclusive_min=(10,), exclusive_max=(20,)))
        >>> shifted.origin
        (10,)
        >>> shifted.shape
        (10,)
        >>> shifted[15].resolve()  # domain 15 -> storage 5, returns data[5]

        >>> # Create a view with negative origin - domain -5 maps to storage 0
        >>> neg = arr.with_domain(IndexDomain(inclusive_min=(-5,), exclusive_max=(5,)))
        >>> neg.origin
        (-5,)
        >>> neg.shape
        (10,)
        >>> neg[-3].resolve()  # domain -3 -> storage 2, returns data[2]
        """
        if new_domain.ndim != self.ndim:
            raise ValueError(
                f"New domain must have same number of dimensions as array. "
                f"Array has {self.ndim} dimensions, new domain has {new_domain.ndim}."
            )
        # Set storage transform offset to the new domain's origin
        # so that domain.origin maps to storage coordinate 0
        return self._with_domain(new_domain, index_transform=new_domain.origin)

    async def resolve_async(
        self,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously resolve (materialize) this array view by loading the data.

        This is the method that actually performs I/O and loads the data from
        storage. The returned numpy array will have shape equal to this array's
        domain shape.

        For single-source arrays, domain coordinates are translated to storage
        coordinates using the index_transform. For multi-source arrays, data is
        assembled from all sources that overlap with the domain.

        If the domain extends beyond storage bounds or has gaps between sources,
        those regions are filled with the array's fill_value.

        Parameters
        ----------
        prototype : BufferPrototype, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The data as a numpy array (or scalar if the domain has size 1 in all dims).
        """
        # Create output array filled with fill_value
        output = np.full(self.shape, self._fill_value, dtype=self._dtype)

        for source in self._sources:
            if isinstance(source, StorageSource):
                await self._resolve_storage_source(source, output, prototype)
            else:
                await self._resolve_array_source(source, output)

        return output

    async def _resolve_storage_source(
        self,
        source: StorageSource,
        output: np.ndarray[Any, Any],
        prototype: BufferPrototype | None,
    ) -> None:
        """Resolve data from a storage source into the output array."""
        # Compute the domain of this source in our coordinate system
        source_domain = IndexDomain(
            inclusive_min=source.index_transform,
            exclusive_max=tuple(
                o + s
                for o, s in zip(source.index_transform, source.storage_shape, strict=True)
            ),
        )

        # Find intersection with our domain
        intersection = source_domain.intersect(self._domain)
        if intersection is None:
            return

        if prototype is None:
            prototype = default_buffer_prototype()

        # Compute chunk_spec once — all chunks in a regular grid share the same spec.
        # This avoids creating ~N ArraySpec objects (each calling parse_shapelike).
        chunk_spec = source.metadata.get_chunk_spec((0,) * len(source.chunks), source.config, prototype=prototype)

        # Build chunk paths as plain strings, bypassing StorePath construction
        # and its normalize_path call (regex + string splitting per chunk).
        store = source.store_path.store
        base_path = source.store_path.path

        chunk_paths = []
        chunk_selections = []
        out_selections = []
        for out_selection, chunk_info in get_chunk_projections(
            storage_shape=source.storage_shape,
            chunk_shape=source.chunks,
            domain=intersection,
            index_transform=source.index_transform,
            output_origin=self._domain.inclusive_min,
        ):
            chunk_key = source.metadata.encode_chunk_key(chunk_info.chunk_coords)
            path = f"{base_path}/{chunk_key}" if base_path else chunk_key
            chunk_paths.append(path)
            chunk_selections.append(chunk_info.selection)
            out_selections.append(out_selection)

        if not chunk_paths:
            return

        # Decode chunks with selections applied in storage-coordinate space
        chunk_arrays = await _decode_chunks_with_selection(
            source.codec_pipeline,
            store,
            chunk_paths,
            chunk_spec,
            chunk_selections,
            prototype,
        )

        # Write decoded chunks into the output array
        fill_value = self._fill_value
        for chunk_array, out_sel in zip(chunk_arrays, out_selections, strict=True):
            if chunk_array is not None:
                output[out_sel] = chunk_array.as_ndarray_like()
            else:
                output[out_sel] = fill_value

    async def _resolve_array_source(
        self,
        source: Array,
        output: np.ndarray[Any, Any],
    ) -> None:
        """Resolve data from an Array source into the output array."""
        # Find intersection of source's domain with our domain
        intersection = source.domain.intersect(self._domain)
        if intersection is None:
            return

        # Resolve the source array's data
        data = await source.resolve_async()

        # Map intersection to output array coordinates (relative to our origin)
        neg_origin = tuple(-x for x in self._domain.inclusive_min)
        output_slices = intersection.translate(neg_origin).to_slices()

        # Map intersection to source data coordinates (relative to source's origin)
        neg_source_origin = tuple(-x for x in source.domain.inclusive_min)
        data_slices = intersection.translate(neg_source_origin).to_slices()

        output[output_slices] = data[data_slices]

    def resolve(
        self,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Resolve (materialize) this array view by loading the data.

        This is the method that actually performs I/O and loads the data from
        storage. The returned numpy array will have shape equal to this array's
        domain shape.

        Parameters
        ----------
        prototype : BufferPrototype, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The data as a numpy array (or scalar if the domain has size 1 in all dims).

        Examples
        --------
        >>> arr = Array.open("path/to/array")
        >>> sliced = arr[10:20]  # Lazy - no I/O yet
        >>> data = sliced.resolve()  # Now I/O happens
        >>> data.shape
        (10,)
        """
        return sync(self.resolve_async(prototype=prototype))

    def _domain_to_selection(self) -> tuple[slice, ...]:
        """Convert the current domain to a selection tuple for the underlying storage."""
        return tuple(
            slice(lo, hi)
            for lo, hi in zip(self.domain.inclusive_min, self.domain.exclusive_max, strict=True)
        )

    # -------------------------------------------------------------------------
    # setitem: async and sync
    # -------------------------------------------------------------------------

    async def setitem_async(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        prototype: BufferPrototype | None = None,
    ) -> None:
        """
        Asynchronously set values in the array using basic indexing.

        Parameters
        ----------
        selection : BasicSelection
            The selection defining the region of the array to set.
        value : npt.ArrayLike
            The values to be written into the selected region.
        prototype : BufferPrototype, optional
            A buffer prototype to use.
        """
        return await _setitem(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            value,
            prototype=prototype,
        )

    def setitem(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        prototype: BufferPrototype | None = None,
    ) -> None:
        """
        Set values in the array using basic indexing.

        Parameters
        ----------
        selection : BasicSelection
            The selection defining the region of the array to set.
        value : npt.ArrayLike
            The values to be written into the selected region.
        prototype : BufferPrototype, optional
            A buffer prototype to use.
        """
        sync(self.setitem_async(selection, value, prototype=prototype))

    def __setitem__(self, selection: BasicSelection, value: npt.ArrayLike) -> None:
        """Set data using indexing syntax."""
        self.setitem(selection, value)

    # -------------------------------------------------------------------------
    # get_orthogonal_selection: async and sync
    # -------------------------------------------------------------------------

    async def get_orthogonal_selection_async(
        self,
        selection: OrthogonalSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously get an orthogonal selection from the array.

        Parameters
        ----------
        selection : OrthogonalSelection
            The orthogonal selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return await _get_orthogonal_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            out=out,
            fields=fields,
            prototype=prototype,
        )

    def get_orthogonal_selection(
        self,
        selection: OrthogonalSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Get an orthogonal selection from the array.

        Parameters
        ----------
        selection : OrthogonalSelection
            The orthogonal selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return sync(
            self.get_orthogonal_selection_async(
                selection, out=out, fields=fields, prototype=prototype
            )
        )

    # -------------------------------------------------------------------------
    # get_mask_selection: async and sync
    # -------------------------------------------------------------------------

    async def get_mask_selection_async(
        self,
        mask: MaskSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously get a mask selection from the array.

        Parameters
        ----------
        mask : MaskSelection
            The boolean mask specifying the selection.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return await _get_mask_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            mask,
            out=out,
            fields=fields,
            prototype=prototype,
        )

    def get_mask_selection(
        self,
        mask: MaskSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Get a mask selection from the array.

        Parameters
        ----------
        mask : MaskSelection
            The boolean mask specifying the selection.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return sync(
            self.get_mask_selection_async(mask, out=out, fields=fields, prototype=prototype)
        )

    # -------------------------------------------------------------------------
    # get_coordinate_selection: async and sync
    # -------------------------------------------------------------------------

    async def get_coordinate_selection_async(
        self,
        selection: CoordinateSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously get a coordinate selection from the array.

        Parameters
        ----------
        selection : CoordinateSelection
            The coordinate selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return await _get_coordinate_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            out=out,
            fields=fields,
            prototype=prototype,
        )

    def get_coordinate_selection(
        self,
        selection: CoordinateSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Get a coordinate selection from the array.

        Parameters
        ----------
        selection : CoordinateSelection
            The coordinate selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return sync(
            self.get_coordinate_selection_async(
                selection, out=out, fields=fields, prototype=prototype
            )
        )

    # -------------------------------------------------------------------------
    # resize: async and sync
    # -------------------------------------------------------------------------

    async def resize_async(self, new_shape: ShapeLike, delete_outside_chunks: bool = True) -> None:
        """
        Asynchronously resize the array to a new shape.

        Parameters
        ----------
        new_shape : ShapeLike
            The desired new shape of the array.
        delete_outside_chunks : bool, optional
            If True (default), chunks that fall outside the new shape will be deleted.
        """
        return await _resize(self, new_shape, delete_outside_chunks)

    def resize(self, new_shape: ShapeLike, delete_outside_chunks: bool = True) -> None:
        """
        Resize the array to a new shape.

        Parameters
        ----------
        new_shape : ShapeLike
            The desired new shape of the array.
        delete_outside_chunks : bool, optional
            If True (default), chunks that fall outside the new shape will be deleted.
        """
        sync(self.resize_async(new_shape, delete_outside_chunks))

    # -------------------------------------------------------------------------
    # append: async and sync
    # -------------------------------------------------------------------------

    async def append_async(self, data: npt.ArrayLike, axis: int = 0) -> tuple[int, ...]:
        """
        Asynchronously append data to the array along the specified axis.

        Parameters
        ----------
        data : npt.ArrayLike
            Data to be appended.
        axis : int
            Axis along which to append.

        Returns
        -------
        tuple[int, ...]
            The new shape of the array after appending.
        """
        return await _append(self, data, axis)

    def append(self, data: npt.ArrayLike, axis: int = 0) -> tuple[int, ...]:
        """
        Append data to the array along the specified axis.

        Parameters
        ----------
        data : npt.ArrayLike
            Data to be appended.
        axis : int
            Axis along which to append.

        Returns
        -------
        tuple[int, ...]
            The new shape of the array after appending.
        """
        return sync(self.append_async(data, axis))

    # -------------------------------------------------------------------------
    # update_attributes: async and sync
    # -------------------------------------------------------------------------

    async def update_attributes_async(self, new_attributes: dict[str, JSON]) -> Self:
        """
        Asynchronously update the array's attributes.

        Parameters
        ----------
        new_attributes : dict[str, JSON]
            A dictionary of new attributes to update or add.

        Returns
        -------
        Array
            The array with the updated attributes.
        """
        await _update_attributes(self, new_attributes)
        return self

    def update_attributes(self, new_attributes: dict[str, JSON]) -> Self:
        """
        Update the array's attributes.

        Parameters
        ----------
        new_attributes : dict[str, JSON]
            A dictionary of new attributes to update or add.

        Returns
        -------
        Array
            The array with the updated attributes.
        """
        return sync(self.update_attributes_async(new_attributes))

    # -------------------------------------------------------------------------
    # info_complete: async and sync
    # -------------------------------------------------------------------------

    async def info_complete_async(self) -> ArrayInfo:
        """
        Asynchronously return all the information for an array, including dynamic information.

        Returns
        -------
        ArrayInfo
            Complete information about the array including chunks initialized and bytes stored.
        """
        return await _info_complete(self)

    def info_complete(self) -> ArrayInfo:
        """
        Return all the information for an array, including dynamic information.

        Returns
        -------
        ArrayInfo
            Complete information about the array including chunks initialized and bytes stored.
        """
        return sync(self.info_complete_async())

    # -------------------------------------------------------------------------
    # __repr__
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._store_path is not None:
            return f"<Array {self._store_path} domain={self._domain} dtype={self._dtype}>"
        else:
            return f"<Array domain={self._domain} dtype={self._dtype} sources={len(self._sources)}>"

    def __eq__(self, other: object) -> bool:
        """
        Check equality between two Arrays.

        Two Arrays are equal if they have the same domain, dtype, fill_value,
        and equivalent sources. For single-source arrays backed by the same
        storage with the same index_transform, they are considered equal.
        """
        if not isinstance(other, Array):
            return NotImplemented

        # Basic properties must match
        if self._domain != other._domain:
            return False
        if self._dtype != other._dtype:
            return False
        if self._fill_value != other._fill_value:
            return False

        # Compare sources
        if len(self._sources) != len(other._sources):
            return False

        for s1, s2 in zip(self._sources, other._sources, strict=True):
            if type(s1) is not type(s2):
                return False
            # StorageSource is a frozen dataclass, so equality works
            # Array uses recursive equality check
            if s1 != s2:
                return False

        return True

    def __array__(
        self, dtype: np.dtype[Any] | None = None, copy: bool | None = None
    ) -> np.ndarray[Any, Any]:
        """
        Convert the array to a numpy array by resolving it.

        This allows using `np.array(arr)` or `np.asarray(arr)` to get the data.

        Parameters
        ----------
        dtype : np.dtype, optional
            The desired dtype for the output array.
        copy : bool, optional
            Whether to copy the data.

        Returns
        -------
        np.ndarray
            The resolved data as a numpy array.
        """
        result = self.resolve()
        if isinstance(result, np.ndarray):
            if dtype is not None and result.dtype != dtype:
                result = result.astype(dtype, copy=copy if copy is not None else True)
            elif copy:
                result = result.copy()
            return result
        # Scalar case
        arr = np.asarray(result, dtype=dtype)
        if copy:
            arr = arr.copy()
        return arr


# -----------------------------------------------------------------------------
# merge: Combine multiple arrays by domain
# -----------------------------------------------------------------------------


def merge(
    arrays: Sequence[Array],
    *,
    fill_value: Any = None,
    domain: IndexDomain | None = None,
) -> Array:
    """
    Merge arrays by their domains.

    This is the inverse of slicing. While slicing takes an array and returns
    a view with a smaller domain, merge takes multiple arrays and returns
    a view with a larger domain (the union/bounding box of all input domains).

    Unlike numpy.concatenate which requires arrays to be adjacent along one axis,
    this function allows arrays to have arbitrary non-overlapping (or overlapping)
    domains. Gaps between arrays are filled with fill_value.

    Parameters
    ----------
    arrays : Sequence[Array]
        The arrays to merge. All must have the same dtype and number
        of dimensions. Must all be Array instances.
    fill_value : Any, optional
        The fill value for regions not covered by any input array.
        Defaults to the first array's fill_value.
    domain : IndexDomain, optional
        The domain of the result. If not provided, uses the bounding box
        of all input arrays' domains.

    Returns
    -------
    Array
        A lazy merge that can be resolved or further sliced.

    Examples
    --------
    >>> # Slice and reassemble
    >>> arr = Array.open("path/to/array")  # domain [0, 100)
    >>> left = arr[0:40]
    >>> right = arr[60:100]
    >>> combined = merge([left, right])
    >>> combined.domain
    IndexDomain([0, 100))
    >>> data = combined.resolve()
    >>> data[40:60]  # Filled with fill_value

    >>> # Verify merge inverts slicing
    >>> arr = Array.open("path/to/array")
    >>> chunks = [arr[i:i+10] for i in range(0, 100, 10)]
    >>> reassembled = merge(chunks)
    >>> np.array_equal(arr.resolve(), reassembled.resolve())
    True

    >>> # Works with overlapping domains (last wins)
    >>> a = arr[0:60]
    >>> b = arr[40:100]  # Overlaps with a in [40, 60)
    >>> combined = merge([a, b])  # b's data used in overlap region
    """
    if not arrays:
        raise ValueError("merge requires at least one array")

    arrays_list = list(arrays)
    first = arrays_list[0]
    ndim = first.domain.ndim

    # Validate all arrays have same ndim and dtype
    for i, arr in enumerate(arrays_list):
        if arr.domain.ndim != ndim:
            raise ValueError(
                f"All arrays must have the same number of dimensions. "
                f"Array 0 has {ndim} dims, array {i} has {arr.domain.ndim} dims."
            )
        if arr.dtype != first.dtype:
            raise ValueError(
                f"All arrays must have the same dtype. "
                f"Array 0 has dtype {first.dtype}, array {i} has dtype {arr.dtype}."
            )

    # Determine fill_value
    if fill_value is None:
        fill_value = first.fill_value

    # Compute domain as bounding box if not provided
    if domain is None:
        inclusive_min = tuple(
            min(arr.domain.inclusive_min[d] for arr in arrays_list) for d in range(ndim)
        )
        exclusive_max = tuple(
            max(arr.domain.exclusive_max[d] for arr in arrays_list) for d in range(ndim)
        )
        domain = IndexDomain(inclusive_min=inclusive_min, exclusive_max=exclusive_max)

    # Create an Array with the input arrays as sources
    # We need to convert ArrayLike to Array - for now we only support Array inputs
    sources: list[StorageSource | Array] = []
    for arr in arrays_list:
        if isinstance(arr, Array):
            sources.append(arr)
        else:
            raise TypeError(f"merge currently only supports Array inputs, got {type(arr).__name__}")

    # Try to merge sources if they all come from the same storage
    merged_source = _try_merge_to_single_source(arrays_list, domain)
    if merged_source is not None:
        # All arrays share the same storage - use the merged source
        return Array._from_sources(
            sources=[merged_source],
            domain=domain,
            dtype=first.dtype,
            fill_value=fill_value,
        )

    return Array._from_sources(
        sources=sources,
        domain=domain,
        dtype=first.dtype,
        fill_value=fill_value,
    )
