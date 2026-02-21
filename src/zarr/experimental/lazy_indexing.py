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

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from itertools import product as itertools_product
from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np

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
    from zarr.storage import StoreLike


Region = tuple[tuple[int, ...], tuple[int, ...]]
"""Low-level representation of a rectangular region: ``(inclusive_min, exclusive_max)``.

Used as the key type for :class:`ChunkMap` to avoid object-creation overhead
during iteration.  Convertible to/from :class:`IndexDomain`.
"""


def _normalize_basic_selection(
    selection: BasicSelection, ndim: int
) -> tuple[int | slice, ...]:
    """Normalize a basic selection to a tuple of ints/slices with length *ndim*."""
    if not isinstance(selection, tuple):
        selection = (selection,)

    result: list[int | slice] = []
    ellipsis_seen = False
    for sel in selection:
        if sel is Ellipsis:
            if ellipsis_seen:
                raise IndexError("an index can only have a single ellipsis ('...')")
            ellipsis_seen = True
            num_missing = ndim - (len(selection) - 1)
            result.extend([slice(None)] * num_missing)
        else:
            result.append(sel)  # type: ignore[arg-type]

    while len(result) < ndim:
        result.append(slice(None))

    if len(result) > ndim:
        raise IndexError(
            f"too many indices for array: array has {ndim} dimensions, "
            f"but {len(result)} were indexed"
        )

    return tuple(result)


@dataclass(frozen=True, slots=True)
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

    def narrow(self, selection: BasicSelection) -> IndexDomain:
        """Apply a basic selection and return a narrowed domain.

        Indices are absolute coordinates in this domain's index space
        (TensorStore convention).  Negative indices mean negative coordinates,
        not "from the end".

        Integer indices produce a length-1 extent (the dimension is *not*
        dropped, unlike NumPy).
        """
        normalized = _normalize_basic_selection(selection, self.ndim)

        new_inclusive_min: list[int] = []
        new_exclusive_max: list[int] = []

        for dim_idx, (sel, dim_lo, dim_hi) in enumerate(
            zip(normalized, self.inclusive_min, self.exclusive_max, strict=True)
        ):
            if isinstance(sel, int):
                abs_idx = sel
                if abs_idx < dim_lo or abs_idx >= dim_hi:
                    raise IndexError(
                        f"index {sel} is out of bounds for dimension {dim_idx} "
                        f"with domain [{dim_lo}, {dim_hi})"
                    )
                new_inclusive_min.append(abs_idx)
                new_exclusive_max.append(abs_idx + 1)
            else:
                start, stop, step = sel.start, sel.stop, sel.step
                if step is not None and step != 1:
                    raise IndexError(
                        "lazy indexing only supports step=1 slices. "
                        f"Got step={step}. Use resolve() first for strided access."
                    )
                abs_start = dim_lo if start is None else start
                abs_stop = dim_hi if stop is None else stop
                abs_start = max(abs_start, dim_lo)
                abs_stop = min(abs_stop, dim_hi)
                abs_stop = max(abs_stop, abs_start)
                new_inclusive_min.append(abs_start)
                new_exclusive_max.append(abs_stop)

        return IndexDomain(
            inclusive_min=tuple(new_inclusive_min),
            exclusive_max=tuple(new_exclusive_max),
        )




# ---------------------------------------------------------------------------
# IndexTransform — domain + offset mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexTransform:
    """Maps coordinates from a user-facing domain to storage coordinates.

    For now, supports offset-only transforms:
        storage_coord[d] = user_coord[d] - offset[d]

    The ``domain`` is the input (user-facing) domain.  The ``offset`` defines
    the mapping.  This can later be extended to support stride, permutation,
    and index arrays.

    Parameters
    ----------
    domain : IndexDomain
        The input (user-facing) domain.
    offset : tuple[int, ...]
        Per-dimension offset: ``storage = user - offset``.
    """

    domain: IndexDomain
    offset: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.offset) != self.domain.ndim:
            raise ValueError(
                f"offset must have same length as domain dimensions. "
                f"Domain has {self.domain.ndim} dims, offset has {len(self.offset)}."
            )

    @classmethod
    def identity(cls, domain: IndexDomain) -> IndexTransform:
        """Create an identity transform (offset=0) for the given domain."""
        return cls(domain=domain, offset=(0,) * domain.ndim)

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> IndexTransform:
        """Create an identity transform for a zero-origin domain with given shape."""
        return cls.identity(IndexDomain.from_shape(shape))

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.domain.ndim

    @property
    def storage_origin(self) -> tuple[int, ...]:
        """Where ``domain.origin`` maps to in storage space."""
        return tuple(
            o - off for o, off in zip(self.domain.origin, self.offset, strict=True)
        )

    def narrow(self, selection: Any) -> IndexTransform:
        """Apply a basic selection, returning a new transform with narrowed domain.

        The offset is preserved — only the domain is narrowed.
        """
        new_domain = self.domain.narrow(selection)
        return IndexTransform(domain=new_domain, offset=self.offset)

    def compose_or_none(self, inner: IndexTransform) -> IndexTransform | None:
        """Compose this (outer) transform with an inner transform.

        The outer transform maps user coordinates to intermediate coordinates:
            intermediate = user - outer.offset

        The inner transform maps intermediate coordinates to storage:
            storage = intermediate - inner.offset

        The composed transform maps:
            storage = user - outer.offset - inner.offset

        To find the valid domain, we map the outer domain into intermediate
        space (subtract outer.offset), intersect with inner.domain
        (which is already in intermediate space), then map back to user space
        (add outer.offset).

        Returns ``None`` if the domains don't overlap in intermediate space.
        """
        neg_offset = tuple(-o for o in self.offset)
        outer_in_intermediate = self.domain.translate(neg_offset)
        intersection = outer_in_intermediate.intersect(inner.domain)
        if intersection is None:
            return None
        # Map intersection back to user space
        user_domain = intersection.translate(self.offset)
        composed_offset = tuple(
            a + b for a, b in zip(self.offset, inner.offset, strict=True)
        )
        return IndexTransform(domain=user_domain, offset=composed_offset)

    def __repr__(self) -> str:
        return f"IndexTransform(domain={self.domain}, offset={self.offset})"


# ---------------------------------------------------------------------------
# ArrayDesc — structural metadata for an array node
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArrayDesc:
    """Structural description of an array node.

    This is the subset of zarr v3 array metadata needed for indexing and I/O,
    without user-facing annotation (attributes, dimension names).

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of this node (may be the full array, a shard, or a single chunk).
    data_type : np.dtype
        The numpy dtype.
    chunk_grid : ChunkGrid | None
        How this node is subdivided into children. ``None`` for leaf nodes
        (single chunks with no further subdivision).
    encode_chunk_key : Callable[[tuple[int, ...]], str] | None
        Maps chunk grid coordinates to storage key strings.
        ``None`` for leaf nodes.
    fill_value : Any
        Fill value for missing data.
    codecs : tuple[Any, ...] | None
        Codec pipeline for encoding/decoding. ``None`` for virtual nodes.
    """

    shape: tuple[int, ...]
    data_type: np.dtype[Any]
    chunk_grid: Any  # ChunkGrid | None
    encode_chunk_key: Any  # Callable[[tuple[int, ...]], str] | None
    fill_value: Any
    codecs: tuple[Any, ...] | None

    @classmethod
    def from_metadata(
        cls,
        metadata: ArrayV2Metadata | ArrayV3Metadata,
    ) -> ArrayDesc:
        """Construct from zarr array metadata, extracting only what's needed
        for indexing and I/O."""
        from zarr.core.chunk_grids import RegularChunkGrid

        if hasattr(metadata, "data_type"):
            dtype = metadata.data_type.to_native_dtype()
        else:
            dtype = metadata.dtype.to_native_dtype()

        chunk_grid = metadata.chunk_grid if isinstance(metadata.chunk_grid, RegularChunkGrid) else None
        encode_key = metadata.encode_chunk_key if chunk_grid is not None else None

        return cls(
            shape=metadata.shape,
            data_type=dtype,
            chunk_grid=chunk_grid,
            encode_chunk_key=encode_key,
            fill_value=metadata.fill_value,
            codecs=tuple(metadata.codecs) if hasattr(metadata, "codecs") else None,
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)


# ---------------------------------------------------------------------------
# Layer / ZarrSource / ChunkEntry — per-source and per-chunk records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layer:
    """A positioned storage source.

    Each layer pairs an ``IndexTransform`` (positioning this layer in the
    combined coordinate space) with a ``ZarrSource`` (the storage backend).

    At resolve time, the Array's outer transform is composed with each
    layer's transform to get the full user-to-storage mapping.

    Parameters
    ----------
    transform : IndexTransform
        Positions this layer in the combined coordinate space.
    source : ZarrSource
        Storage backend for this layer.
    """

    transform: IndexTransform
    source: ZarrSource


@dataclass(frozen=True, slots=True)
class ZarrSource:
    """Immutable, self-contained description of one zarr array's storage.

    Carries everything needed to generate chunk reads for any sub-domain.
    Coordinate mapping is NOT stored here — it lives on ``Layer.transform``
    and ``Array._transform``.

    Parameters
    ----------
    store_path : StorePath
        Where the array bytes live.
    metadata : ArrayV2Metadata | ArrayV3Metadata
        Parsed zarr metadata.
    codec_pipeline : Any
        Codec pipeline (possibly fast-optimized) for decode/encode.
    config : ArrayConfig
        Array configuration.
    desc : ArrayDesc
        Structural description (shape, chunk grid, etc.).
    """

    store_path: StorePath
    metadata: ArrayV2Metadata | ArrayV3Metadata
    codec_pipeline: Any  # CodecPipeline
    config: ArrayConfig
    desc: ArrayDesc


@dataclass(frozen=True, slots=True)
class ChunkEntry:
    """Lightweight record for a single chunk within a ``ChunkMap``.

    Contains only what the resolve path needs.

    Parameters
    ----------
    domain : IndexDomain
        The chunk's domain (intersection of parent domain with chunk region).
    path : str
        Storage key for this chunk's bytes.
    chunk_selection : tuple[slice, ...]
        Chunk-local slices for decoding.
    chunk_coords : tuple[int, ...]
        Grid coordinates of this chunk.
    chunk_shape : tuple[int, ...]
        Actual shape of this chunk (may be truncated at boundary).
    """

    domain: IndexDomain
    path: str
    chunk_selection: tuple[slice, ...]
    chunk_coords: tuple[int, ...]
    chunk_shape: tuple[int, ...]


# ---------------------------------------------------------------------------
# Advanced indexing types
# ---------------------------------------------------------------------------


class SelectionKind(Enum):
    """The kind of advanced (non-basic) indexing operation."""

    ORTHOGONAL = auto()  # arr.oindex[[1,2], :, [3,4]]
    COORDINATE = auto()  # arr.vindex[([1,5], [2,4])]
    MASK = auto()  # arr[mask]


@dataclass(frozen=True, slots=True)
class PendingSelection:
    """A non-rectangular selection deferred until ``resolve()`` time.

    Stores the raw selection arrays/masks and precomputed output shape,
    but does NOT load any data.  The selection is applied at ``resolve()``
    by resolving the bounding-box region first, then indexing the dense
    result with numpy.

    Parameters
    ----------
    kind : SelectionKind
        The type of advanced selection.
    raw_selection : tuple[Any, ...]
        Normalized selection — numpy arrays, slices, or ints.
        For MASK: ``(mask_array,)``.
        For COORDINATE: ``(idx_array_dim0, idx_array_dim1, ...)``.
        For ORTHOGONAL: ``(per_dim_sel_0, per_dim_sel_1, ...)``.
    output_shape : tuple[int, ...]
        Precomputed shape of the resolved result.
    bounding_domain : IndexDomain
        Tightest rectangular bounding box covering the selected indices.
    """

    kind: SelectionKind
    raw_selection: tuple[Any, ...]
    output_shape: tuple[int, ...]
    bounding_domain: IndexDomain


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



def _try_merge_to_single_layer(
    all_layers: list[Layer],
    domain: IndexDomain,
) -> Layer | None:
    """
    Try to collapse multiple layers into a single layer.

    Succeeds when all layers share the same storage identity
    (store_path and transform offset) AND their domains fully cover the
    target domain (no gaps that should be filled with fill_value).
    """
    if not all_layers:
        return None

    first = all_layers[0]
    for layer in all_layers[1:]:
        if (
            layer.source.store_path != first.source.store_path
            or layer.transform.offset != first.transform.offset
        ):
            return None

    # Check coverage: total volume of layer domains must be >= target domain volume
    ndim = domain.ndim
    total_input_volume = sum(
        int(np.prod([
            lay.transform.domain.exclusive_max[i] - lay.transform.domain.inclusive_min[i]
            for i in range(ndim)
        ]))
        for lay in all_layers
    )
    bbox_volume = int(
        np.prod([domain.exclusive_max[i] - domain.inclusive_min[i] for i in range(ndim)])
    )

    if total_input_volume < bbox_volume:
        return None

    # All layers share same storage — use single layer with full domain
    return Layer(
        transform=IndexTransform(domain=domain, offset=first.transform.offset),
        source=first.source,
    )


class ChunkMap(Mapping[Region, Any]):
    """A lazy mapping from array regions to child arrays.

    Maps ``Region -> Array`` where each key is a ``(inclusive_min, exclusive_max)``
    tuple pair describing the chunk's full region in storage coordinates, and
    each value is a child ``Array`` representing that chunk (or shard).

    Keys are plain tuples — no object creation overhead during iteration.

    All spatial arithmetic is delegated to ``desc.chunk_grid``, so this works
    identically for regular and rectilinear grids.

    Parameters
    ----------
    desc : ArrayDesc
        Structural description of the *parent* array (must have a non-None
        ``chunk_grid``).
    domain : IndexDomain
        The region of interest (in the parent array's coordinate system).
    index_transform : tuple[int, ...]
        Offset from domain coordinates to storage coordinates:
        ``storage_coord = domain_coord - index_transform``.
    make_child : callable
        Factory ``(chunk_coords, chunk_selection, chunk_domain) -> Array``
        that creates the child array for a given chunk.  The ``chunk_selection``
        is a tuple of slices in chunk-local coordinates.  The ``chunk_domain``
        is an ``IndexDomain`` in the parent's storage coordinate space.
    """

    __slots__ = (
        "_chunk_end",
        "_chunk_start",
        "_empty",
        "_make_child",
        "_valid_hi",
        "_valid_lo",
        "desc",
        "domain",
        "index_transform",
    )

    desc: ArrayDesc
    domain: IndexDomain
    index_transform: tuple[int, ...]
    _make_child: Any  # Callable
    _valid_lo: tuple[int, ...]
    _valid_hi: tuple[int, ...]
    _chunk_start: tuple[int, ...]
    _chunk_end: tuple[int, ...]
    _empty: bool

    def __init__(
        self,
        desc: ArrayDesc,
        domain: IndexDomain,
        index_transform: tuple[int, ...],
        make_child: Any,
    ) -> None:
        chunk_grid = desc.chunk_grid
        storage_shape = desc.shape
        ndim = len(storage_shape)

        if domain.ndim != ndim:
            raise ValueError(
                f"Dimension mismatch: desc.shape has {ndim} dims, "
                f"domain has {domain.ndim} dims"
            )
        if len(index_transform) != ndim:
            raise ValueError(
                f"index_transform has {len(index_transform)} dims, expected {ndim}"
            )
        if chunk_grid is None:
            raise ValueError("Cannot create ChunkMap: desc.chunk_grid is None (leaf node)")

        self.desc = desc
        self.domain = domain
        self.index_transform = index_transform
        self._make_child = make_child

        # Translate domain to storage coords, clamp to [0, storage_dim)
        valid_lo = tuple(
            max(d_lo - it, 0)
            for d_lo, it in zip(domain.inclusive_min, index_transform, strict=True)
        )
        valid_hi = tuple(
            min(d_hi - it, s)
            for d_hi, it, s in zip(
                domain.exclusive_max, index_transform, storage_shape, strict=True
            )
        )
        self._empty = any(lo >= hi for lo, hi in zip(valid_lo, valid_hi, strict=True))
        self._valid_lo = valid_lo
        self._valid_hi = valid_hi

        if not self._empty:
            # Use chunk_grid to find the range of chunk coords that overlap
            chunk_start = chunk_grid.array_index_to_chunk_coord(storage_shape, valid_lo)
            # valid_hi is exclusive, so use valid_hi - 1 for the last element
            last_idx = tuple(h - 1 for h in valid_hi)
            last_chunk = chunk_grid.array_index_to_chunk_coord(storage_shape, last_idx)
            chunk_end = tuple(c + 1 for c in last_chunk)
        else:
            chunk_start = (0,) * ndim
            chunk_end = (0,) * ndim

        self._chunk_start = chunk_start
        self._chunk_end = chunk_end

    def _chunk_coords_iter(self) -> Iterator[tuple[int, ...]]:
        """Iterate over chunk grid coordinates (internal)."""
        if self._empty:
            return
        yield from itertools_product(
            *(range(s, e) for s, e in zip(self._chunk_start, self._chunk_end, strict=True))
        )

    def _region_to_chunk_coords(self, key: Region) -> tuple[int, ...] | None:
        """Map a Region key back to chunk grid coordinates, or None if invalid.

        Returns None if the region doesn't exactly match a chunk's bounds.
        """
        lo, hi = key
        chunk_grid = self.desc.chunk_grid
        storage_shape = self.desc.shape
        if len(lo) != len(storage_shape):
            return None
        coords = chunk_grid.array_index_to_chunk_coord(storage_shape, lo)
        # Check coords are in range
        if not all(
            s <= c < e
            for c, s, e in zip(coords, self._chunk_start, self._chunk_end, strict=True)
        ):
            return None
        # Verify the region matches the actual chunk bounds
        expected_start = chunk_grid.get_chunk_start(storage_shape, coords)
        expected_shape = chunk_grid.get_chunk_shape(storage_shape, coords)
        expected_end = tuple(
            s + cs for s, cs in zip(expected_start, expected_shape, strict=True)
        )
        if lo != expected_start or hi != expected_end:
            return None
        return coords

    def _get_by_chunk_coords(self, chunk_coords: tuple[int, ...]) -> Any:
        """Build and return the child for the given grid coordinates (internal)."""
        chunk_grid = self.desc.chunk_grid
        storage_shape = self.desc.shape
        valid_lo = self._valid_lo
        valid_hi = self._valid_hi

        chunk_storage_start = chunk_grid.get_chunk_start(storage_shape, chunk_coords)
        chunk_shape = chunk_grid.get_chunk_shape(storage_shape, chunk_coords)
        chunk_storage_end = tuple(
            s + cs for s, cs in zip(chunk_storage_start, chunk_shape, strict=True)
        )

        inter_lo = tuple(
            max(a, b) for a, b in zip(chunk_storage_start, valid_lo, strict=True)
        )
        inter_hi = tuple(
            min(a, b) for a, b in zip(chunk_storage_end, valid_hi, strict=True)
        )

        selection = tuple(
            slice(lo - cs, hi - cs)
            for lo, hi, cs in zip(inter_lo, inter_hi, chunk_storage_start, strict=True)
        )

        chunk_domain = IndexDomain(inclusive_min=inter_lo, exclusive_max=inter_hi)
        return self._make_child(chunk_coords, selection, chunk_domain)

    def __len__(self) -> int:
        if self._empty:
            return 0
        result = 1
        for s, e in zip(self._chunk_start, self._chunk_end, strict=True):
            result *= e - s
        return result

    def __iter__(self) -> Iterator[Region]:
        """Iterate over chunk regions as ``(inclusive_min, exclusive_max)`` tuples."""
        chunk_grid = self.desc.chunk_grid
        storage_shape = self.desc.shape
        for chunk_coords in self._chunk_coords_iter():
            start = chunk_grid.get_chunk_start(storage_shape, chunk_coords)
            shape = chunk_grid.get_chunk_shape(storage_shape, chunk_coords)
            end = tuple(s + cs for s, cs in zip(start, shape, strict=True))
            yield (start, end)

    def __contains__(self, key: object) -> bool:
        if self._empty:
            return False
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        lo, hi = key
        if not isinstance(lo, tuple) or not isinstance(hi, tuple):
            return False
        return self._region_to_chunk_coords(key) is not None  # type: ignore[arg-type]

    def __getitem__(self, key: Region) -> Any:
        """Return the child Array for the given chunk region."""
        if self._empty:
            raise KeyError(key)

        chunk_coords = self._region_to_chunk_coords(key)
        if chunk_coords is None:
            raise KeyError(key)

        return self._get_by_chunk_coords(chunk_coords)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.desc.ndim

    def __repr__(self) -> str:
        return (
            f"ChunkMap(domain={self.domain}, desc.shape={self.desc.shape}, "
            f"len={len(self)})"
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
    """Drop-in replacement for BytesCodec decode/encode that avoids:
    1. isinstance(x, NDArrayLike) Protocol check (~16 getattr_static calls)
    2. _batching_helper/concurrent_map/Semaphore overhead
    3. _noop_for_none wrapper and its typing introspection

    This is not a full Codec subclass — it only implements the decode/encode
    paths needed by the inner shard pipeline. It stores the endian
    configuration from the original BytesCodec.
    """

    __slots__ = ("_endian_str",)

    def __init__(self, endian_str: str | None) -> None:
        self._endian_str = endian_str

    def decode_single(self, chunk_bytes: Any, chunk_spec: Any) -> Any:
        """Synchronous fast decode — delegates to ``_decode_bytes_fast``."""
        return _decode_bytes_fast(self._endian_str, chunk_bytes, chunk_spec)

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



def _make_chunk_entry(
    *,
    base_path: str,
    parent_desc: ArrayDesc,
) -> Any:
    """Return a factory that creates ``ChunkEntry`` records for chunks.

    The factory is called by ``ChunkMap.__getitem__`` with
    ``(chunk_coords, selection, chunk_domain)`` and returns a ``ChunkEntry``.
    Only captures ``base_path`` and ``parent_desc`` — storage/codec info
    lives on the ``ZarrSource``.
    """

    def _factory(
        chunk_coords: tuple[int, ...],
        selection: tuple[slice, ...],
        chunk_domain: IndexDomain,
    ) -> ChunkEntry:
        encode_key = parent_desc.encode_chunk_key
        if encode_key is not None:
            key = encode_key(chunk_coords)
        else:
            key = "/".join(map(str, ("c",) + chunk_coords))

        path = f"{base_path}/{key}" if base_path else key
        chunk_shape = parent_desc.chunk_grid.get_chunk_shape(parent_desc.shape, chunk_coords)

        return ChunkEntry(
            domain=chunk_domain,
            path=path,
            chunk_selection=selection,
            chunk_coords=chunk_coords,
            chunk_shape=chunk_shape,
        )

    return _factory


# ---------------------------------------------------------------------------
# Lazy indexing accessors (oindex / vindex)
# ---------------------------------------------------------------------------


class LazyOIndex:
    """Lazy orthogonal indexing accessor.

    ``arr.oindex[[1, 2], :, [3, 4]]`` returns a lazy ``Array`` whose
    ``resolve()`` yields the outer-product selection.
    """

    __slots__ = ("_array",)

    def __init__(self, array: Array) -> None:
        self._array = array

    def __getitem__(self, selection: Any) -> Array:
        return self._array._apply_advanced_selection(selection, SelectionKind.ORTHOGONAL)


class LazyVIndex:
    """Lazy vectorized (coordinate) indexing accessor.

    ``arr.vindex[([1, 5], [2, 4])]`` returns a lazy ``Array`` whose
    ``resolve()`` yields the point selection.
    """

    __slots__ = ("_array",)

    def __init__(self, array: Array) -> None:
        self._array = array

    def __getitem__(self, selection: Any) -> Array:
        from zarr.core.indexing import is_bool_array, is_mask_selection

        sel_tuple = selection if isinstance(selection, tuple) else (selection,)
        if is_bool_array(selection) or (
            isinstance(selection, tuple) and is_mask_selection(sel_tuple, self._array.shape)
        ):
            return self._array._apply_advanced_selection(selection, SelectionKind.MASK)
        return self._array._apply_advanced_selection(selection, SelectionKind.COORDINATE)


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

    _transform: IndexTransform
    _layers: tuple[Layer, ...]
    _dtype: np.dtype[Any]
    _fill_value: Any
    _pending_selection: PendingSelection | None  # advanced indexing deferred to resolve()

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
        codec_pipeline = _make_fast_codec_pipeline(codec_pipeline)

        desc = ArrayDesc.from_metadata(metadata_parsed)

        # Default domain is origin at zero with shape from metadata
        if domain is None:
            domain = IndexDomain.from_shape(metadata_parsed.shape)

        # Default storage transform offset is zero (domain coords = storage coords)
        if index_transform is None:
            offset = (0,) * domain.ndim
        else:
            offset = index_transform

        source = ZarrSource(
            store_path=store_path,
            metadata=metadata_parsed,
            codec_pipeline=codec_pipeline,
            config=config_parsed,
            desc=desc,
        )

        # The layer covers the full storage with identity transform
        storage_domain = IndexDomain.from_shape(metadata_parsed.shape)
        layer = Layer(
            transform=IndexTransform.identity(storage_domain),
            source=source,
        )

        self._transform = IndexTransform(domain=domain, offset=offset)
        self._layers = (layer,)
        self._dtype = desc.data_type
        self._fill_value = desc.fill_value
        self._pending_selection = None

    @classmethod
    def _from_layers(
        cls,
        *,
        transform: IndexTransform,
        layers: tuple[Layer, ...],
        dtype: np.dtype[Any],
        fill_value: Any,
    ) -> Array:
        """Create an Array from a transform and positioned layers."""
        arr = object.__new__(cls)
        arr._transform = transform
        arr._layers = layers
        arr._dtype = dtype
        arr._fill_value = fill_value
        arr._pending_selection = None
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
        return self._transform.domain

    @property
    def layers(self) -> tuple[Layer, ...]:
        """The positioned storage layers backing this array."""
        return self._layers

    @property
    def source(self) -> ZarrSource | None:
        """The single ZarrSource if this is a single-layer array, else None."""
        if len(self._layers) == 1:
            return self._layers[0].source
        return None

    @property
    def origin(self) -> tuple[int, ...]:
        """The origin (inclusive lower bounds) of this array's domain."""
        return self._transform.domain.origin

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions in the Array."""
        if self._pending_selection is not None:
            return len(self._pending_selection.output_shape)
        return self._transform.domain.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the Array (from the domain, not metadata)."""
        if self._pending_selection is not None:
            return self._pending_selection.output_shape
        return self._transform.domain.shape

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
        """The combined offset for single-source storage arrays, or None."""
        if len(self._layers) != 1:
            return None
        composed = self._transform.compose_or_none(self._layers[0].transform)
        return composed.offset if composed is not None else None

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
        s = self.source
        if s is None:
            return None
        it = self.index_transform
        if it is None:
            return None
        return ChunkLayout(grid_origin=it, chunk_shape=s.metadata.chunks)

    @property
    def chunk_map(self) -> ChunkMap | None:
        """
        A lazy mapping from chunk coordinates to child arrays.

        Returns a :class:`ChunkMap` that maps ``IndexDomain -> Array`` for
        all chunks overlapping this array's domain.  Keys are the chunk regions
        (as ``IndexDomain`` objects in storage coordinates) and values are child
        ``Array`` nodes.

        Composes naturally with ``__getitem__``: narrowing the domain creates a
        new Array whose ``chunk_map`` reflects the narrowed view.

        Returns ``None`` for leaf nodes (single chunks) or virtual arrays.

        Returns
        -------
        ChunkMap | None
            The chunk map, or None for multi-source arrays.

        Examples
        --------
        >>> arr = Array.open("path/to/array")  # shape (100,), chunks (10,)
        >>> len(arr.chunk_map)  # 10 chunks
        10
        >>> list(arr.chunk_map.keys())[:3]
        [IndexDomain([0, 10)), IndexDomain([10, 20)), IndexDomain([20, 30))]
        """
        if len(self._layers) != 1:
            return None
        layer = self._layers[0]
        s = layer.source
        if s.desc.chunk_grid is None:
            return None
        composed = self._transform.compose_or_none(layer.transform)
        if composed is None:
            return None
        return ChunkMap(
            desc=s.desc,
            domain=self._transform.domain,
            index_transform=composed.offset,
            make_child=_make_chunk_entry(
                base_path=s.store_path.path,
                parent_desc=s.desc,
            ),
        )

    @property
    def oindex(self) -> LazyOIndex:
        """Lazy orthogonal indexing accessor.

        Usage: ``arr.oindex[[1, 2], :, [3, 4]]`` returns a lazy Array.
        """
        return LazyOIndex(self)

    @property
    def vindex(self) -> LazyVIndex:
        """Lazy vectorized (coordinate) indexing accessor.

        Usage: ``arr.vindex[([1, 5], [2, 4])]`` returns a lazy Array.
        """
        return LazyVIndex(self)

    @property
    def size(self) -> int:
        """Returns the total number of elements in the array."""
        return product(self.shape)

    @property
    def cdata_shape(self) -> tuple[int, ...] | None:
        """The shape of the chunk grid for this array."""
        return self._chunk_grid_shape

    @property
    def _chunk_grid_shape(self) -> tuple[int, ...] | None:
        """The shape of the chunk grid for this array."""
        s = self.source
        if s is None:
            return None
        return tuple(starmap(ceildiv, zip(self.shape, s.metadata.chunks, strict=True)))

    @property
    def _shard_grid_shape(self) -> tuple[int, ...] | None:
        """The shape of the shard grid for this array."""
        s = self.source
        if s is None:
            return None
        shard_shape = s.metadata.shards if s.metadata.shards is not None else s.metadata.chunks
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
        if self.source is None:
            return None
        return self._info()

    def _info(
        self, count_chunks_initialized: int | None = None, count_bytes_stored: int | None = None
    ) -> ArrayInfo | None:
        s = self.source
        if s is None:
            return None
        m = s.metadata

        # Extract zdtype (v2 vs v3)
        if m.zarr_format == 2:
            zdtype = m.dtype
        else:
            zdtype = m.data_type

        # Extract order (v2 vs v3)
        if m.zarr_format == 2:
            order = m.order
        else:
            order = s.config.order

        # Extract filters (v2 vs v3)
        if m.zarr_format == 2:
            from zarr.abc.numcodec import Numcodec

            filters: tuple[Numcodec, ...] | tuple[Any, ...] = m.filters if m.filters is not None else ()
        else:
            from zarr.abc.codec import ArrayArrayCodec

            filters = tuple(c for c in m.inner_codecs if isinstance(c, ArrayArrayCodec))

        # Extract serializer (v3 only)
        if m.zarr_format == 2:
            serializer = None
        else:
            from zarr.abc.codec import ArrayBytesCodec

            serializer = next(c for c in m.inner_codecs if isinstance(c, ArrayBytesCodec))

        # Extract compressors (v2 vs v3)
        if m.zarr_format == 2:
            compressors: tuple[Any, ...] = (m.compressor,) if m.compressor is not None else ()
        else:
            from zarr.abc.codec import BytesBytesCodec

            compressors = tuple(c for c in m.inner_codecs if isinstance(c, BytesBytesCodec))

        return ArrayInfo(
            _zarr_format=m.zarr_format,
            _data_type=zdtype,
            _fill_value=self._fill_value,
            _shape=self.shape,
            _order=order,
            _shard_shape=m.shards,
            _chunk_shape=m.chunks,
            _read_only=s.store_path.read_only,
            _compressors=compressors,
            _filters=filters,
            _serializer=serializer,
            _store_type=type(s.store_path.store).__name__,
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
        s = self.source
        if s is None:
            raise ValueError("nbytes_stored requires a single-source array")
        return await _nbytes_stored(s.store_path)

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
        s = self.source
        if s is None:
            raise ValueError("getitem requires a single-source array")
        return await _getitem(
            s.store_path,
            s.metadata,
            s.codec_pipeline,
            s.config,
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

    def __getitem__(self, selection: Any) -> Self:
        """
        Lazy indexing: returns a new Array with a narrowed domain.

        Supports both basic indexing (slices, ints) and advanced indexing
        (boolean masks, integer arrays).  In either case, no I/O is performed —
        call ``resolve()`` to load data.

        Basic indexing narrows the rectangular domain.  Advanced indexing
        stores a :class:`PendingSelection` that is applied at resolve time.

        Indices are **absolute coordinates** in the domain's index space
        (TensorStore convention).  Negative indices mean negative coordinates,
        not "from the end".

        Parameters
        ----------
        selection
            Basic: ``int``, ``slice``, ``Ellipsis``, or tuple thereof.
            Advanced: boolean mask (``ndarray[bool]``), integer array, or
            tuple mixing arrays with slices/ints.

        Returns
        -------
        Array
            A new lazy Array.
        """
        from zarr.core.indexing import (
            is_bool_array,
            is_bool_list,
            is_integer_array,
            is_integer_list,
            is_pure_fancy_indexing,
            is_pure_orthogonal_indexing,
        )

        # --- Detect advanced indexing ---
        sel_tuple = selection if isinstance(selection, tuple) else (selection,)
        has_advanced = any(
            is_integer_array(s) or is_integer_list(s) or is_bool_array(s) or is_bool_list(s)
            for s in sel_tuple
        )

        if has_advanced:
            # Determine the kind of advanced indexing
            domain_shape = self._transform.domain.shape  # use domain shape, not self.shape
            if is_bool_array(selection) and np.asarray(selection).shape == domain_shape:
                return self._apply_advanced_selection(selection, SelectionKind.MASK)
            if is_pure_fancy_indexing(selection, self._transform.domain.ndim):
                # Check if it's a mask selection (tuple of one bool array)
                if (
                    isinstance(selection, tuple)
                    and len(sel_tuple) == 1
                    and is_bool_array(sel_tuple[0])
                    and np.asarray(sel_tuple[0]).shape == domain_shape
                ):
                    return self._apply_advanced_selection(
                        sel_tuple[0], SelectionKind.MASK
                    )
                return self._apply_advanced_selection(
                    selection, SelectionKind.COORDINATE
                )
            if is_pure_orthogonal_indexing(selection, self._transform.domain.ndim):
                return self._apply_advanced_selection(
                    selection, SelectionKind.ORTHOGONAL
                )
            # Fallback: treat as orthogonal
            return self._apply_advanced_selection(
                selection, SelectionKind.ORTHOGONAL
            )

        # --- Basic indexing path ---
        if self._pending_selection is not None:
            raise IndexError(
                "Cannot apply basic indexing to an array with a pending "
                "advanced selection. Call resolve() first."
            )
        new_transform = self._transform.narrow(selection)
        return self._with_transform(new_transform)

    # -------------------------------------------------------------------------
    # Advanced indexing
    # -------------------------------------------------------------------------

    def _apply_advanced_selection(self, selection: Any, kind: SelectionKind) -> Self:
        """Create a new Array with a pending advanced selection.

        This does NOT load data — it normalizes the selection, computes
        the output shape and bounding domain, and attaches a
        :class:`PendingSelection` to the returned Array.
        """
        if self._pending_selection is not None:
            raise IndexError(
                "Cannot apply advanced indexing to an array that already has "
                "a pending advanced selection. Call resolve() first."
            )

        domain = self._transform.domain

        if kind == SelectionKind.MASK:
            mask = np.asarray(selection, dtype=bool)
            if mask.shape != domain.shape:
                raise IndexError(
                    f"Boolean mask shape {mask.shape} doesn't match "
                    f"array shape {domain.shape}"
                )
            output_shape = (int(np.count_nonzero(mask)),)
            bounding_domain = self._mask_bounding_domain(mask, domain)
            # Crop the mask to the bounding domain so raw_selection aligns
            # with the narrowed domain (bbox_data) at resolve time.
            origin = domain.inclusive_min
            bbox_crop = tuple(
                slice(
                    bounding_domain.inclusive_min[d] - origin[d],
                    bounding_domain.exclusive_max[d] - origin[d],
                )
                for d in range(domain.ndim)
            )
            raw_selection = (mask[bbox_crop],)

        elif kind == SelectionKind.COORDINATE:
            raw_selection, output_shape, bounding_domain = (
                self._normalize_coordinate_selection(selection, domain)
            )

        elif kind == SelectionKind.ORTHOGONAL:
            raw_selection, output_shape, bounding_domain = (
                self._normalize_orthogonal_selection(selection, domain)
            )

        else:
            raise ValueError(f"Unknown selection kind: {kind}")

        pending = PendingSelection(
            kind=kind,
            raw_selection=raw_selection,
            output_shape=output_shape,
            bounding_domain=bounding_domain,
        )

        # Narrow the transform to the bounding box, then attach pending selection
        narrowed = IndexTransform(domain=bounding_domain, offset=self._transform.offset)
        result = self._with_transform(narrowed)
        result._pending_selection = pending
        return result

    @staticmethod
    def _mask_bounding_domain(mask: np.ndarray[Any, Any], domain: IndexDomain) -> IndexDomain:
        """Compute the tightest rectangular bounding box around True values."""
        indices = np.nonzero(mask)
        if any(len(idx) == 0 for idx in indices):
            # Empty selection — zero-size domain
            return IndexDomain(
                inclusive_min=domain.inclusive_min,
                exclusive_max=domain.inclusive_min,
            )
        origin = domain.inclusive_min
        lo = tuple(int(idx.min()) + origin[d] for d, idx in enumerate(indices))
        hi = tuple(int(idx.max()) + 1 + origin[d] for d, idx in enumerate(indices))
        return IndexDomain(inclusive_min=lo, exclusive_max=hi)

    @staticmethod
    def _normalize_coordinate_selection(
        selection: Any,
        domain: IndexDomain,
    ) -> tuple[tuple[Any, ...], tuple[int, ...], IndexDomain]:
        """Normalize coordinate (vindex) selection.

        Returns ``(raw_selection, output_shape, bounding_domain)``.
        """
        sel_tuple = selection if isinstance(selection, tuple) else (selection,)
        origin = domain.inclusive_min

        # Convert lists to arrays
        arrays: list[np.ndarray[Any, Any]] = []
        for i, s in enumerate(sel_tuple):
            arr = np.asarray(s)
            if arr.dtype == bool:
                # Bool in coordinate context → convert to indices
                arr = np.nonzero(arr)[0] + origin[i]
            arrays.append(arr)

        # Broadcast all arrays to same shape
        broadcasted = np.broadcast_arrays(*arrays)
        flat_len = broadcasted[0].size
        output_shape = (flat_len,)

        # Compute bounding box
        lo = list(domain.inclusive_min)
        hi = list(domain.exclusive_max)
        for d, arr in enumerate(broadcasted):
            if arr.size > 0:
                lo[d] = max(int(arr.min()), domain.inclusive_min[d])
                hi[d] = min(int(arr.max()) + 1, domain.exclusive_max[d])
            else:
                hi[d] = lo[d]

        bounding = IndexDomain(inclusive_min=tuple(lo), exclusive_max=tuple(hi))
        raw_selection = tuple(broadcasted)
        return raw_selection, output_shape, bounding

    @staticmethod
    def _normalize_orthogonal_selection(
        selection: Any,
        domain: IndexDomain,
    ) -> tuple[tuple[Any, ...], tuple[int, ...], IndexDomain]:
        """Normalize orthogonal (oindex) selection.

        Returns ``(raw_selection, output_shape, bounding_domain)``.
        """
        sel_tuple = selection if isinstance(selection, tuple) else (selection,)
        ndim = domain.ndim

        # Pad with full slices for missing trailing dims
        if len(sel_tuple) < ndim:
            sel_tuple = sel_tuple + (slice(None),) * (ndim - len(sel_tuple))

        # Handle ellipsis
        expanded: list[Any] = []
        for s in sel_tuple:
            if s is Ellipsis:
                n_expand = ndim - (len(sel_tuple) - 1)
                expanded.extend([slice(None)] * n_expand)
            else:
                expanded.append(s)
        sel_tuple = tuple(expanded[:ndim])

        normalized: list[Any] = []
        shape_parts: list[int] = []
        lo = list(domain.inclusive_min)
        hi = list(domain.exclusive_max)

        for d, s in enumerate(sel_tuple):
            dim_lo = domain.inclusive_min[d]
            dim_hi = domain.exclusive_max[d]
            dim_size = dim_hi - dim_lo

            if isinstance(s, (int, np.integer)):
                # Single integer — selects one element, dimension dropped in output
                # but we keep it in the selection for later indexing
                idx = int(s)
                normalized.append(np.array([idx]))
                shape_parts.append(1)
                lo[d] = max(idx, dim_lo)
                hi[d] = min(idx + 1, dim_hi)

            elif isinstance(s, slice):
                start, stop, step = s.start, s.stop, s.step
                if start is None:
                    start = dim_lo
                if stop is None:
                    stop = dim_hi
                if step is None:
                    step = 1
                # Clamp to domain
                start = max(start, dim_lo)
                stop = min(stop, dim_hi)
                rng = range(start, stop, step)
                normalized.append(slice(start, stop, step))
                shape_parts.append(len(rng))
                if len(rng) > 0:
                    lo[d] = rng[0]
                    hi[d] = rng[-1] + 1
                else:
                    hi[d] = lo[d]

            else:
                # Array-like
                arr = np.asarray(s)
                if arr.dtype == bool:
                    if len(arr) != dim_size:
                        raise IndexError(
                            f"Boolean array length {len(arr)} doesn't match "
                            f"dimension size {dim_size} for axis {d}"
                        )
                    # Convert to integer indices in domain coords
                    int_idx = np.nonzero(arr)[0] + dim_lo
                    normalized.append(int_idx)
                    shape_parts.append(len(int_idx))
                    if len(int_idx) > 0:
                        lo[d] = max(int(int_idx.min()), dim_lo)
                        hi[d] = min(int(int_idx.max()) + 1, dim_hi)
                    else:
                        hi[d] = lo[d]
                else:
                    int_arr = np.asarray(arr, dtype=np.intp)
                    normalized.append(int_arr)
                    shape_parts.append(len(int_arr))
                    if len(int_arr) > 0:
                        lo[d] = max(int(int_arr.min()), dim_lo)
                        hi[d] = min(int(int_arr.max()) + 1, dim_hi)
                    else:
                        hi[d] = lo[d]

        bounding = IndexDomain(inclusive_min=tuple(lo), exclusive_max=tuple(hi))
        return tuple(normalized), tuple(shape_parts), bounding

    def _with_transform(self, new_transform: IndexTransform) -> Self:
        """Create a new Array with a different outer transform.

        Layers are unchanged. Only the user-facing transform changes.
        This is the single unified code path for both single-source and
        multi-source arrays — no branching needed.
        """
        return self.__class__._from_layers(
            transform=new_transform,
            layers=self._layers,
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
        if self._pending_selection is not None:
            raise IndexError(
                "Cannot change domain on an array with a pending advanced "
                "selection. Call resolve() first."
            )
        if new_domain.ndim != self._transform.domain.ndim:
            raise ValueError(
                f"New domain must have same number of dimensions as array. "
                f"Array has {self._transform.domain.ndim} dimensions, new domain has {new_domain.ndim}."
            )
        # Set offset to the new domain's origin so that
        # domain.origin maps to storage coordinate 0
        new_transform = IndexTransform(domain=new_domain, offset=new_domain.origin)
        return self._with_transform(new_transform)

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
        if self._pending_selection is not None:
            return await self._resolve_with_advanced_selection(prototype)

        output = np.full(self._transform.domain.shape, self._fill_value, dtype=self._dtype)

        if prototype is None:
            prototype = default_buffer_prototype()

        for layer in self._layers:
            await self._resolve_layer(layer, output, prototype)

        return output

    async def _resolve_with_advanced_selection(
        self,
        prototype: BufferPrototype | None,
    ) -> NDArrayLikeOrScalar:
        """Resolve an array with a pending advanced selection.

        Two-step approach:
        1. Resolve the bounding-box domain to a dense numpy array.
        2. Apply the advanced selection to the dense array using numpy indexing.
        """
        pending = self._pending_selection
        assert pending is not None

        # Step 1: resolve the rectangular bounding box
        # Create a temporary copy without the pending selection
        temp = self._copy_without_pending()
        bbox_data = await temp.resolve_async(prototype=prototype)

        # Step 2: apply the advanced selection to the dense array
        # Translate indices from domain coords to bbox-local (zero-based) coords
        bbox_origin = pending.bounding_domain.inclusive_min

        if pending.kind == SelectionKind.MASK:
            # The mask was pre-cropped to the bounding domain in
            # _apply_advanced_selection, so mask.shape == bbox_data.shape.
            mask = pending.raw_selection[0]
            return bbox_data[mask]

        elif pending.kind == SelectionKind.ORTHOGONAL:
            local_sel = self._translate_orthogonal_to_local(
                pending.raw_selection, bbox_origin
            )
            # Use np.ix_ to create an open mesh for orthogonal indexing
            ix_args = []
            for s in local_sel:
                if isinstance(s, np.ndarray):
                    ix_args.append(s)
                elif isinstance(s, slice):
                    start = s.start if s.start is not None else 0
                    stop = s.stop if s.stop is not None else 0
                    step = s.step if s.step is not None else 1
                    ix_args.append(np.arange(start, stop, step))
                else:
                    ix_args.append(np.array([s]))
            return bbox_data[np.ix_(*ix_args)]

        elif pending.kind == SelectionKind.COORDINATE:
            local_sel = self._translate_coordinate_to_local(
                pending.raw_selection, bbox_origin
            )
            return bbox_data[tuple(local_sel)]

        raise ValueError(f"Unknown selection kind: {pending.kind}")

    def _copy_without_pending(self) -> Array:
        """Return a copy of this Array with ``_pending_selection`` cleared."""
        return self._with_transform(self._transform)

    @staticmethod
    def _translate_orthogonal_to_local(
        raw_selection: tuple[Any, ...],
        bbox_origin: tuple[int, ...],
    ) -> tuple[Any, ...]:
        """Translate orthogonal selection indices to bbox-local (zero-based) coords."""
        result: list[Any] = []
        for d, s in enumerate(raw_selection):
            off = bbox_origin[d]
            if isinstance(s, np.ndarray):
                result.append(s - off)
            elif isinstance(s, slice):
                start = (s.start - off) if s.start is not None else None
                stop = (s.stop - off) if s.stop is not None else None
                result.append(slice(start, stop, s.step))
            elif isinstance(s, (int, np.integer)):
                result.append(int(s) - off)
            else:
                result.append(s)
        return tuple(result)

    @staticmethod
    def _translate_coordinate_to_local(
        raw_selection: tuple[Any, ...],
        bbox_origin: tuple[int, ...],
    ) -> tuple[Any, ...]:
        """Translate coordinate selection indices to bbox-local (zero-based) coords."""
        return tuple(
            arr - bbox_origin[d] if isinstance(arr, np.ndarray) else arr
            for d, arr in enumerate(raw_selection)
        )

    async def _decode_and_scatter(
        self,
        *,
        codec_pipeline: Any,
        store: Any,
        metadata: ArrayV2Metadata | ArrayV3Metadata,
        config: ArrayConfig,
        chunk_paths: list[str],
        chunk_selections: list[tuple[slice, ...]],
        out_selections: list[tuple[slice, ...]],
        output: np.ndarray[Any, Any],
        prototype: BufferPrototype,
    ) -> None:
        """Decode chunks and scatter the results into *output*.

        This is the shared tail of ``_resolve_via_chunk_map`` and
        ``_resolve_storage_source``: given pre-collected chunk paths,
        chunk-local selections, and output-space selections, decode
        everything in one batch and write the results into *output*.
        """
        if not chunk_paths:
            return

        chunk_spec = metadata.get_chunk_spec(
            (0,) * len(output.shape), config, prototype=prototype
        )

        chunk_arrays = await _decode_chunks_with_selection(
            codec_pipeline,
            store,
            chunk_paths,
            chunk_spec,
            chunk_selections,
            prototype,
        )

        fill_value = self._fill_value
        for chunk_array, out_sel in zip(chunk_arrays, out_selections, strict=True):
            if chunk_array is not None:
                output[out_sel] = chunk_array.as_ndarray_like()
            else:
                output[out_sel] = fill_value

    async def _resolve_layer(
        self,
        layer: Layer,
        output: np.ndarray[Any, Any],
        prototype: BufferPrototype,
    ) -> None:
        """Resolve data from a single Layer into the output.

        Composes the Array's outer transform with the layer's transform
        to get the full user-to-storage mapping, then builds a ChunkMap
        and does batched decode.
        """
        source = layer.source
        desc = source.desc
        if desc.chunk_grid is None:
            return

        # Compose outer transform with layer transform
        composed = self._transform.compose_or_none(layer.transform)
        if composed is None:
            return  # No overlap between user domain and this layer

        # composed.domain = intersection of user domain and layer domain
        # composed.offset = outer.offset + layer.offset
        # storage_coord = user_coord - composed.offset

        chunk_map = ChunkMap(
            desc=desc,
            domain=composed.domain,
            index_transform=composed.offset,
            make_child=_make_chunk_entry(
                base_path=source.store_path.path,
                parent_desc=desc,
            ),
        )

        # Output offset: translate from storage coords to output-array coords
        output_origin = self._transform.domain.inclusive_min
        storage_to_output = tuple(
            off - oo for off, oo in zip(composed.offset, output_origin, strict=True)
        )

        # Collect chunk info for batched decode
        chunk_paths: list[str] = []
        chunk_selections: list[tuple[slice, ...]] = []
        out_selections: list[tuple[slice, ...]] = []

        for cc in chunk_map._chunk_coords_iter():
            entry = chunk_map._get_by_chunk_coords(cc)

            chunk_paths.append(entry.path)
            chunk_selections.append(entry.chunk_selection)

            out_selections.append(
                tuple(
                    slice(lo + off, hi + off)
                    for lo, hi, off in zip(
                        entry.domain.inclusive_min,
                        entry.domain.exclusive_max,
                        storage_to_output,
                        strict=True,
                    )
                )
            )

        await self._decode_and_scatter(
            codec_pipeline=source.codec_pipeline,
            store=source.store_path.store,
            metadata=source.metadata,
            config=source.config,
            chunk_paths=chunk_paths,
            chunk_selections=chunk_selections,
            out_selections=out_selections,
            output=output,
            prototype=prototype,
        )

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
        s = self.source
        if s is None:
            raise ValueError("setitem requires a single-source array")
        return await _setitem(
            s.store_path,
            s.metadata,
            s.codec_pipeline,
            s.config,
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
        s = self.source
        if s is None:
            raise ValueError("get_orthogonal_selection requires a single-source array")
        return await _get_orthogonal_selection(
            s.store_path,
            s.metadata,
            s.codec_pipeline,
            s.config,
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
        s = self.source
        if s is None:
            raise ValueError("get_mask_selection requires a single-source array")
        return await _get_mask_selection(
            s.store_path,
            s.metadata,
            s.codec_pipeline,
            s.config,
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
        s = self.source
        if s is None:
            raise ValueError("get_coordinate_selection requires a single-source array")
        return await _get_coordinate_selection(
            s.store_path,
            s.metadata,
            s.codec_pipeline,
            s.config,
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
        s = self.source
        if s is not None:
            return f"<Array {s.store_path} domain={self._transform.domain} dtype={self._dtype}>"
        else:
            return f"<Array domain={self._transform.domain} dtype={self._dtype} sources={len(self._layers)}>"

    def __eq__(self, other: object) -> bool:
        """
        Check equality between two Arrays.

        Two Arrays are equal if they have the same transform, dtype, fill_value,
        and layers.
        """
        if not isinstance(other, Array):
            return NotImplemented

        return (
            self._transform == other._transform
            and self._dtype == other._dtype
            and self._fill_value == other._fill_value
            and self._layers == other._layers
        )

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

    for arr in arrays:
        if arr._pending_selection is not None:
            raise ValueError(
                "Cannot merge arrays with pending advanced selections. "
                "Call resolve() on each array first."
            )

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

    # Flatten: compose each child's outer transform into its layers
    all_layers: list[Layer] = []
    for arr in arrays_list:
        for layer in arr._layers:
            composed = arr._transform.compose_or_none(layer.transform)
            if composed is not None:
                all_layers.append(Layer(transform=composed, source=layer.source))

    # Try to collapse to a single layer if all share same storage
    merged_layer = _try_merge_to_single_layer(all_layers, domain)
    if merged_layer is not None:
        layers = (merged_layer,)
    else:
        layers = tuple(all_layers)

    outer_transform = IndexTransform.identity(domain)

    return Array._from_layers(
        transform=outer_transform,
        layers=layers,
        dtype=first.dtype,
        fill_value=fill_value,
    )
