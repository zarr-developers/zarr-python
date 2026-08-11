"""Index transforms — composable, lazy coordinate mappings.

An `IndexTransform` pairs an **input domain** (the coordinates a user sees)
with a tuple of **output maps** (the output coordinates those inputs map to).
One output map per output dimension. See `output_map.py` for the three
output map types.

Key operations:

- **Indexing** (`transform[2:8]`, `.oindex[idx]`, `.vindex[idx]`) —
  produces a new transform with a narrower input domain and adjusted output
  maps. No I/O occurs. This is how lazy slicing works.

- **intersect(output_domain)** — restrict to output coordinates within a
  region. This is chunk resolution: "which of my coordinates fall in this
  chunk?"

- **translate(shift)** — shift all output coordinates. This makes coordinates
  chunk-local: "express my coordinates relative to the chunk origin."

- **`transform.compose(inner)`** — chain two transforms into one.

The transform is the atomic unit that connects user-facing indexing to
chunk-level I/O. A wrapper holds one — `LazyArray` starts from the identity —
and `.lazy[...]` composes a new transform lazily rather than reading. Reading
resolves the transform against the chunk grid via intersect + translate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from zarr_indexing._affine import checked_affine
from zarr_indexing._selector import as_scalar_index, require_index
from zarr_indexing.boundary import validate_advanced_selection
from zarr_indexing.domain import IndexDomain
from zarr_indexing.errors import BoundsCheckError, VindexInvalidSelectionError
from zarr_indexing.output_map import (
    ArrayMap,
    ConstantMap,
    DimensionMap,
    OutputIndexMap,
    array_map_or_constant,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from zarr_indexing.json import IndexTransformJSON


@dataclass(frozen=True, slots=True)
class _PointOutOfBounds(Exception):
    """Internal signal from the shared point kernel: one coordinate left the domain.

    Never escapes this module. `apply` and `apply_many` format it as the
    public `BoundsCheckError`, each in its own vocabulary — the kernel knows
    batches, but a single-point caller must never hear about them.
    """

    dimension: int
    value: int
    lower: int
    upper: int
    batch_position: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IndexTransform:
    """A composable mapping from input coordinates to output coordinates.

    An `IndexTransform` has:

    - `domain`: an `IndexDomain` describing the valid input coordinates
      (the result's coordinate range, possibly with non-zero origin).
    - `output`: a tuple of output maps (one per output dimension), each
      describing which output coordinates the inputs touch.

    In array-indexing terms: `domain` describes the coordinates of the result
    array an indexing operation produces, and `output` is the rule relating
    each result coordinate to a coordinate in the source. Note the direction —
    the transform's input side is the result, its output side addresses the
    source; the coordinate mapping runs opposite to the data flow.

    Indexing an existing transform composes a new one without I/O.

    Examples
    --------
    The operation "every other element of a 100-element array, starting at
    index 0" — `array[::2]` — is a 50-cell domain whose cell `i` reads
    output coordinate `2 * i`:

    >>> domain = IndexDomain.from_shape((50,))
    >>> output = (DimensionMap(input_dimension=0, offset=0, stride=2),)
    >>> transform = IndexTransform(domain=domain, output=output)
    >>> transform.apply((0,)), transform.apply((1,)), transform.apply((49,))
    ((0,), (2,), (98,))

    The selection compiler derives the identical transform from the source's
    shape and the slice:

    >>> transform == IndexTransform.from_shape((100,))[::2]
    True
    """

    domain: IndexDomain
    """The input domain: the request coordinates this transform accepts."""

    output: tuple[OutputIndexMap, ...]
    """One output map per output dimension, each producing that dimension's coordinate."""

    def __post_init__(self) -> None:
        for i, m in enumerate(self.output):
            if isinstance(m, DimensionMap):
                if m.input_dimension < 0 or m.input_dimension >= self.domain.ndim:
                    raise ValueError(
                        f"output[{i}].input_dimension = {m.input_dimension} "
                        f"is out of range for input rank {self.domain.ndim}"
                    )
            elif isinstance(m, ArrayMap):
                # An index array carries the transform's full input rank: the axis
                # a map varies over is full-sized, every other axis a singleton.
                # The rank is what makes the dependency axes readable from the
                # shape, so a mismatch is a bug rather than a spelling. External
                # JSON may use a lower-rank array that broadcasts against the
                # domain; `from_json` widens those on the way in,
                # so the invariant holds for every transform that exists.
                if m.index_array.ndim != self.domain.ndim:
                    raise ValueError(
                        f"output[{i}].index_array has {m.index_array.ndim} dims "
                        f"but input domain has {self.domain.ndim} dims"
                    )
                # Every axis is either the domain's extent or a singleton it
                # broadcasts over. Any other size addresses input coordinates the
                # array has no entry for, which reads as a smaller selection
                # rather than as the error it is.
                bad = [
                    (axis, size, extent)
                    for axis, (size, extent) in enumerate(
                        zip(m.index_array.shape, self.domain.shape, strict=True)
                    )
                    if size not in (1, extent)
                ]
                if len(bad) > 0:
                    axis, size, extent = bad[0]
                    raise ValueError(
                        f"output[{i}].index_array has {size} entries on axis {axis}, "
                        f"which is neither 1 nor the domain's extent of {extent} "
                        f"(index_array shape {m.index_array.shape}, "
                        f"domain shape {self.domain.shape})"
                    )

    def __eq__(self, other: object) -> bool:
        """Value equality. `ArrayMap` compares its index array element-wise, so
        a transform holding one can be compared at all — the generated `__eq__`
        raised `ValueError: the truth value of an array ... is ambiguous`."""
        if not isinstance(other, IndexTransform):
            return NotImplemented
        return self.domain == other.domain and self.output == other.output

    def __hash__(self) -> int:
        """Hashed by value, so a transform can key a cache or enter a set."""
        return hash((self.domain, self.output))

    @property
    def input_rank(self) -> int:
        """Number of input dimensions — the rank of `domain`."""
        return self.domain.ndim

    @property
    def output_rank(self) -> int:
        """Number of output dimensions — one per output map."""
        return len(self.output)

    @classmethod
    def identity(cls, domain: IndexDomain) -> IndexTransform:
        """The identity transform over `domain`: every result cell reads the source at its own address."""
        output = tuple(DimensionMap(input_dimension=i) for i in range(domain.ndim))
        return cls(domain=domain, output=output)

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> IndexTransform:
        """The identity transform over a zero-origin domain of the given `shape`."""
        return cls.identity(IndexDomain.from_shape(shape))

    def apply(self, point: Sequence[int]) -> tuple[int, ...]:
        """Map one coordinate of `domain` to the source coordinate that fills it.

        In array-indexing terms: `point` names a cell of the result array, and
        the returned tuple — each `output` map evaluated at `point` — names the
        source-array cell its value is read from: the coordinate arrow,
        running result to source.

        Parameters
        ----------
        point : Sequence[int]
            One literal coordinate for each input dimension.

        Returns
        -------
        tuple[int, ...]
            One coordinate for each output map.

        Raises
        ------
        ValueError
            If ``point`` does not have exactly one coordinate per input
            dimension.
        TypeError
            If the coordinates do not have an integer dtype.
        BoundsCheckError
            If a coordinate lies outside the input domain.
        OverflowError
            If a mapped output coordinate cannot be represented by
            ``np.intp``.

        Examples
        --------
        The `[::2]` transform reads result cell `i` from source coordinate
        `2 * i`, so cell 3 of the result holds `source[6]`:

        >>> transform = IndexTransform.from_shape((100,))[::2]
        >>> transform.apply((3,))
        (6,)
        """
        coordinates = np.asarray(point)
        expected_shape = (self.input_rank,)
        if coordinates.shape != expected_shape:
            raise ValueError(f"point must have shape {expected_shape}, got {coordinates.shape}")
        # An empty Python sequence has no elements from which NumPy can infer
        # an integer dtype, but it is the unique point in a rank-zero domain.
        if self.input_rank == 0 and isinstance(point, (list, tuple)):
            coordinates = coordinates.astype(np.intp)
        try:
            result = self._apply_points(coordinates)
        except _PointOutOfBounds as error:
            raise BoundsCheckError(
                f"coordinate {error.value} on input dimension {error.dimension} "
                f"is outside the domain [{error.lower}, {error.upper})"
            ) from None
        return tuple(int(value) for value in result)

    def apply_many(self, points: npt.ArrayLike) -> npt.NDArray[np.intp]:
        """Map a batch of `domain` coordinates to the source coordinates that fill them.

        The vectorized form of `apply`: each row of `points` names a result
        cell, and the corresponding output row names the source-array cell
        its value is read from.

        Parameters
        ----------
        points : numpy.typing.ArrayLike
            Integer coordinates with shape ``batch_shape + (input_rank,)``.

        Returns
        -------
        numpy.typing.NDArray[numpy.intp]
            An owned array with shape ``batch_shape + (output_rank,)``.

        Raises
        ------
        ValueError
            If ``points`` has no trailing coordinate axis or that axis does
            not contain exactly one coordinate per input dimension.
        TypeError
            If the coordinates do not have an integer dtype.
        BoundsCheckError
            If a coordinate lies outside the input domain.
        OverflowError
            If a mapped output coordinate cannot be represented by
            ``np.intp``.

        Examples
        --------
        Three result cells of the `[::2]` transform, located in one call:

        >>> transform = IndexTransform.from_shape((100,))[::2]
        >>> transform.apply_many(np.array([[0], [1], [49]])).tolist()
        [[0], [2], [98]]
        """
        coordinates = np.asarray(points)
        if coordinates.ndim == 0 or coordinates.shape[-1] != self.input_rank:
            raise ValueError(
                "points must have a trailing coordinate axis of size "
                f"{self.input_rank}, got shape {coordinates.shape}"
            )
        try:
            return self._apply_points(coordinates)
        except _PointOutOfBounds as error:
            raise BoundsCheckError(
                f"point at batch position {error.batch_position} has input dimension "
                f"{error.dimension} coordinate {error.value} outside "
                f"[{error.lower}, {error.upper})"
            ) from None

    def _apply_points(self, points: np.ndarray[Any, Any]) -> npt.NDArray[np.intp]:
        """Vectorized implementation shared by ``apply`` and ``apply_many``.

        Out-of-domain coordinates raise the internal `_PointOutOfBounds`
        signal; each public entry point formats it in its own vocabulary —
        `apply` never mentions a batch, `apply_many` names the batch position."""
        if not np.issubdtype(points.dtype, np.integer):
            raise TypeError(f"points must have an integer dtype, got {points.dtype}")

        invalid = np.zeros(points.shape, dtype=np.bool_)
        for dimension, (lower, upper) in enumerate(
            zip(self.domain.inclusive_min, self.domain.exclusive_max, strict=True)
        ):
            invalid[..., dimension] = (points[..., dimension] < lower) | (
                points[..., dimension] >= upper
            )
        invalid_positions = np.argwhere(invalid)
        if invalid_positions.size > 0:
            first = invalid_positions[0]
            dimension = int(first[-1])
            batch_position = tuple(int(position) for position in first[:-1])
            point_index = tuple(int(position) for position in first)
            value = int(points[point_index])
            lower = self.domain.inclusive_min[dimension]
            upper = self.domain.exclusive_max[dimension]
            raise _PointOutOfBounds(dimension, value, lower, upper, batch_position)

        batch_shape = points.shape[:-1]
        result = np.empty(batch_shape + (self.output_rank,), dtype=np.intp)
        for output_dimension, output_map in enumerate(self.output):
            if isinstance(output_map, ConstantMap):
                if result[..., output_dimension].size == 0:
                    continue
                result[..., output_dimension] = checked_affine(output_map.offset, 0, 0)
            elif isinstance(output_map, DimensionMap):
                result[..., output_dimension] = checked_affine(
                    output_map.offset,
                    output_map.stride,
                    points[..., output_map.input_dimension],
                )
            else:
                index = tuple(
                    np.zeros(batch_shape, dtype=np.intp)
                    if output_map.index_array.shape[axis] == 1
                    else _positions_from_origin(points[..., axis], self.domain.inclusive_min[axis])
                    for axis in range(self.input_rank)
                )
                result[..., output_dimension] = checked_affine(
                    output_map.offset,
                    output_map.stride,
                    np.asarray(output_map.index_array[index]),
                )
        return result

    def inverted(self) -> IndexTransform:
        """Return the restricted, exactly representable inverse transform.

        Inversion is defined for square transforms containing only constants
        and unique unit-stride dimension maps. Any input dimension not named by
        a dimension map must have singleton extent, so its coordinate can be
        recovered as a constant.

        Returns
        -------
        IndexTransform
            A new transform mapping output coordinates back to input
            coordinates.

        Raises
        ------
        ValueError
            If this transform does not have a representable inverse, including
            when input labels cannot be transferred to unlabeled output
            dimensions.
        """
        if self.domain.labels is not None:
            raise ValueError(
                "cannot invert transform: input labels cannot be represented "
                "because output dimensions do not carry labels"
            )
        if self.input_rank != self.output_rank:
            raise ValueError(
                "cannot invert transform: input rank must equal output rank, got "
                f"{self.input_rank} and {self.output_rank}"
            )

        referenced: set[int] = set()
        for output_dimension, output_map in enumerate(self.output):
            if isinstance(output_map, ArrayMap):
                raise ValueError(  # noqa: TRY004 - valid map, invalid inverse
                    f"cannot invert transform: output[{output_dimension}] is an ArrayMap"
                )
            if isinstance(output_map, DimensionMap):
                if output_map.stride not in (-1, 1):
                    raise ValueError(
                        "cannot invert transform: DimensionMap stride must be +1 or -1, "
                        f"got {output_map.stride} for output[{output_dimension}]"
                    )
                if output_map.input_dimension in referenced:
                    raise ValueError(
                        "cannot invert transform: input dimension "
                        f"{output_map.input_dimension} is referenced more than once"
                    )
                referenced.add(output_map.input_dimension)

        for input_dimension, extent in enumerate(self.domain.shape):
            if input_dimension not in referenced and extent != 1:
                raise ValueError(
                    "cannot invert transform: unreferenced input dimension "
                    f"{input_dimension} has extent {extent}, not 1"
                )

        inverse_min: list[int] = []
        inverse_max: list[int] = []
        inverse_output: dict[int, OutputIndexMap] = {}
        for output_dimension, output_map in enumerate(self.output):
            if isinstance(output_map, ConstantMap):
                inverse_min.append(output_map.offset)
                inverse_max.append(output_map.offset + 1)
                continue

            assert isinstance(output_map, DimensionMap)
            input_dimension = output_map.input_dimension
            lower = self.domain.inclusive_min[input_dimension]
            upper = self.domain.exclusive_max[input_dimension]
            if output_map.stride == 1:
                inverse_min.append(output_map.offset + lower)
                inverse_max.append(output_map.offset + upper)
                inverse_output[input_dimension] = DimensionMap(
                    output_dimension,
                    offset=-output_map.offset,
                )
            else:
                inverse_min.append(output_map.offset - upper + 1)
                inverse_max.append(output_map.offset - lower + 1)
                inverse_output[input_dimension] = DimensionMap(
                    output_dimension,
                    offset=output_map.offset,
                    stride=-1,
                )

        for input_dimension, lower in enumerate(self.domain.inclusive_min):
            if input_dimension not in referenced:
                inverse_output[input_dimension] = ConstantMap(lower)

        return IndexTransform(
            domain=IndexDomain(tuple(inverse_min), tuple(inverse_max)),
            output=tuple(inverse_output[dimension] for dimension in range(self.input_rank)),
        )

    @property
    def selection_repr(self) -> str:
        """Compact domain string, e.g. `'{ [2, 8), [0, 10) }'`.

        Follows TensorStore's IndexDomain notation: each dimension shown
        as `[inclusive_min, exclusive_max)` with stride annotation if not 1.
        Constant (integer-indexed) dimensions show as a single value.
        Array-indexed dimensions show the set of selected coordinates.
        """
        parts: list[str] = []
        for m in self.output:
            if isinstance(m, ConstantMap):
                parts.append(str(m.offset))
            elif isinstance(m, DimensionMap):
                d = m.input_dimension
                lo = self.domain.inclusive_min[d]
                hi = self.domain.exclusive_max[d]
                start = m.offset + m.stride * lo
                stop = m.offset + m.stride * hi
                if m.stride == 1:
                    parts.append(f"[{start}, {stop})")
                else:
                    parts.append(f"[{start}, {stop}) step {m.stride}")
            else:
                # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
                storage = m.offset + m.stride * m.index_array
                n = int(storage.size)  # .size, not len(): index_array may be 0-d
                if n <= 5:
                    vals = ", ".join(str(int(v)) for v in storage.ravel())
                    parts.append("{" + vals + "}")
                else:
                    parts.append("{" + f"array({n})" + "}")
        return "{ " + ", ".join(parts) + " }"

    def __repr__(self) -> str:
        maps: list[str] = []
        for i, m in enumerate(self.output):
            if isinstance(m, ConstantMap):
                maps.append(f"out[{i}] = {m.offset}")
            elif isinstance(m, DimensionMap):
                maps.append(f"out[{i}] = {m.offset} + {m.stride} * in[{m.input_dimension}]")
            else:
                # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
                maps.append(f"out[{i}] = {m.offset} + {m.stride} * arr{m.index_array.shape}[in]")
        maps_str = ", ".join(maps)
        return f"IndexTransform(domain={self.domain}, {maps_str})"

    def intersect(
        self, output_domain: IndexDomain
    ) -> (
        tuple[
            IndexTransform,
            dict[int, np.ndarray[Any, np.dtype[np.intp]]]
            | np.ndarray[Any, np.dtype[np.intp]]
            | None,
        ]
        | None
    ):
        """Keep only the cells whose source coordinates fall inside `output_domain`.

        Chunk resolution is the canonical caller: intersecting a request with
        one chunk's box keeps the cells that chunk can serve.

        Returns `(restricted_transform, out_indices)` or None if empty.

        `out_indices` carries the surviving output positions: `None` when all
        positions survive (ConstantMap/DimensionMap only), a single integer array
        for one ArrayMap (or correlated/vectorized ArrayMaps), or a dict keyed by
        output dimension for >= 2 orthogonal ArrayMaps (an outer product).
        """
        return _intersect(self, output_domain)

    def translate(self, shift: tuple[int, ...]) -> IndexTransform:
        """Shift the source coordinates every cell reads by `shift`, per dimension.

        The domain is untouched: the result keeps its cells, and each one
        reads from a shifted source address — for example, making a chunk's
        global addresses chunk-local by translating by the chunk's negated
        origin.
        """
        if len(shift) != self.output_rank:
            raise ValueError(f"shift must have length {self.output_rank}, got {len(shift)}")
        new_output: list[OutputIndexMap] = []
        for m, s in zip(self.output, shift, strict=True):
            if isinstance(m, ConstantMap):
                new_output.append(ConstantMap(offset=m.offset + s))
            elif isinstance(m, DimensionMap):
                new_output.append(
                    DimensionMap(
                        input_dimension=m.input_dimension,
                        offset=m.offset + s,
                        stride=m.stride,
                    )
                )
            else:
                # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
                new_output.append(
                    ArrayMap(
                        index_array=m.index_array,
                        offset=m.offset + s,
                        stride=m.stride,
                    )
                )
        return IndexTransform(domain=self.domain, output=tuple(new_output))

    def __getitem__(self, selection: Any) -> IndexTransform:
        """Compose a basic selection (int, slice, ellipsis, newaxis) into a new transform.

        No I/O occurs. Integers and slice bounds are literal domain coordinates
        (TensorStore convention): negative values are not counted from the end,
        and out-of-domain values raise `BoundsCheckError`. Integer indices drop
        their input dimension; `None` inserts a size-1 dimension.
        """
        return _apply_basic_indexing(self, selection)

    def translate_domain_by(self, shift: tuple[int, ...]) -> IndexTransform:
        """Shift the *input* domain by `shift`, preserving which cells are addressed.

        TensorStore's `translate_by`: the domain moves, and every output map is
        re-offset so that new coordinate `c` addresses the cell that `c - shift`
        addressed before. ArrayMaps are indexed positionally over the domain, so
        their index arrays are unchanged.
        """
        if len(shift) != self.input_rank:
            raise ValueError(f"shift must have length {self.input_rank}, got {len(shift)}")
        new_domain = self.domain.translate(shift)
        new_output: list[OutputIndexMap] = []
        for m in self.output:
            if isinstance(m, DimensionMap):
                s = shift[m.input_dimension]
                new_output.append(
                    DimensionMap(
                        input_dimension=m.input_dimension,
                        offset=m.offset - m.stride * s,
                        stride=m.stride,
                    )
                )
            else:
                # ConstantMap: no input dependence. ArrayMap: positional over
                # the domain, invariant under domain translation.
                new_output.append(m)
        return IndexTransform(domain=new_domain, output=tuple(new_output))

    def translate_domain_to(self, origins: tuple[int, ...]) -> IndexTransform:
        """Move the input domain so its per-dimension origins equal `origins`.

        TensorStore's `translate_to`; `translate_domain_to((0,) * rank)`
        re-zeros a view's coordinate system without changing which cells it
        addresses.
        """
        if len(origins) != self.input_rank:
            raise ValueError(f"origins must have length {self.input_rank}, got {len(origins)}")
        shift = tuple(o - m for o, m in zip(origins, self.domain.inclusive_min, strict=True))
        return self.translate_domain_by(shift)

    @property
    def oindex(self) -> _OIndexHelper:
        """Accessor for the orthogonal (outer-product) indexing dialect.

        `transform.oindex[sel]` applies each index array independently per
        dimension and returns a new transform.
        """
        return _OIndexHelper(self)

    @property
    def vindex(self) -> _VIndexHelper:
        """Accessor for the vectorized (coordinate/mask) indexing dialect.

        `transform.vindex[sel]` broadcasts all index arrays together, NumPy
        fancy-indexing style, and returns a new transform.
        """
        return _VIndexHelper(self)

    @property
    def index_array_structure(self) -> Literal["none", "orthogonal", "general"]:
        """Classify how a transform's index arrays relate to its input axes.

        Returns
        -------
        `"none"` when no output map is an `ArrayMap`; `"orthogonal"` when every
        `ArrayMap` varies over exactly one input axis, each its own (an outer
        product, one independent gather per axis); `"general"` otherwise —
        correlated (`vindex`) maps sharing their non-singleton axes, maps produced
        by composing fancy steps, maps sharing an input axis (a diagonal gather),
        and empty or hand-built all-singleton maps whose shape names no axis. The
        orthogonal resolvers narrow one axis at a time and are only sound for
        `"orthogonal"`; everything else takes the pointwise path that collapses
        the joint block. Everything is read off the index arrays' shapes.

        Examples
        --------
        >>> t = IndexTransform.from_shape((4, 5))
        >>> t.index_array_structure
        'none'

        `oindex` arrays each vary over their own axis (an outer product):

        >>> t.oindex[[0, 2], [1, 3]].index_array_structure
        'orthogonal'

        `vindex` arrays are correlated — they share the broadcast axis:

        >>> t.vindex[np.array([0, 2]), np.array([1, 3])].index_array_structure
        'general'
        """
        seen: set[int] = set()
        has_array = False
        for m in self.output:
            if not isinstance(m, ArrayMap):
                continue
            has_array = True
            dep = m.dependency_axes
            if len(dep) != 1 or dep[0] in seen:
                return "general"
            seen.add(dep[0])
        return "orthogonal" if has_array else "none"

    def select(
        self,
        selection: Any,
        mode: Literal["basic", "orthogonal", "vectorized"] = "basic",
    ) -> IndexTransform:
        """Convert a user selection into a composed IndexTransform.

        Negative indices are treated as literal coordinates (TensorStore convention).
        The caller (Array layer) is responsible for converting numpy-style negative
        indices before calling this function.

        Examples
        --------
        The `mode` picks the dialect; the result is the composed self the
        corresponding accessor builds:

        >>> t = IndexTransform.from_shape((10,))
        >>> t.select(slice(2, 8)) == t[2:8]
        True
        >>> s = t.select(([9, 0, 0],), mode="orthogonal")
        >>> s.apply((0,)), s.apply((1,)), s.apply((2,))
        ((9,), (0,), (0,))
        """
        if mode == "basic":
            _validate_basic_selection(selection)
            return self[selection]
        elif mode == "orthogonal":
            _validate_array_selection(selection, self.domain.shape, mode)
            return self.oindex[selection]
        elif mode == "vectorized":
            _validate_array_selection(selection, self.domain.shape, mode)
            return self.vindex[selection]
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def compose(self, inner: IndexTransform) -> IndexTransform:
        """Chain `inner` onto this transform, yielding one direct transform.

        This transform maps its own input coordinates to `inner`'s input
        coordinates, and `inner` maps those onward; the result maps this
        transform's input coordinates straight to `inner`'s output
        coordinates. Composition is what keeps a view of a view a single
        description rather than a stack of layers, and it is exact: index
        arrays are evaluated at the new coordinates rather than accumulated.

        The precondition is that this transform's output rank equals `inner`'s
        input rank; a mismatch, or coordinates leaving `inner`'s domain, raises.

        Examples
        --------
        Chained indexing — `source[2:5]`, then `[::-1]` on the result —
        collapses to one transform (a reversed axis keeps literal coordinates,
        so the composed domain is `[-4, -1)`):

        >>> inner = IndexTransform.from_shape((10,))[2:5]
        >>> outer = IndexTransform.identity(inner.domain)[::-1]
        >>> chained = outer.compose(inner)
        >>> chained == inner[::-1]
        True
        >>> [chained.apply((i,)) for i in (-4, -3, -2)]
        [(4,), (3,), (2,)]
        """
        from zarr_indexing._composition import compose

        return compose(self, inner)

    # -- serialization ------------------------------------------------------

    def to_json(self) -> IndexTransformJSON:
        """Convert to the canonical ndsel transform body (spec section 4.3).

        The result is fully explicit: `input_rank`, fully written bounds and
        labels, and an `output` carrying `offset`/`stride` on every affine and
        array map. It is field-for-field a TensorStore `IndexTransform` minus
        the `kind` discriminator, so it loads directly into
        `tensorstore.IndexTransform(json=...)`.

        Examples
        --------
        >>> body = IndexTransform.from_shape((6,))[1:5:2].to_json()
        >>> (body["input_inclusive_min"], body["input_exclusive_max"])
        ([0], [2])
        >>> body["output"]
        [{'offset': 1, 'stride': 2, 'input_dimension': 0}]
        """
        from zarr_indexing._wire import emit_labels

        return {
            "input_rank": self.domain.ndim,
            "input_inclusive_min": list(self.domain.inclusive_min),
            "input_exclusive_max": list(self.domain.exclusive_max),
            "input_labels": emit_labels(self.domain.labels, self.domain.ndim),
            "output": [m.to_json() for m in self.output],
        }

    @classmethod
    def from_json(cls, data: IndexTransformJSON) -> IndexTransform:
        """Construct from a canonical (or canonicalizable) ndsel transform body.

        The body is first run through the message layer (`normalize_ndsel`) so
        that omitted fields — identity `output`, default bounds and labels —
        are filled and validated, then lowered to the engine representation.
        Lower-rank `index_array`s are widened to the full input rank on the way
        in.

        Examples
        --------
        >>> body = IndexTransform.from_shape((6,))[1:5:2].to_json()
        >>> transform = IndexTransform.from_json(body)
        >>> transform.domain.shape
        (2,)
        >>> transform.to_json() == body  # the round trip is exact
        True
        """
        from zarr_indexing._wire import (
            full_rank_index_array,
            lower_bound,
            lower_index_array,
            lower_labels,
        )
        from zarr_indexing.messages import NdselError, normalize_ndsel

        if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise NdselError(
                "invalid_json", f"a transform body must be a JSON object, got {data!r}"
            )
        kind = data.get("kind", "transform")
        if kind != "transform":
            # Spelled before normalization so a body carrying its own `kind`
            # cannot reinterpret the document as some other message and return
            # a selection this constructor never promised.
            raise NdselError("invalid_json", f"a transform body cannot carry kind {kind!r}")
        body = normalize_ndsel({**data, "kind": "transform"})

        domain = IndexDomain(
            inclusive_min=tuple(
                lower_bound(b, f"input_inclusive_min[{i}]")
                for i, b in enumerate(body["input_inclusive_min"])
            ),
            exclusive_max=tuple(
                lower_bound(b, f"input_exclusive_max[{i}]")
                for i, b in enumerate(body["input_exclusive_max"])
            ),
            labels=lower_labels(body["input_labels"]),
        )

        output: list[OutputIndexMap] = []
        for i, om in enumerate(body["output"]):
            if "index_array" in om:
                where = f"output[{i}]"
                arr = lower_index_array(om["index_array"], f"{where}.index_array")
                # ndsel leaves index-array rank unvalidated, so an external
                # producer may send an array of lower rank that broadcasts
                # against the domain. Widen it here, on the way in, so every
                # transform that exists holds the full-rank invariant the
                # engine reads dependency axes from.
                output.append(
                    ArrayMap(
                        index_array=full_rank_index_array(arr, domain, where),
                        offset=om.get("offset", 0),
                        stride=om.get("stride", 1),
                    )
                )
            elif "input_dimension" in om:
                output.append(
                    DimensionMap(
                        input_dimension=om["input_dimension"],
                        offset=om.get("offset", 0),
                        stride=om.get("stride", 1),
                    )
                )
            else:
                output.append(ConstantMap(offset=om.get("offset", 0)))

        try:
            return cls(domain=domain, output=tuple(output))
        except ValueError as exc:
            # The engine's invariants are the last gate a document passes, and
            # they speak in the engine's vocabulary. A document that fails them
            # is invalid input, so it leaves here as one — with the engine's
            # account of what was wrong kept, since it names the offending
            # output map and axis.
            raise NdselError("rank_mismatch", str(exc)) from exc


def _positions_from_origin(coordinates: np.ndarray[Any, Any], origin: int) -> npt.NDArray[np.intp]:
    """Convert literal coordinates to positional indices without wrapping."""
    return checked_affine(-int(origin), 1, coordinates)


def _intersect(
    transform: IndexTransform, output_domain: IndexDomain
) -> (
    tuple[
        IndexTransform,
        dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None,
    ]
    | None
):
    """Intersect a transform with an output domain (e.g., a chunk's bounds).

    For each output dimension, restrict to output coordinates within
    `[output_domain.inclusive_min[d], output_domain.exclusive_max[d])`.

    Two flavors of fancy indexing require different treatment, distinguished by
    the ArrayMaps' dependency axes (see `ArrayMap.dependency_axes`):

    - **orthogonal** (`oindex`): each ArrayMap varies over a single, distinct
      input axis, forming an outer product. Every output dimension is intersected
      independently and the input domain narrowed per axis.
    - **correlated** (`vindex`): the ArrayMaps share their (broadcast) dependency
      axes and scatter through a single flat index. A point survives only if ALL
      its output coordinates fall within the output domain; residual slice
      dimensions are intersected independently, as in the orthogonal case.

    The routing is `index_array_structure`: only a pure per-axis outer product
    takes the orthogonal path.

    Returns `None` if the intersection is empty.
    """
    if output_domain.ndim != transform.output_rank:
        raise ValueError(
            f"output_domain rank ({output_domain.ndim}) != "
            f"transform output rank ({transform.output_rank})"
        )

    if any(size == 0 for size in transform.domain.shape):
        # An empty input domain addresses no coordinates at all, so it meets no
        # output domain. Deciding it here keeps the per-flavor intersections from
        # having to reconcile an empty domain with an index array that is *not*
        # empty: a genuine extent-1 axis is stored as a broadcast singleton, so
        # emptying its domain leaves the array at size 1.
        return None

    if transform.index_array_structure == "general":
        return _intersect_general(transform, output_domain)
    return _intersect_orthogonal(transform, output_domain)


def _intersect_dimension_map(
    m: DimensionMap, input_lo: int, input_hi: int, lo: int, hi: int
) -> tuple[int, int] | None:
    """Narrow a DimensionMap's input range to output coordinates in `[lo, hi)`.

    `input_lo`/`input_hi` are the current (possibly already narrowed) input
    range for the map's axis. Returns the new `(input_lo, input_hi)` or `None`
    if no input produces an in-bounds output coordinate.
    """
    if input_lo >= input_hi:
        return None
    if m.stride > 0:
        new_input_lo = max(input_lo, _ceil_div(lo - m.offset, m.stride))
        new_input_hi = min(input_hi, _ceil_div(hi - m.offset, m.stride))
    elif m.stride < 0:
        new_input_lo = max(input_lo, _ceil_div(hi - 1 - m.offset, m.stride))
        new_input_hi = min(input_hi, _ceil_div(lo - 1 - m.offset, m.stride))
    else:
        if lo <= m.offset < hi:
            new_input_lo, new_input_hi = input_lo, input_hi
        else:
            return None
    if new_input_lo >= new_input_hi:
        return None
    return new_input_lo, new_input_hi


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return ``ceil(numerator / denominator)`` using exact integer arithmetic."""
    return -((-numerator) // denominator)


def _intersect_orthogonal(
    transform: IndexTransform, output_domain: IndexDomain
) -> (
    tuple[
        IndexTransform,
        dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None,
    ]
    | None
):
    """Intersect a transform with no correlated ArrayMaps.

    Every output dimension is intersected independently. Multiple ArrayMaps bound
    to distinct input dimensions form an outer product, so each array's surviving
    *output* positions are tracked separately.
    """
    new_min = list(transform.domain.inclusive_min)
    new_max = list(transform.domain.exclusive_max)
    new_output: list[OutputIndexMap] = []
    out_positions: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}

    for out_dim, m in enumerate(transform.output):
        lo = output_domain.inclusive_min[out_dim]
        hi = output_domain.exclusive_max[out_dim]

        if isinstance(m, ConstantMap):
            if lo <= m.offset < hi:
                new_output.append(m)
            else:
                return None

        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            narrowed = _intersect_dimension_map(m, new_min[d], new_max[d], lo, hi)
            if narrowed is None:
                return None
            new_min[d], new_max[d] = narrowed
            new_output.append(m)

        else:
            # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
            # Orthogonal: the array varies over a single axis. Filter along that
            # axis and keep the array at full input rank so the singleton axes
            # it broadcasts over are preserved.
            axis = m.dependent_axis
            if axis is None:
                raise ValueError(
                    f"output[{out_dim}] is an ArrayMap that varies over no input "
                    "dimension; a map with no dependency axis should have been "
                    "collapsed to a ConstantMap"
                )
            d = axis
            storage = checked_affine(m.offset, m.stride, m.index_array)
            mask = (storage >= lo) & (storage < hi)
            # The array is singleton on every axis but `d`, so its mask reduces
            # to a 1-D vector along `d`.
            survivors = np.nonzero(mask.reshape(-1))[0].astype(np.intp)
            if survivors.size == 0:
                return None
            filtered = np.take(m.index_array, survivors, axis=d)
            new_output.append(
                ArrayMap(
                    index_array=np.asarray(filtered, dtype=np.intp),
                    offset=m.offset,
                    stride=m.stride,
                )
            )
            new_max[d] = new_min[d] + int(survivors.size)
            out_positions[out_dim] = survivors

    new_domain = IndexDomain(
        inclusive_min=tuple(new_min),
        exclusive_max=tuple(new_max),
    )
    result = IndexTransform(domain=new_domain, output=tuple(new_output))

    # Hand back the surviving output positions in the shape the bridge expects:
    # None (no arrays), a single vector (one array), or a per-output-dim dict
    # (>= 2 orthogonal arrays → outer product).
    out_indices: (
        dict[int, np.ndarray[Any, np.dtype[np.intp]]] | np.ndarray[Any, np.dtype[np.intp]] | None
    )
    if len(out_positions) == 0:
        out_indices = None
    elif len(out_positions) == 1:
        out_indices = next(iter(out_positions.values()))
    else:
        out_indices = out_positions
    return (result, out_indices)


def _intersect_general(
    transform: IndexTransform,
    output_domain: IndexDomain,
) -> tuple[IndexTransform, np.ndarray[Any, np.dtype[np.intp]]] | None:
    """Intersect a transform with any index-array structure, pointwise.

    Every `ArrayMap` — correlated, orthogonal, or several sharing an axis — is
    treated as a lookup table over the joint block of non-slice axes: a block
    point survives only if ALL its output coordinates fall within the output
    domain. Residual DimensionMap dimensions are intersected independently (as in
    the orthogonal case) and preserved, so a partial vindex — e.g. two coordinate
    arrays over a 3-D array, leaving one slice dimension — resolves correctly.
    Treating an orthogonal map this way forfeits its per-axis independence (the
    block enumerates the outer product), which is why the pure-orthogonal case
    keeps its own resolver.

    The surviving broadcast axes collapse to a single axis; the returned
    `out_indices` is the flat scatter index into the (row-major flattened)
    output buffer, of shape `(surviving_points,) + (residual slice sizes)`.

    A rank-0 broadcast block — every coordinate array a scalar, as after
    `vindex[...]` narrowed to a single point — has no axis to collapse and stays
    rank 0: the block either survives whole or the intersection is empty. The
    result keeps only the residual slice axes and `out_indices` loses its leading
    points axis, so the sub-transform's rank still matches the view's.
    """
    correlated_dims = [i for i, m in enumerate(transform.output) if isinstance(m, ArrayMap)]

    # The broadcast axes are exactly the input axes no `DimensionMap` binds: a
    # correlated transform's input domain is its residual slice axes plus the
    # collapsed broadcast block. Deriving them by complement rather than from the
    # index array's non-singleton axes keeps this correct when a broadcast axis
    # is itself size 1, and when NumPy's placement rule puts the broadcast block
    # somewhere other than the front (see `_broadcast_insertion_point`).
    bound_axes = {m.input_dimension for m in transform.output if isinstance(m, DimensionMap)}
    broadcast_axes = tuple(a for a in range(transform.input_rank) if a not in bound_axes)
    broadcast_shape = tuple(transform.domain.shape[a] for a in broadcast_axes)

    for out_dim in correlated_dims:
        arr_map = cast("ArrayMap", transform.output[out_dim])
        if any(a not in broadcast_axes for a in arr_map.dependency_axes):
            # Reachable only by hand-building a transform: no selection binds
            # the same input axis to both a slice map and an index array.
            raise NotImplementedError(
                "intersecting a transform whose index array varies over an "
                "input dimension also bound by a slice map is not supported"
            )

    # Joint bounds mask over the broadcast block.
    combined: np.ndarray[Any, np.dtype[np.bool_]] | None = None
    for out_dim in correlated_dims:
        cm = cast("ArrayMap", transform.output[out_dim])
        storage = checked_affine(cm.offset, cm.stride, cm.index_array)
        lo = output_domain.inclusive_min[out_dim]
        hi = output_domain.exclusive_max[out_dim]
        mask = (storage >= lo) & (storage < hi)
        combined = mask if combined is None else (combined & mask)
    assert combined is not None
    # Index arrays are singleton on every non-broadcast axis, so the mask
    # collapses (C-order) to the broadcast block. A map may also be singleton
    # along a block axis it does not vary over (an orthogonal member, or a
    # leftover broadcast axis), so the collapsed mask is broadcast up to the
    # full block rather than reshaped.
    combined_block = combined.reshape(tuple(combined.shape[a] for a in broadcast_axes))
    combined_bcast = np.broadcast_to(combined_block, broadcast_shape)
    surviving = np.nonzero(combined_bcast.reshape(-1))[0].astype(np.intp)
    if surviving.size == 0:
        return None

    # Intersect residual (slice / constant) dimensions independently. Slice dims
    # are ordered by input dimension so their flat-buffer strides are row-major.
    slice_dims: list[tuple[int, int, int, int, DimensionMap]] = []  # (in_dim, lo, hi, full, m)
    for out_dim, m in enumerate(transform.output):
        if out_dim in correlated_dims:
            continue
        lo = output_domain.inclusive_min[out_dim]
        hi = output_domain.exclusive_max[out_dim]
        if isinstance(m, ConstantMap):
            if not (lo <= m.offset < hi):
                return None
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            input_lo = transform.domain.inclusive_min[d]
            input_hi = transform.domain.exclusive_max[d]
            narrowed = _intersect_dimension_map(m, input_lo, input_hi, lo, hi)
            if narrowed is None:
                return None
            slice_dims.append((d, narrowed[0], narrowed[1], input_hi - input_lo, m))
    slice_dims.sort(key=lambda item: item[0])

    n_points = int(surviving.size)
    n_slice = len(slice_dims)
    corr_values: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    for out_dim in correlated_dims:
        arr = cast("ArrayMap", transform.output[out_dim]).index_array
        block = arr.reshape(tuple(arr.shape[a] for a in broadcast_axes))
        corr_values[out_dim] = np.asarray(
            np.ascontiguousarray(np.broadcast_to(block, broadcast_shape)).reshape(-1)[surviving],
            dtype=np.intp,
        )

    # A rank-0 broadcast block contributes no axis: the leading `(n_points,)` of
    # the domain, of every index array, and of `out_indices` is present only when
    # there was a block to collapse.
    points_shape = (n_points,) if len(broadcast_shape) > 0 else ()

    # New domain: the collapsed broadcast axis if there is one, then one axis per
    # residual slice.
    new_min = [0] * len(points_shape)
    new_max = list(points_shape)
    new_input_dim_of = {}
    for new_axis, (d, nlo, nhi, _full, _m) in enumerate(slice_dims, start=len(points_shape)):
        new_min.append(nlo)
        new_max.append(nhi)
        new_input_dim_of[d] = new_axis
    new_domain = IndexDomain(inclusive_min=tuple(new_min), exclusive_max=tuple(new_max))

    corr_shape = points_shape + (1,) * n_slice
    new_output: list[OutputIndexMap] = []
    for out_dim, m in enumerate(transform.output):
        if out_dim in correlated_dims:
            corr = cast("ArrayMap", m)
            new_output.append(
                ArrayMap(
                    index_array=corr_values[out_dim].reshape(corr_shape).astype(np.intp),
                    offset=corr.offset,
                    stride=corr.stride,
                )
            )
        elif isinstance(m, ConstantMap):
            new_output.append(m)
        else:
            assert isinstance(m, DimensionMap)
            new_output.append(
                DimensionMap(
                    input_dimension=new_input_dim_of[m.input_dimension],
                    offset=m.offset,
                    stride=m.stride,
                )
            )
    result = IndexTransform(domain=new_domain, output=tuple(new_output))

    # Flat scatter index into the caller's row-major output buffer, whose shape
    # is the *input* domain's shape. The buffer is addressed positionally, so
    # this assumes a zero-origin domain — the resolvers normalize with
    # `translate_domain_to` before resolving.
    #
    # Each surviving point is a flat index into the broadcast block; unravel it
    # to per-axis coordinates so the buffer stride of each broadcast axis is
    # applied at its real position, wherever NumPy's placement rule put it.
    domain_shape = transform.domain.shape
    buffer_strides = [1] * len(domain_shape)
    for axis in range(len(domain_shape) - 2, -1, -1):
        buffer_strides[axis] = buffer_strides[axis + 1] * domain_shape[axis + 1]

    point_offsets = np.zeros(n_points, dtype=np.intp)
    if len(broadcast_shape) > 0:
        for axis, coords_along_axis in zip(
            broadcast_axes, np.unravel_index(surviving, broadcast_shape), strict=True
        ):
            point_offsets = point_offsets + coords_along_axis.astype(np.intp) * buffer_strides[axis]

    n_lead = len(points_shape)
    out_indices: np.ndarray[Any, np.dtype[np.intp]] = point_offsets.reshape(
        points_shape + (1,) * n_slice
    )
    for j in range(n_slice):
        d, nlo, nhi, _full, _m = slice_dims[j]
        coords = np.arange(nlo, nhi, dtype=np.intp) * buffer_strides[d]
        shape = [1] * (n_lead + n_slice)
        shape[n_lead + j] = coords.size
        out_indices = out_indices + coords.reshape(shape)
    return (result, out_indices.astype(np.intp))


def _normalize_basic_selection(selection: Any, ndim: int) -> tuple[int | slice | None, ...]:
    """Normalize a selection to a tuple of int, slice, or None (newaxis),
    expanding ellipsis and padding with slice(None) as needed.
    """
    if not isinstance(selection, tuple):
        selection = (selection,)

    # Count non-newaxis, non-ellipsis entries to determine how many real dims are addressed
    n_newaxis = sum(1 for s in selection if s is None)
    has_ellipsis = any(s is Ellipsis for s in selection)
    n_real = len(selection) - n_newaxis - (1 if has_ellipsis else 0)

    if n_real > ndim:
        raise IndexError(
            f"too many indices for array: array has {ndim} dimensions, but {n_real} were indexed"
        )

    result: list[int | slice | None] = []
    ellipsis_seen = False
    for sel in selection:
        if sel is Ellipsis:
            if ellipsis_seen:
                raise IndexError("an index can only have a single ellipsis ('...')")
            ellipsis_seen = True
            num_missing = ndim - n_real
            result.extend([slice(None)] * num_missing)
        elif (scalar := as_scalar_index(sel)) is not None:
            result.append(scalar)
        elif isinstance(sel, slice) or sel is None:
            result.append(sel)
        else:
            raise IndexError(f"unsupported selection type for basic indexing: {type(sel)!r}")

    # Pad remaining dimensions with slice(None)
    while sum(1 for s in result if s is not None) < ndim:
        result.append(slice(None))

    return tuple(result)


def _positional_slice(pos: int, size: int, step: int) -> slice:
    """A NumPy slice selecting `size` elements from `pos`, walking by `step`.

    The stop is `pos + size*step`, except in two cases. An empty selection is
    written out explicitly, because the arithmetic form can be a negative stop
    that NumPy would read as counting from the end. And a downward walk
    reaching the start of the array must stop at `None`, for the same reason:
    `slice(6, -1, -1)` selects nothing where `slice(6, None, -1)` selects the
    first seven elements in reverse.
    """
    if size <= 0:
        return slice(0, 0, 1)
    stop = pos + size * step
    if step < 0 and stop < 0:
        return slice(pos, None, step)
    return slice(pos, stop, step)


def _reindex_array(
    m: ArrayMap,
    normalized: tuple[int | slice | None, ...],
    domain: IndexDomain,
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Apply basic indexing operations to an ArrayMap's index_array.

    The array's axes correspond to the transform's input dimensions (0-indexed
    over the domain shape). Each axis is either a **dependency axis** — the array
    varies with that input dimension — or a **singleton** axis it
    broadcasts over. Integer indexing, slicing, or newaxis is applied to the
    array only along its dependency axes; a selection on a singleton axis does not
    touch the array's values (it just narrows or drops that broadcast axis).
    """
    dependent = set(m.dependency_axes)
    arr = m.index_array

    # Build a numpy indexing tuple: one entry per old input dimension
    idx: list[Any] = []
    old_dim = 0
    newaxis_positions: list[int] = []
    result_axis = 0

    for sel in normalized:
        if sel is None:
            newaxis_positions.append(result_axis)
            result_axis += 1
        elif isinstance(sel, int):
            if old_dim < arr.ndim:
                if old_dim in dependent:
                    # Convert absolute domain coordinate to 0-based array index
                    idx.append(sel - domain.inclusive_min[old_dim])
                else:
                    # Broadcast axis: keep the single element and drop the axis.
                    idx.append(0)
            old_dim += 1
        else:
            # sel: slice (normalized: tuple[int | slice | None, ...])
            if old_dim < arr.ndim:
                if old_dim in dependent:
                    lo = domain.inclusive_min[old_dim]
                    hi = domain.exclusive_max[old_dim]
                    # Bounds are literal domain coordinates; the stored array is
                    # indexed positionally, so shift by the domain origin.
                    start, step, _origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
                    pos = start - lo
                    idx.append(_positional_slice(pos, size, step))
                else:
                    # Broadcast axis: preserve the singleton (it still broadcasts
                    # over the narrowed domain), regardless of the slice bounds.
                    idx.append(slice(None))
            old_dim += 1
            result_axis += 1

    result = arr[tuple(idx)] if idx else arr

    for pos in newaxis_positions:
        result = np.expand_dims(result, axis=pos)

    return np.asarray(result, dtype=np.intp)


def _compose_selection(
    transform: IndexTransform,
    selection: Any,
    mode: Literal["orthogonal", "vectorized"],
) -> IndexTransform:
    """Apply an advanced selection to an array-carrying transform by composition.

    The selection is applied to an identity transform over the current domain —
    the same code path a fresh transform takes, so the dialect (placement,
    bounds, domains) is identical by construction — and the result is chained
    onto `transform` with `compose`, which evaluates the existing index arrays
    at the new coordinates. This is how a second fancy step lands on *any* axis
    of an already-fancy view: axes an existing array varies over, axes it merely
    broadcasts along, or a mixture.
    """
    # Deferred import: `composition` imports this module at import time.

    identity = IndexTransform.identity(transform.domain)
    if mode == "orthogonal":
        outer = _apply_oindex(identity, selection)
    else:
        outer = _apply_vindex(identity, selection)
    return outer.compose(transform)


def _apply_basic_indexing(transform: IndexTransform, selection: Any) -> IndexTransform:
    """Apply basic indexing (int, slice, ellipsis, newaxis) to an IndexTransform."""
    normalized = _normalize_basic_selection(selection, transform.domain.ndim)

    new_inclusive_min: list[int] = []
    new_exclusive_max: list[int] = []
    old_dim = 0
    new_dim_idx = 0
    old_to_new_dim: dict[int, int] = {}
    dropped_dims: set[int] = set()

    # Per old-dim: the slice parameters (for computing new output maps)
    dim_slice_params: dict[int, tuple[int, int, int]] = {}  # old_dim -> (start, stop, step)
    dim_int_val: dict[int, int] = {}  # old_dim -> integer index value

    for sel in normalized:
        if sel is None:
            # newaxis: add a size-1 dimension
            new_inclusive_min.append(0)
            new_exclusive_max.append(1)
            new_dim_idx += 1
        elif isinstance(sel, int):
            # Integer index: drop this input dimension.
            # Negative indices are literal coordinates (TensorStore convention),
            # NOT "from the end" like NumPy. The Array layer handles conversion.
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            idx = sel
            if idx < lo or idx >= hi:
                hint = _LITERAL_HINT if sel < 0 else ""
                raise BoundsCheckError(
                    f"index {sel} is out of bounds for dimension {old_dim} "
                    f"(valid indices [{lo}, {hi})){hint}"
                )
            dropped_dims.add(old_dim)
            dim_int_val[old_dim] = idx
            old_dim += 1
        else:
            # sel: slice (normalized: tuple[int | slice | None, ...])
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]

            # TensorStore semantics: bounds are literal coordinates; a step-1
            # slice keeps them as the new domain, a strided slice's domain is
            # [trunc(start/step), trunc(start/step) + size).
            start, step, origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
            new_inclusive_min.append(origin)
            new_exclusive_max.append(origin + size)
            dim_slice_params[old_dim] = (start, step, origin)
            old_to_new_dim[old_dim] = new_dim_idx
            new_dim_idx += 1
            old_dim += 1

    new_domain = IndexDomain(
        inclusive_min=tuple(new_inclusive_min),
        exclusive_max=tuple(new_exclusive_max),
    )

    # Now update output maps
    new_output: list[OutputIndexMap] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            new_output.append(m)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            if d in dropped_dims:
                # Integer index: this output becomes constant
                new_offset = m.offset + m.stride * dim_int_val[d]
                new_output.append(ConstantMap(offset=new_offset))
            elif d in old_to_new_dim:
                # Slice: new coordinate `origin + k` maps to old coordinate
                # `start + k*step`, i.e. old = start - step*origin + step*new.
                start, step, origin = dim_slice_params[d]
                new_offset = m.offset + m.stride * (start - step * origin)
                new_stride = m.stride * step
                new_input_dim = old_to_new_dim[d]
                new_output.append(
                    DimensionMap(
                        input_dimension=new_input_dim, offset=new_offset, stride=new_stride
                    )
                )
            else:
                raise RuntimeError(f"unexpected: dimension {d} not handled")
        else:
            # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap).
            # A result narrowed to a single coordinate collapses to the
            # ConstantMap it equals — whether an integer consumed the dependency
            # axis or a slice narrowed it to one entry — so a non-empty ArrayMap
            # always varies over at least one axis. Nothing here renumbers: the
            # array's axes are the new domain's axes by construction.
            new_arr = _reindex_array(m, normalized, transform.domain)
            new_output.append(array_map_or_constant(new_arr, offset=m.offset, stride=m.stride))

    return IndexTransform(domain=new_domain, output=tuple(new_output))


def _reshape_to_axis(
    values: np.ndarray[Any, np.dtype[np.intp]], axis: int, ndim: int
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Reshape a 1-D selection to full rank `ndim` varying only along `axis`.

    The result has `values` laid out along `axis` and singleton (size-1) axes
    everywhere else, so its dependency axis is derivable from its shape.
    """
    flat = np.asarray(values, dtype=np.intp).ravel()
    shape = [1] * ndim
    shape[axis] = flat.shape[0]
    return flat.reshape(shape)


class _OIndexHelper:
    """Helper that provides orthogonal (outer) indexing via `transform.oindex[...]`."""

    def __init__(self, transform: IndexTransform) -> None:
        self._transform = transform

    def __getitem__(self, selection: Any) -> IndexTransform:
        return _apply_oindex(self._transform, selection)


def _normalize_oindex_selection(
    selection: Any, ndim: int
) -> tuple[np.ndarray[Any, np.dtype[np.intp]] | slice, ...]:
    """Normalize an oindex selection: arrays, slices, booleans, integers."""
    if not isinstance(selection, tuple):
        selection = (selection,)

    # Expand ellipsis
    has_ellipsis = any(s is Ellipsis for s in selection)
    n_ellipsis = 1 if has_ellipsis else 0
    n_real = len(selection) - n_ellipsis

    result: list[np.ndarray[Any, np.dtype[np.intp]] | slice] = []
    for sel in selection:
        if sel is Ellipsis:
            num_missing = ndim - n_real
            result.extend([slice(None)] * num_missing)
        elif isinstance(sel, np.ndarray) and sel.dtype == np.bool_:
            # Boolean array -> integer indices
            (indices,) = np.nonzero(sel)
            result.append(indices.astype(np.intp))
        elif isinstance(sel, np.ndarray):
            result.append(sel.astype(np.intp))
        elif isinstance(sel, slice):
            result.append(sel)
        elif (scalar := as_scalar_index(sel)) is not None:
            # Convert integer scalars to 1-element arrays for orthogonal indexing
            result.append(np.array([scalar], dtype=np.intp))
        elif isinstance(sel, (list, tuple)):
            array = np.asarray(sel)
            if array.dtype == np.bool_:
                (indices,) = np.nonzero(array)
                result.append(indices.astype(np.intp))
            else:
                result.append(np.asarray(sel, dtype=np.intp))
        else:
            result.append(sel)

    # Pad with slice(None)
    while len(result) < ndim:
        result.append(slice(None))

    return tuple(result)


def _apply_oindex(transform: IndexTransform, selection: Any) -> IndexTransform:
    """Apply orthogonal indexing to an IndexTransform.

    Each index array is applied independently per dimension (outer product).

    A transform that already carries index arrays takes the composition path
    (`_compose_selection`) instead of being rewritten in place, so the new
    selection may land on any axis — including axes an existing array merely
    broadcasts along.
    """
    validate_advanced_selection(selection, transform.domain, "orthogonal")
    if any(isinstance(m, ArrayMap) for m in transform.output):
        return _compose_selection(transform, selection, "orthogonal")
    normalized = _normalize_oindex_selection(selection, transform.domain.ndim)

    new_inclusive_min: list[int] = []
    new_exclusive_max: list[int] = []
    new_dim_idx = 0
    old_to_new_dim: dict[int, int] = {}

    # Info per old dim
    dim_array: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    dim_slice_params: dict[int, tuple[int, int, int]] = {}

    for old_dim, sel in enumerate(normalized):
        if isinstance(sel, np.ndarray):
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            # Index-array values are literal domain coordinates; the fancy dim
            # they create gets a fresh zero-origin [0, n) domain (TensorStore).
            _check_array_in_bounds(sel, lo, hi)
            dim_array[old_dim] = sel
            new_inclusive_min.append(0)
            new_exclusive_max.append(len(sel))
            old_to_new_dim[old_dim] = new_dim_idx
            new_dim_idx += 1
        else:
            # sel: slice (_normalize_oindex_selection returns
            # tuple[np.ndarray | slice, ...])
            lo = transform.domain.inclusive_min[old_dim]
            hi = transform.domain.exclusive_max[old_dim]
            start, step, origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
            new_inclusive_min.append(origin)
            new_exclusive_max.append(origin + size)
            dim_slice_params[old_dim] = (start, step, origin)
            old_to_new_dim[old_dim] = new_dim_idx
            new_dim_idx += 1

    new_domain = IndexDomain(
        inclusive_min=tuple(new_inclusive_min),
        exclusive_max=tuple(new_exclusive_max),
    )

    new_output: list[OutputIndexMap] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            new_output.append(m)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            if d in dim_array:
                new_axis = old_to_new_dim[d]
                # Normalize to full input rank: the selection varies along its
                # own new axis and is singleton on every other axis, so the
                # dependency axis is readable from the shape. A single-entry
                # selection holds one coordinate and collapses to the
                # ConstantMap it equals; its length-1 axis stays in the domain.
                full_arr = _reshape_to_axis(dim_array[d], new_axis, new_dim_idx)
                new_output.append(array_map_or_constant(full_arr, offset=m.offset, stride=m.stride))
            elif d in dim_slice_params:
                start, step, origin = dim_slice_params[d]
                new_offset = m.offset + m.stride * (start - step * origin)
                new_stride = m.stride * step
                new_input_dim = old_to_new_dim[d]
                new_output.append(
                    DimensionMap(
                        input_dimension=new_input_dim, offset=new_offset, stride=new_stride
                    )
                )
            else:
                raise RuntimeError(f"unexpected: dimension {d} not handled")
        else:
            # m: ArrayMap — unreachable: array-carrying transforms took the
            # composition path at the top of this function.
            raise AssertionError(  # noqa: TRY004 - unreachable, not a dispatch
                "unreachable: ArrayMap transforms are composed"
            )

    return IndexTransform(domain=new_domain, output=tuple(new_output))


class _VIndexHelper:
    """Helper that provides vectorized (fancy) indexing via `transform.vindex[...]`."""

    def __init__(self, transform: IndexTransform) -> None:
        self._transform = transform

    def __getitem__(self, selection: Any) -> IndexTransform:
        return _apply_vindex(self._transform, selection)


def _broadcast_insertion_point(array_dims: Sequence[int], slice_dims: Sequence[int]) -> int:
    """Where the broadcast dimensions land, as a count of leading slice dimensions.

    NumPy's advanced-indexing placement rule: when the advanced indices are all
    next to each other in the index tuple, the broadcast dimensions are inserted
    at the spot they occupied; when a slice separates them, they lead. So
    `a[:, i, j]` has shape `(len(a), *broadcast)` while `a[i, :, j]` has shape
    `(*broadcast, a.shape[1])`.

    Returns the number of slice dimensions that precede the broadcast block; `0`
    means the broadcast dimensions lead.
    """
    if len(array_dims) == 0:
        return 0
    first, last = array_dims[0], array_dims[-1]
    separated = any(first < d < last for d in slice_dims)
    if separated:
        return 0
    return sum(1 for d in slice_dims if d < first)


def _as_boolean_index_array(selection: Any) -> np.ndarray[Any, np.dtype[np.bool_]] | None:
    """Return an array-like boolean index as an ndarray, else None."""
    if not isinstance(selection, (np.ndarray, list, tuple)):
        return None
    array = np.asarray(selection)
    if array.dtype != np.bool_:
        return None
    return array


def _selection_axis_count(selection: Any) -> int:
    """Return how many input axes one vectorized selection entry consumes."""
    boolean_array = _as_boolean_index_array(selection)
    return boolean_array.ndim if boolean_array is not None else 1


def _apply_vindex(transform: IndexTransform, selection: Any) -> IndexTransform:
    """Apply vectorized indexing to an IndexTransform.

    All array indices are broadcast together. Broadcast dimensions are prepended,
    followed by non-array (slice) dimensions.

    A transform that already carries index arrays takes the composition path
    (`_compose_selection`) instead of being rewritten in place; see
    `_apply_oindex`.
    """
    validate_advanced_selection(selection, transform.domain, "vectorized")
    if any(isinstance(m, ArrayMap) for m in transform.output):
        return _compose_selection(transform, selection, "vectorized")
    if not isinstance(selection, tuple):
        selection = (selection,)

    # Expand ellipsis and count consumed dimensions. Boolean masks consume one
    # input axis per mask dimension, whether spelled as an ndarray or a list.
    n_consumed = sum(_selection_axis_count(s) for s in selection if s is not Ellipsis)
    ndim = transform.domain.ndim

    expanded: list[Any] = []
    for sel in selection:
        if sel is Ellipsis:
            num_missing = ndim - n_consumed
            expanded.extend([slice(None)] * num_missing)
        else:
            expanded.append(sel)
    # Count dimensions already consumed by expanded entries
    n_expanded_dims = sum(_selection_axis_count(sel) for sel in expanded)
    while n_expanded_dims < ndim:
        expanded.append(slice(None))
        n_expanded_dims += 1

    # Convert booleans, lists, ints to integer arrays
    processed: list[np.ndarray[Any, np.dtype[np.intp]] | slice] = []
    for sel in expanded:
        boolean_array = _as_boolean_index_array(sel)
        if boolean_array is not None:
            indices_tuple = np.nonzero(boolean_array)
            processed.extend(indices.astype(np.intp) for indices in indices_tuple)
        elif isinstance(sel, np.ndarray):
            processed.append(sel.astype(np.intp))
        elif isinstance(sel, (list, tuple)):
            processed.append(np.asarray(sel, dtype=np.intp))
        elif (scalar := as_scalar_index(sel)) is not None:
            processed.append(np.array([scalar], dtype=np.intp))
        else:
            processed.append(sel)

    # Separate array dims and slice dims
    array_dims: list[int] = []
    slice_dims: list[int] = []
    arrays: list[np.ndarray[Any, np.dtype[np.intp]]] = []

    for i, sel in enumerate(processed):
        if isinstance(sel, np.ndarray):
            lo = transform.domain.inclusive_min[i]
            hi = transform.domain.exclusive_max[i]
            _check_array_in_bounds(sel, lo, hi)
            array_dims.append(i)
            arrays.append(sel)
        else:
            slice_dims.append(i)

    # Broadcast all arrays together
    broadcast_arrays: list[np.ndarray[Any, np.dtype[np.intp]]]
    if len(arrays) > 0:
        broadcast_arrays = list(np.broadcast_arrays(*arrays))
        broadcast_shape = broadcast_arrays[0].shape
    else:
        broadcast_arrays = []
        broadcast_shape = ()

    # Slice dimensions (preserved-domain literal semantics, like basic indexing)
    slice_dim_params: dict[int, tuple[int, int, int]] = {}
    slice_bounds: list[tuple[int, int]] = []
    for old_dim in slice_dims:
        sel = processed[old_dim]
        assert isinstance(sel, slice)
        lo = transform.domain.inclusive_min[old_dim]
        hi = transform.domain.exclusive_max[old_dim]
        start, step, origin, size = _resolve_slice_ts(sel, old_dim, lo, hi)
        slice_bounds.append((origin, origin + size))
        slice_dim_params[old_dim] = (start, step, origin)

    n_before = _broadcast_insertion_point(array_dims, slice_dims)

    # Build the new domain with NumPy's placement rule: the broadcast
    # (correlated) dimensions sit where the advanced indices sat when those are
    # adjacent, and lead when a slice separates them.
    new_inclusive_min = [lo for lo, _ in slice_bounds[:n_before]]
    new_exclusive_max = [hi for _, hi in slice_bounds[:n_before]]
    new_inclusive_min.extend([0] * len(broadcast_shape))
    new_exclusive_max.extend(broadcast_shape)
    new_inclusive_min.extend(lo for lo, _ in slice_bounds[n_before:])
    new_exclusive_max.extend(hi for _, hi in slice_bounds[n_before:])

    new_domain = IndexDomain(
        inclusive_min=tuple(new_inclusive_min),
        exclusive_max=tuple(new_exclusive_max),
    )

    # Build output maps
    array_dim_to_broadcast: dict[int, np.ndarray[Any, np.dtype[np.intp]]] = {}
    for i, d in enumerate(array_dims):
        array_dim_to_broadcast[d] = broadcast_arrays[i]

    # New dim index for slice dims starts after broadcast dims
    n_broadcast_dims = len(broadcast_shape)

    new_output: list[OutputIndexMap] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            new_output.append(m)
        elif isinstance(m, DimensionMap):
            d = m.input_dimension
            if d in array_dim_to_broadcast:
                # Normalize to full input rank: the broadcast (correlated) axes
                # come first, followed by a singleton axis per slice dimension.
                # Every vectorized array shares the same broadcast axes, so the
                # dependency axes derived from the shape coincide — the signature
                # of a pointwise scatter rather than an outer product.
                broadcast_arr = array_dim_to_broadcast[d]
                full_arr = broadcast_arr.reshape(
                    (1,) * n_before + broadcast_shape + (1,) * (len(slice_dims) - n_before)
                )
                new_output.append(array_map_or_constant(full_arr, offset=m.offset, stride=m.stride))
            else:
                # Slice dim: new coord `origin + k` maps to old `start + k*step`
                start, step, origin = slice_dim_params[d]
                new_offset = m.offset + m.stride * (start - step * origin)
                new_stride = m.stride * step
                position = slice_dims.index(d)
                new_input_dim = position if position < n_before else position + n_broadcast_dims
                new_output.append(
                    DimensionMap(
                        input_dimension=new_input_dim, offset=new_offset, stride=new_stride
                    )
                )
        else:
            # m: ArrayMap — unreachable: array-carrying transforms took the
            # composition path at the top of this function.
            raise AssertionError(  # noqa: TRY004 - unreachable, not a dispatch
                "unreachable: ArrayMap transforms are composed"
            )

    return IndexTransform(domain=new_domain, output=tuple(new_output))


_LITERAL_HINT = (
    "; within this transform layer, indices are literal domain coordinates (the "
    "public Array boundary wraps NumPy-style negatives before they reach here)"
)


def _trunc_div(a: int, b: int) -> int:
    """Integer division rounded toward zero (C semantics), as TensorStore uses
    for strided-slice domain origins — distinct from Python's floor division
    for negative operands (`trunc(-9/2) == -4` where `-9 // 2 == -5`)."""
    q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


def _resolve_slice_ts(sel: slice, dim: int, lo: int, hi: int) -> tuple[int, int, int, int]:
    """Resolve a slice against domain `[lo, hi)` with TensorStore semantics.

    Slice bounds are **literal domain coordinates** — never from-the-end, never
    clamped. One rule covers both signs of the step (each part verified against
    tensorstore 0.1.84, and matching ndsel 1.0-draft.2 section 5.3):

    - defaults follow the direction of travel: `start = lo`, `stop = hi` going
      up; `start = hi - 1`, `stop = lo - 1` going down;
    - the traversal runs from `start` toward `stop`, which is excluded, so the
      source interval is `[start, stop)` going up and `[stop + 1, start + 1)`
      going down;
    - a non-empty interval must be contained in the domain (no clamping — a
      NumPy-style out-of-range or negative bound is an error, not a shorter or
      wrapped result);
    - an empty interval is valid anywhere, for either sign;
    - an interval running the wrong way (`stop` on the far side of `start` from
      the direction of travel) is an error, not an empty result;
    - the result's domain origin is `trunc(start/step)` — toward zero, for both
      signs — and coordinate `origin + k` maps to input `start + k*step`.

    A negative step normally produces a negative origin: reversing a
    zero-origin axis of length 20 gives the domain `[-19, 1)`. The coordinate
    frame stays anchored to the source; a caller that needs non-negative
    coordinates re-bases explicitly with `translate_domain_to`.

    Returns `(start, step, origin, size)` in domain coordinates.
    """
    start_bound = None if sel.start is None else require_index(sel.start)
    stop_bound = None if sel.stop is None else require_index(sel.stop)
    step = 1 if sel.step is None else require_index(sel.step)
    if step == 0:
        raise IndexError("slice step must not be zero")
    if step > 0:
        start = lo if start_bound is None else start_bound
        stop = hi if stop_bound is None else stop_bound
        interval_lo, interval_hi = start, stop
    else:
        start = hi - 1 if start_bound is None else start_bound
        stop = lo - 1 if stop_bound is None else stop_bound
        interval_lo, interval_hi = stop + 1, start + 1
    length = interval_hi - interval_lo
    if length < 0:
        raise IndexError(
            f"slice from {start} to {stop} with step {step} does not specify a "
            f"valid interval for dimension {dim}: the derived interval "
            f"[{interval_lo}, {interval_hi}) runs the wrong way. An empty "
            "selection is spelled stop == start."
        )
    if length > 0 and (interval_lo < lo or interval_hi > hi):
        hint = _LITERAL_HINT if (start < 0 or stop < 0) and lo >= 0 else ""
        raise BoundsCheckError(
            f"slice interval [{interval_lo}, {interval_hi}) is not contained "
            f"within domain [{lo}, {hi}) for dimension {dim}{hint}"
        )
    size = -(-length // abs(step))  # ceil(length / |step|)
    origin = _trunc_div(start, step)
    return start, step, origin, size


def _check_array_in_bounds(arr: np.ndarray[Any, np.dtype[np.intp]], lo: int, hi: int) -> None:
    """Reject index-array values outside the domain `[lo, hi)`.

    Index-array values are literal domain coordinates (TensorStore semantics):
    a value below `inclusive_min` is out of bounds rather than counting from
    the end. Out-of-range values raise instead of silently wrapping.
    """
    if arr.size == 0:
        return
    lo_val, hi_val = int(arr.min()), int(arr.max())
    if lo_val < lo:
        hint = _LITERAL_HINT if lo_val < 0 and lo >= 0 else ""
        raise BoundsCheckError(
            f"index {lo_val} is out of bounds (valid indices [{lo}, {hi})){hint}"
        )
    if hi_val >= hi:
        raise BoundsCheckError(f"index {hi_val} is out of bounds (valid indices [{lo}, {hi}))")


def _validate_array_selection(selection: Any, shape: tuple[int, ...], mode: str) -> None:
    """Validate array-based selections (orthogonal, vectorized).

    Rejects types that are not valid for coordinate/vectorized indexing.
    Does not check bounds — the transform operations handle that.
    """
    items = selection if isinstance(selection, tuple) else (selection,)
    for sel in items:
        if isinstance(sel, slice):
            # vindex is coordinate-only (matches eager zarr): every axis needs an
            # integer/boolean array, never a slice. Orthogonal (oindex) allows slices.
            if mode == "vectorized":
                raise VindexInvalidSelectionError(
                    "unsupported selection type for vectorized indexing; only "
                    "coordinate selection (tuple of integer arrays) and mask selection "
                    f"(single Boolean array) are supported; got {selection!r}"
                )
            continue
        if sel is Ellipsis or as_scalar_index(sel) is not None:
            continue
        if isinstance(sel, (list, np.ndarray)):
            if mode == "orthogonal":
                array = np.asarray(sel)
                # An orthogonal selection is per-axis, so an integer array names
                # coordinates along one axis and can only be one-dimensional.
                # Left to the engine, this surfaced much later as a rank
                # complaint about an `index_array` the caller never wrote.
                if array.dtype.kind in "iu" and array.ndim > 1:
                    raise IndexError(
                        f"integer arrays in an orthogonal selection must be "
                        f"1-dimensional only; got one with {array.ndim} dimensions"
                    )
            continue
        raise IndexError(f"unsupported selection type for {mode} indexing: {type(sel)!r}")


def _validate_basic_selection(selection: Any) -> None:
    """Validate that a selection only contains basic indexing types (int, slice, Ellipsis).

    Rejects None (newaxis), arrays, lists, floats, strings, etc.
    """
    items = selection if isinstance(selection, tuple) else (selection,)
    for s in items:
        if s is Ellipsis or isinstance(s, slice) or as_scalar_index(s) is not None:
            continue
        raise IndexError(f"unsupported selection type for basic indexing: {type(s)!r}")
