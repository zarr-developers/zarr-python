"""Output index maps — three ordered mappings to integer coordinates.

An output index map describes how request cells address one dimension of
storage. Its coordinates form an **ordered, duplicate-preserving sequence**
aligned with the input domain, never a mathematical set. Three representations
cover the cases that arise in practice:

- `ConstantMap(offset=5)` — every request cell maps to coordinate `5`
- `DimensionMap(input_dimension=0, offset=3, stride=2)` over input `[0, 5)`
  — the ordered arithmetic progression `[3, 5, 7, 9, 11]`
- `ArrayMap(index_array=[5, 1, 1])` — the explicit sequence `[5, 1, 1]`,
  preserving both order and the repeated coordinate

Every output map participates in two operations defined on `IndexTransform`,
which provides the input-domain context these maps lack:

- **intersect** — retain mapped cells whose coordinates lie within a range
  (e.g., a chunk), without changing their order or multiplicity.
  Restricting `[3, 5, 5, 9]` to `[4, 8)` produces `[5, 5]`.
- **translate** — shift every coordinate by a constant (e.g., make chunk-local).
  Translating `[5, 5, 7]` by `-4` produces `[1, 1, 3]`.

These two operations are the foundation of chunk resolution: for each chunk,
intersect the map with the chunk's range, then translate to chunk-local
coordinates.

The three types exist because they trade off generality for efficiency:

- `ConstantMap`: O(1) storage, O(1) intersection
- `DimensionMap`: O(1) storage, O(1) intersection (analytical)
- `ArrayMap`: O(n) storage, O(n) intersection (must scan the array)

Collapsing everything to `ArrayMap` would be correct but wasteful — a
billion-element slice would materialize a billion coordinates just to group
them by chunk, when `DimensionMap` does it with three integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing.affine import checked_affine

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ConstantMap:
    """A constant storage-coordinate mapping.

    Every input cell maps to `offset`. Arises from integer indexing (e.g.,
    `arr[5]` fixes one dimension to coordinate 5).
    """

    offset: int = 0
    """The fixed storage coordinate every input cell maps to."""


@dataclass(frozen=True, slots=True)
class DimensionMap:
    """An ordered affine mapping to storage coordinates.

    Maps each input coordinate `i` to `offset + stride * i`, where the input
    range comes from the enclosing `IndexTransform`'s domain. Arises from slice
    indexing (e.g., `arr[2:10:3]` gives offset=2, stride=3).
    """

    input_dimension: int
    """The input (domain) dimension whose coordinate this map reads."""

    offset: int = 0
    """The storage coordinate that input coordinate `0` maps to."""

    stride: int = 1
    """The storage step per unit input step; negative walks backward, zero repeats `offset`."""


@dataclass(frozen=True, slots=True)
class ArrayMap:
    """An explicit ordered, duplicate-preserving coordinate mapping.

    Maps each input position `i` to `offset + stride * index_array[i]`.
    Index-array order and repeated entries are semantic and remain present in
    the result. Arises from fancy indexing (e.g., `arr[[5, 1, 1]]` or boolean
    masks).

    Freshly constructed maps are normalized to the **full input rank** of their
    enclosing transform: `index_array` has the enclosing domain's rank, sized
    fully on the axes it varies over and singleton (size 1) elsewhere. The
    shape is the single source of truth for what the map depends on — its
    **dependency axes** are exactly its non-singleton axes (see
    `transform._array_map_dependency_axes`) — and it distinguishes the two
    flavors of multi-array fancy indexing:

    - **orthogonal** (`oindex`): each array varies along a single, *distinct*
      axis (all others singleton); the result is their outer product.
    - **vectorized** (`vindex`): the arrays are correlated and share the same
      non-singleton (broadcast) axes; the result is a pointwise scatter.

    A map holding exactly one coordinate carries no shape to read a dependency
    from, and none is needed: it is the `ConstantMap` it equals, and the
    selection layer builds that instead (see `array_map_or_constant`). A
    hand-built all-singleton `ArrayMap` is still a valid value; resolution
    classifies it with the correlated maps and reads it pointwise.
    """

    index_array: npt.NDArray[np.integer[Any]]
    """Explicit coordinates at the enclosing transform's full input rank; order and
    duplicates are semantic. Its non-singleton axes are the map's dependency axes."""

    offset: int = 0
    """Constant term of the affine adjustment: storage is `offset + stride * index_array[i]`."""

    stride: int = 1
    """Multiplier applied to each `index_array` value before `offset` is added."""

    def __post_init__(self) -> None:
        """Own the index array and expose it read-only.

        A map is frozen, but the array inside it was not: reaching through a
        view's transform to `index_array[0] = 9` silently changed what the view
        returned, in a package whose whole contract is that a view is a
        description of a read and resolving it twice answers alike. Owning the
        array also prevents the caller from changing the contents behind the
        read-only view, which would invalidate this value object's hash.
        """
        # Immutable bytes are the ultimate owner so callers cannot re-enable
        # the WRITEABLE flag, as they can on a read-only array that owns its
        # allocation. `asarray` also accepts the NumPy scalars that reach here
        # after indexing an array down to one element.
        array = np.asarray(self.index_array)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"index_array must have an integer dtype, got {array.dtype}")
        normalized = checked_affine(0, 1, array)
        frozen = np.frombuffer(normalized.tobytes(), dtype=np.intp).reshape(normalized.shape)
        object.__setattr__(self, "index_array", frozen)

    def __reduce__(self) -> tuple[object, tuple[object, int, int]]:
        """Reconstruct through `__init__`, preserving the ownership invariant."""
        return (
            type(self),
            (self.index_array, self.offset, self.stride),
        )

    def __eq__(self, other: object) -> bool:
        """Value equality, comparing index arrays element-wise.

        The generated `__eq__` compares them with `==`, whose result for two
        arrays is an array — so asking whether two maps are equal raised
        `ValueError: the truth value of an array ... is ambiguous`. `frozen=True`
        reads as a promise that a value can be compared and hashed, and this is
        what makes good on it.
        """
        if not isinstance(other, ArrayMap):
            return NotImplemented
        return (
            self.offset == other.offset
            and self.stride == other.stride
            and self.index_array.shape == other.index_array.shape
            and bool(np.array_equal(self.index_array, other.index_array))
        )

    def __hash__(self) -> int:
        """Hashed by the array's contents, so equal maps hash alike.

        The generated `__hash__` hashed the ndarray itself, which is unhashable;
        a map could therefore not go in a set, or key a cache.
        """
        return hash(
            (
                self.offset,
                self.stride,
                self.index_array.shape,
                self.index_array.tobytes(),
            )
        )


def array_map_or_constant(
    index_array: npt.NDArray[np.integer[Any]],
    offset: int = 0,
    stride: int = 1,
) -> ArrayMap | ConstantMap:
    """An `ArrayMap`, collapsed to the `ConstantMap` it equals when it can be.

    An index array holding exactly one coordinate maps every input cell to the
    same place; representing it as a lookup table would leave a map whose shape
    names no dependency axis, the one form the shape-derived classifier cannot
    read. The selection and composition layers build their array maps through
    this helper so that a non-empty `ArrayMap` always varies over at least one
    axis. An empty array stays an `ArrayMap`: it maps no cell at all, and the
    emptiness lives in the domain that accompanies it.
    """
    arr = np.asarray(index_array)
    if arr.size == 1:
        return ConstantMap(offset=checked_affine(offset, stride, int(arr.reshape(-1)[0])))
    return ArrayMap(index_array=arr, offset=offset, stride=stride)


OutputIndexMap = ConstantMap | DimensionMap | ArrayMap
