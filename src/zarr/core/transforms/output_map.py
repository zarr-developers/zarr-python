"""Output index maps — three representations of a set of integer coordinates.

An output index map describes, for one dimension of storage, which coordinates
an array access will touch. Conceptually it is a **set of integers**. Three
representations cover the cases that arise in practice:

- `ConstantMap(offset=5)` — a singleton set: `{5}`
- `DimensionMap(input_dimension=0, offset=3, stride=2)` over input `[0, 5)`
  — an arithmetic progression: `{3, 5, 7, 9, 11}`
- `ArrayMap(index_array=[1, 5, 9])` — an explicit enumeration: `{1, 5, 9}`

Every output map supports two set-theoretic operations (defined on
`IndexTransform`, which provides the input domain context these maps lack):

- **intersect** — restrict to coordinates within a range (e.g., a chunk).
  `{3, 5, 7, 9, 11} ∩ [4, 8) = {5, 7}`
- **translate** — shift every coordinate by a constant (e.g., make chunk-local).
  `{5, 7} - 4 = {1, 3}`

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ConstantMap:
    """A singleton set: one storage coordinate.

    Represents `{offset}`. Arises from integer indexing (e.g., `arr[5]`
    fixes one dimension to coordinate 5).
    """

    offset: int = 0


@dataclass(frozen=True, slots=True)
class DimensionMap:
    """An arithmetic progression of storage coordinates.

    Represents `{offset + stride * i : i in input_range}`, where the input
    range comes from the enclosing `IndexTransform`'s domain. Arises from
    slice indexing (e.g., `arr[2:10:3]` gives offset=2, stride=3).
    """

    input_dimension: int
    offset: int = 0
    stride: int = 1


@dataclass(frozen=True, slots=True)
class ArrayMap:
    """An explicit enumeration of storage coordinates.

    Represents `{offset + stride * index_array[i] : i in input_range}`.
    Arises from fancy indexing (e.g., `arr[[1, 5, 9]]` or boolean masks).

    Freshly constructed maps are normalized to the **full input rank** of their
    enclosing transform: `index_array` has the enclosing domain's rank, sized
    fully on the axes it varies over and singleton (size 1) elsewhere. The
    dependency axes are therefore derivable from the shape (see
    `transform._array_map_dependency_axes`), which distinguishes the two flavours
    of multi-array fancy indexing:

    - **orthogonal** (`oindex`): each array varies along a single, *distinct*
      axis (all others singleton); the result is their outer product.
    - **vectorized** (`vindex`): the arrays are correlated and share the same
      non-singleton (broadcast) axes; the result is a pointwise scatter.

    `input_dimension` records the single axis an orthogonal array varies over
    (`None` for vectorized), binding it the way `DimensionMap` is bound. It is
    usually redundant with the shape-derived classifier, but stays authoritative
    for the shapes the classifier cannot distinguish: a length-1 orthogonal
    selection normalizes to an all-singleton array (no non-singleton axis), and
    length-1 vectorized arrays are equally degenerate. `None` therefore marks a
    map as correlated, and an integer pins the dependency axis of a degenerate
    orthogonal map (see `transform._array_map_dependent_axis`).
    """

    index_array: npt.NDArray[np.intp]
    offset: int = 0
    stride: int = 1
    input_dimension: int | None = None


OutputIndexMap = ConstantMap | DimensionMap | ArrayMap
