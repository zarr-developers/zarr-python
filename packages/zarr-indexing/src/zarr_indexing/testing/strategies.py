"""Hypothesis strategies for the selections `LazyArray` accepts.

Each strategy takes the shape of the array being indexed and generates one
selection for it — an index tuple with one entry per axis, in the spelling its
mode expects. They are the generators behind
[`ChainedIndexingStateMachine`][zarr_indexing.testing.stateful.ChainedIndexingStateMachine]
and are exported on their own for a project that has its own test harness and
wants only the hard part.

```python
from hypothesis import given, strategies as st
from zarr_indexing.testing.strategies import basic_selections

@given(selection=basic_selections((7, 5, 4)))
def test_my_array_slices_like_numpy(selection):
    assert_array_equal(my_array[selection], reference[selection])
```

Every axis of `shape` must be non-empty: a selection over an axis of extent 0
has no coordinates to draw. Filter or narrow the shape before calling.

Requires the `testing` extra (`pip install zarr-indexing[testing]`).
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import numpy as np
from hypothesis import strategies as st

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "basic_selections",
    "masks",
    "orthogonal_selections",
    "slice_selections",
    "vectorized_selections",
]


def _entries(shape: tuple[int, ...], entry: Callable[[int], st.SearchStrategy[Any]]) -> Any:
    """One `entry` strategy per axis, as an index tuple."""
    return st.tuples(*[entry(size) for size in shape])


def _basic_entry(size: int) -> st.SearchStrategy[Any]:
    steps = st.integers(1, 3)
    return st.one_of(
        # A scalar integer drops its axis, in every mode, exactly as NumPy does.
        st.integers(-size, size - 1),
        st.builds(slice, st.integers(0, size), st.integers(0, size), steps),
        # Downward. The start is drawn from below `-size` as well, where the walk
        # begins off the front and selects nothing — a case that reads as an
        # ordinary negative index but is empty — and a stop that falls off the
        # front is spelled `None`.
        st.builds(
            slice,
            st.integers(-2 * size - 1, size - 1),
            st.none() | st.integers(0, size),
            steps.map(operator.neg),
        ),
        st.just(slice(None)),
    )


def _orthogonal_entry(size: int) -> st.SearchStrategy[Any]:
    coordinate = st.integers(-size, size - 1)
    return st.one_of(
        coordinate,
        st.lists(coordinate, min_size=1, max_size=4),
        masks((size,)),
        st.builds(slice, st.integers(0, size - 1), st.just(size)),
    )


def _slice_entry(size: int) -> st.SearchStrategy[slice]:
    return st.one_of(
        st.builds(slice, st.integers(0, size - 1), st.just(size), st.integers(1, 2)),
        st.just(slice(None, None, -1)),
        st.just(slice(None)),
    )


@st.composite
def masks(draw: st.DrawFn, shape: tuple[int, ...]) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Boolean masks over `shape`, each selecting at least one cell.

    An all-False mask is legal but is a separate concern — it empties the view,
    and a chain of selections is more interesting when every step leaves
    something to index — so one cell is always forced True.
    """
    size = int(np.prod(shape))
    flags = np.array(draw(st.lists(st.booleans(), min_size=size, max_size=size)))
    flags[draw(st.integers(0, size - 1))] = True
    return flags.reshape(shape)


def basic_selections(shape: tuple[int, ...]) -> st.SearchStrategy[tuple[Any, ...]]:
    """Basic selections: one scalar integer or slice per axis.

    Slices run in both directions, including the two empty spellings — a
    forward slice whose stop precedes its start, and a backward one whose start
    is off the front of the axis.
    """
    return _entries(shape, _basic_entry)


def orthogonal_selections(shape: tuple[int, ...]) -> st.SearchStrategy[tuple[Any, ...]]:
    """Orthogonal (`oindex`) selections: an outer product of per-axis choices.

    Each axis draws a scalar, a coordinate list (unsorted, with duplicates), a
    boolean mask, or a slice.
    """
    return _entries(shape, _orthogonal_entry)


def slice_selections(shape: tuple[int, ...]) -> st.SearchStrategy[tuple[Any, ...]]:
    """Selections of slices alone, for the `oindex` spelling that carries no coordinates.

    Such a step is not a fancy selection — it narrows the view's own axes and
    composes like basic indexing — so it is legal after a fancy step, where
    genuine coordinates are not. The starts reach past the origin, which is what
    distinguishes a step that walks an existing index array's dependency axes
    from one that walks its broadcast singletons.
    """
    return _entries(shape, _slice_entry)


@st.composite
def vectorized_selections(draw: st.DrawFn, shape: tuple[int, ...]) -> tuple[Any, ...]:
    """Vectorized (`vindex`) selections over a leading or trailing block of axes.

    `vindex` is coordinate-only — it rejects a slice outright — so a partial
    selection names its axes by position: a leading block, or a trailing one
    reached through an ellipsis. Either a single boolean mask spanning the whole
    covered block, or one entry per axis, each a coordinate array or a scalar
    (a scalar being a basic index NumPy applies before the coordinates).
    """
    ndim = len(shape)
    trailing = draw(st.booleans())
    count = draw(st.integers(1, ndim))
    axes = range(ndim - count, ndim) if trailing else range(count)
    sizes = [shape[axis] for axis in axes]

    entries: list[Any]
    if draw(st.booleans()):
        entries = [draw(masks(tuple(sizes)))]
    else:
        length = draw(st.integers(1, 4))
        entries = [
            draw(
                st.one_of(
                    st.integers(-size, size - 1),
                    st.lists(st.integers(-size, size - 1), min_size=length, max_size=length).map(
                        lambda values: np.array(values, dtype=np.intp)
                    ),
                )
            )
            for size in sizes
        ]
    return (Ellipsis, *entries) if trailing else tuple(entries)
