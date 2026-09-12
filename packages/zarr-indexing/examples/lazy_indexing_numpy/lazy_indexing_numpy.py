# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr-indexing>=0.1",
#   "numpy==2.4.3",
#   "pytest==9.0.2"
# ]
# ///
#

"""
Demonstrate lazy indexing over a plain NumPy array with zarr_indexing.LazyArray
"""

import sys

import numpy as np
import pytest

from zarr_indexing import LazyArray


def test_wrap_and_compose() -> None:
    """Wrap an array, compose selections without reading, then materialize once."""
    data = np.arange(12 * 8).reshape(12, 8)
    lazy = LazyArray.from_numpy(data)

    # The wrapper forwards the attributes an array consumer expects.
    assert lazy.shape == (12, 8)
    assert lazy.dtype == data.dtype
    assert lazy.ndim == 2

    # `.lazy[...]` returns another LazyArray. No element of `data` is read.
    view = lazy.lazy[2:10, ::2]
    print(view)
    assert view.shape == (8, 4)

    # Selections compose. Each step narrows the view; still nothing is read.
    smaller = view.lazy[1:5, 1:3]

    # `result()` performs the read. NumPy is the reference for the whole chain.
    assert np.array_equal(smaller.result(), data[2:10, ::2][1:5, 1:3])

    # Selections use positional NumPy semantics: indices count from zero within
    # the current view, and negative indices count from the end.
    assert np.array_equal(lazy.lazy[-1].result(), data[-1])
    assert np.array_equal(lazy.lazy[::-1].result(), data[::-1])

    # Orthogonal and vectorized indexing are available under the same accessor.
    rows = np.array([9, 1, 4])
    assert np.array_equal(lazy.lazy.oindex[rows, :].result(), data[rows, :])
    cols = np.array([0, 3, 7])
    assert np.array_equal(lazy.lazy.vindex[rows, cols].result(), data[rows, cols])

    # A LazyArray is also an ordinary duck array: __getitem__ reads immediately,
    # and np.asarray materializes the view.
    assert np.array_equal(lazy[2:4, 0], data[2:4, 0])
    assert np.array_equal(np.asarray(view), data[2:10, ::2])


def test_box_and_query_selections() -> None:
    """Distinguish selections that describe a region from selections that gather points."""
    data = np.arange(12 * 8).reshape(12, 8)
    lazy = LazyArray.from_numpy(data)

    # A box selection is built from slices and integers alone. It is described
    # completely by an interval and a step per dimension, so a consumer can
    # serve it as one strided read.
    box = lazy.lazy[2:10, ::2]
    print(f"box: is_box={box.is_box} bounding_box={box.bounding_box()} strides={box.strides()}")
    assert box.is_box
    assert box.bounding_box() == ((2, 10), (0, 7))
    assert box.strides() == (1, 2)

    # A query selection gathers points through an index array. Its coordinates
    # are a lookup table, so `strides()` is undefined and `bounding_box()` is
    # the hull of the points rather than an exact description.
    query = lazy.lazy.oindex[np.array([9, 1, 4]), :]
    print(f"query: is_box={query.is_box} bounding_box={query.bounding_box()}")
    assert not query.is_box
    assert query.strides() is None
    assert query.bounding_box() == ((1, 10), (0, 8))

    # Composing a box onto a query keeps it a query.
    assert not query.lazy[0:2, 0:2].is_box


def test_parts() -> None:
    """Iterate the partitions a view covers, and assemble the result from them."""
    data = np.arange(12 * 8).reshape(12, 8)

    # A plain NumPy array declares no partitioning, so `with_parts` states one.
    # Partitioning changes the granularity of reads, never the result.
    lazy = LazyArray.from_numpy(data).with_parts((4, 4))
    view = lazy.lazy[2:10, ::2]

    parts = list(view.parts())
    print(f"{len(parts)} parts")
    for part in parts[:2]:
        print(f"  base_coords={part.base_coords} box={part.box} complete={part.is_complete}")

    # Each part carries a sub-view of its own, where that sub-view lands in the
    # result, and whether it covers its partition completely. Resolving the
    # parts and placing them is what `result()` does.
    assembled = np.empty(view.shape, dtype=view.dtype)
    for part in parts:
        assembled[part.out_selection] = part.view.result()
    assert np.array_equal(assembled, view.result())

    # The partitioning is a read strategy, so a different one gives the same data.
    assert np.array_equal(
        LazyArray.from_numpy(data).with_parts((5, 3)).lazy[2:10, ::2].result(), assembled
    )


if __name__ == "__main__":
    # Run the example with printed output, and a dummy pytest configuration file specified.
    # Without the dummy configuration file, at test time pytest will attempt to use the
    # configuration file in the project root, which will error because Zarr is using some
    # plugins that are not installed in this example.
    sys.exit(
        pytest.main(
            [
                "-s",
                __file__,
                f"-c {__file__}",
                # Suppress: "PytestAssertRewriteWarning: Module already imported so
                # cannot be rewritten; zarr"
                "-W",
                "ignore::pytest.PytestAssertRewriteWarning",
            ]
        )
    )
