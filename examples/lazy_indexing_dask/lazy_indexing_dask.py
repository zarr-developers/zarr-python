# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",
#   "zarr-indexing @ git+https://github.com/zarr-developers/zarr-python.git@main#subdirectory=packages/zarr-indexing",
#   "dask[array]==2025.3.0",
#   "numpy==2.4.3",
#   "pytest==9.0.2"
# ]
# ///
#

"""
Demonstrate using zarr_indexing.LazyArray with Dask
"""

import sys

import dask
import dask.array as da
import numpy as np
import pytest
from dask.base import tokenize
from zarr_indexing import LazyArray

import zarr


@pytest.fixture
def source() -> zarr.Array:
    """A chunked Zarr array to wrap."""
    array = zarr.create_array(store={}, shape=(40, 30), chunks=(10, 10), dtype="i4")
    array[:] = np.arange(40 * 30).reshape(40, 30)
    return array


def test_from_array(source: zarr.Array) -> None:
    """Hand a LazyArray to `dask.array.from_array`."""
    lazy = LazyArray(source)

    # `from_array` needs `shape`, `dtype`, and `__getitem__`, which the wrapper
    # provides. Each Dask block reads its own region through the wrapper.
    array = da.from_array(lazy, chunks=(10, 10))
    print(array)
    assert np.array_equal(array.compute(scheduler="threads"), source[:])

    # A view works the same way, and its shape is the shape of the selection.
    view = LazyArray(source).lazy[5:35, 3:27]
    array = da.from_array(view, chunks=(10, 10))
    assert array.shape == (30, 24)
    assert np.array_equal(array.compute(scheduler="threads"), source[5:35, 3:27])


def test_parts_as_tasks(source: zarr.Array) -> None:
    """Build one task per partition and compute them in parallel."""
    view = LazyArray(source).lazy[5:35, 3:27]

    # The partitioning is discovered from the wrapped array's chunks, so each
    # partition of the view lies within one stored chunk.
    parts = list(view.parts())
    print(f"{len(parts)} parts for a {view.shape} view of a {source.shape} array")

    # A partition carries a sub-view to resolve and where its result belongs, so
    # the reads are independent and the placement needs no coordination.
    @dask.delayed
    def read(part: object) -> np.ndarray:
        return part.array.result()

    blocks = dask.compute(*[read(part) for part in parts], scheduler="threads")

    result = np.empty(view.shape, dtype=view.dtype)
    for part, block in zip(parts, blocks, strict=True):
        result[part.out_selection] = block
    assert np.array_equal(result, source[5:35, 3:27])

    # `is_complete` reports whether a partition covers its whole partition of
    # the base array, which a writer uses to choose between overwriting a chunk
    # and reading it first.
    complete = [part.box for part in parts if part.is_complete]
    print(f"{len(complete)} of {len(parts)} parts cover their chunk completely")


def test_tokenize(source: zarr.Array) -> None:
    """Deterministic tokens let Dask cache and deduplicate work."""
    lazy = LazyArray(source)

    # Two wrappers over the same array and the same selection are the same task
    # to Dask, whether or not they are the same Python object.
    assert tokenize(lazy) == tokenize(LazyArray(source))
    assert tokenize(lazy.lazy[0:10]) == tokenize(LazyArray(source).lazy[0:10])

    # Different selections are different tasks.
    assert tokenize(lazy.lazy[0:10]) != tokenize(lazy.lazy[10:20])

    # Selections that describe the same region are the same task, however they
    # were composed.
    assert tokenize(lazy.lazy[0:20].lazy[5:10]) == tokenize(lazy.lazy[5:10])


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
