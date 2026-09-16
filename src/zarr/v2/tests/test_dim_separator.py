import pathlib

import pytest
from numpy.testing import assert_array_equal
from functools import partial

import zarr
from zarr.core import Array
from zarr.storage import DirectoryStore, NestedDirectoryStore, FSStore
from zarr.tests.util import have_fsspec


needs_fsspec = pytest.mark.skipif(not have_fsspec, reason="needs fsspec")


@pytest.fixture(
    params=(
        "static_flat",
        "static_flat_legacy",
        "static_nested",
        "static_nested_legacy",
        "directory_nested",
        "directory_flat",
        "directory_default",
        "nesteddirectory_nested",
        "nesteddirectory_default",
        pytest.param("fs_nested", marks=needs_fsspec),
        pytest.param("fs_flat", marks=needs_fsspec),
        pytest.param("fs_default", marks=needs_fsspec),
    )
)
def dataset(tmpdir, request):
    """
    Generate a variety of different Zarrs using
    different store implementations as well as
    different dimension_separator arguments.
    """

    loc = tmpdir.join("dim_sep_test.zarr")
    which = request.param
    kwargs = {}

    if which.startswith("static"):
        project_root = pathlib.Path(zarr.__file__).resolve().parent.parent
        suffix = which[len("static_") :]
        static = project_root / "fixture" / suffix

        if not static.exists():  # pragma: no cover
            if "nested" in which:
                # No way to reproduce the nested_legacy file via code
                generator = NestedDirectoryStore
            else:
                if "legacy" in suffix:
                    # No dimension_separator metadata included
                    generator = DirectoryStore
                else:
                    # Explicit dimension_separator metadata included
                    generator = partial(DirectoryStore, dimension_separator=".")

            # store the data - should be one-time operation
            s = generator(str(static))
            a = zarr.open(store=s, mode="w", shape=(2, 2), dtype="<i8")
            a[:] = [[1, 2], [3, 4]]

        return str(static)

    if which.startswith("directory"):
        store_class = DirectoryStore
    elif which.startswith("nested"):
        store_class = NestedDirectoryStore
    else:
        store_class = FSStore
        kwargs["mode"] = "w"
        kwargs["auto_mkdir"] = True

    if which.endswith("nested"):
        kwargs["dimension_separator"] = "/"
    elif which.endswith("flat"):
        kwargs["dimension_separator"] = "."

    store = store_class(str(loc), **kwargs)
    zarr.creation.array(store=store, data=[[1, 2], [3, 4]])
    return str(loc)


def verify(array, expect_failure=False):
    try:
        assert_array_equal(array[:], [[1, 2], [3, 4]])
    except AssertionError:
        if expect_failure:
            pytest.xfail()
        else:
            raise  # pragma: no cover


def test_open(dataset):
    """
    Use zarr.open to open the dataset fixture. Legacy nested datasets
    without the dimension_separator metadata are not expected to be
    openable.
    """
    failure = "nested_legacy" in dataset
    verify(zarr.open(dataset, "r"), failure)


@needs_fsspec
def test_fsstore(dataset):
    """
    Use FSStore to open the dataset fixture. Legacy nested datasets
    without the dimension_separator metadata are not expected to be
    openable.
    """
    failure = "nested_legacy" in dataset
    verify(Array(store=FSStore(dataset)), failure)


def test_directory(dataset):
    """
    Use DirectoryStore to open the dataset fixture. Legacy nested datasets
    without the dimension_separator metadata are not expected to be
    openable.
    """
    failure = "nested_legacy" in dataset
    verify(zarr.Array(store=DirectoryStore(dataset)), failure)


def test_nested(dataset):
    """
    Use NestedDirectoryStore to open the dataset fixture. This is the only
    method that is expected to successfully open legacy nested datasets
    without the dimension_separator metadata. However, for none-Nested
    datasets without any metadata, NestedDirectoryStore will fail.
    """
    failure = "flat_legacy" in dataset or "directory_default" in dataset or "fs_default" in dataset
    verify(Array(store=NestedDirectoryStore(dataset)), failure)
