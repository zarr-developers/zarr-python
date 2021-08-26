import pathlib

import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.core import Array
from zarr.storage import (DirectoryStore, NestedDirectoryStore, FSStore)
from zarr.tests.util import have_fsspec


needs_fsspec = pytest.mark.skipif(not have_fsspec, reason="needs fsspec")


@pytest.fixture(params=("static_nested",
                        "static_flat",
                        "directory_nested",
                        "directory_flat",
                        "directory_default",
                        "nesteddirectory_nested",
                        "nesteddirectory_default",
                        pytest.param("fs_nested", marks=needs_fsspec),
                        pytest.param("fs_flat", marks=needs_fsspec),
                        pytest.param("fs_default", marks=needs_fsspec)))
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
        if which.endswith("nested"):
            static = project_root / "fixture/nested"
            generator = NestedDirectoryStore
        else:
            static = project_root / "fixture/flat"
            generator = DirectoryStore

        if not static.exists():  # pragma: no cover
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


def verify(array):
    assert_array_equal(array[:], [[1, 2], [3, 4]])


def test_open(dataset):
    verify(zarr.open(dataset, "r"))


@needs_fsspec
def test_fsstore(dataset):
    verify(Array(store=FSStore(dataset)))


def test_directory(dataset):
    verify(zarr.Array(store=DirectoryStore(dataset)))


def test_nested(dataset):
    verify(Array(store=NestedDirectoryStore(dataset)))
