import pytest
from numpy.testing import assert_array_equal

import zarr
from zarr.core import Array
from zarr.storage import (DirectoryStore, NestedDirectoryStore, FSStore)
from zarr.tests.util import have_fsspec


@pytest.fixture(params=("static_nested",
                        "static_flat",
                        "directory_nested",
                        "directory_flat",
                        "directory_default",
                        "nesteddirectory_nested",
                        "nesteddirectory_default",
                        "fs_nested",
                        "fs_flat",
                        "fs_default"))
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
        if which.endswith("nested"):
            return "fixture/nested"
        else:
            return "fixture/flat"

    if which.startswith("directory"):
        store_class = DirectoryStore
    elif which.startswith("nested"):
        store_class = NestedDirectoryStore
    else:
        if have_fsspec is False:
            pytest.skip("no fsspec")
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
    verify(zarr.open(dataset))


def test_fsstore(dataset):
    verify(Array(store=FSStore(dataset)))


def test_directory(dataset):
    verify(zarr.Array(store=DirectoryStore(dataset)))


def test_nested(dataset):
    verify(Array(store=NestedDirectoryStore(dataset)))
