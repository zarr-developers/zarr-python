import pathlib

import pytest
from zarr.store import LocalStore, MemoryStore, RemoteStore, StorePath


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request):
    return request.param


# todo: harmonize this with local_store fixture
@pytest.fixture
def store_path(tmpdir):
    store = LocalStore(str(tmpdir))
    p = StorePath(store)
    return p


@pytest.fixture(scope="function")
def local_store(tmpdir):
    return LocalStore(str(tmpdir))


@pytest.fixture(scope="function")
def remote_store():
    return RemoteStore()


@pytest.fixture(scope="function")
def memory_store():
    return MemoryStore()
