import pathlib
import pytest

from zarr.v3.store import LocalStore, StorePath, MemoryStore
from zarr.v3.store.remote import RemoteStore


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


@pytest.fixture(scope="function")
def store(request: str, tmpdir):
    param = request.param
    if param == "local_store":
        return LocalStore(str(tmpdir))
    elif param == "memory_store":
        return MemoryStore()
    elif param == "remote_store":
        return RemoteStore()
    else:
        assert False
