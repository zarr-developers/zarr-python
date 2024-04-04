from __future__ import annotations
from typing import TYPE_CHECKING

from zarr.v3.common import ZarrFormat
from zarr.v3.group import AsyncGroup

if TYPE_CHECKING:
    from typing import Any, Literal
from dataclasses import dataclass, field
import pathlib

import pytest

from zarr.v3.config import RuntimeConfiguration
from zarr.v3.store import LocalStore, StorePath, MemoryStore
from zarr.v3.store.remote import RemoteStore


def parse_store(
    store: Literal["local", "memory", "remote"], path: str
) -> LocalStore | MemoryStore | RemoteStore:
    if store == "local":
        return LocalStore(path)
    if store == "memory":
        return MemoryStore()
    if store == "remote":
        return RemoteStore()
    assert False


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
    return parse_store(param, str(tmpdir))


@dataclass
class AsyncGroupRequest:
    zarr_format: ZarrFormat
    store: str
    attributes: dict[str, Any] = field(default_factory=dict)
    runtime_configuration: RuntimeConfiguration = RuntimeConfiguration()


@pytest.fixture(scope="function")
async def async_group(request: AsyncGroupRequest, tmpdir) -> AsyncGroup:
    param: AsyncGroupRequest = request.param

    store = parse_store(param.store, str(tmpdir))
    agroup = await AsyncGroup.create(
        store,
        attributes=param.attributes,
        zarr_format=param.zarr_format,
        runtime_configuration=param.runtime_configuration,
        exists_ok=False,
    )
    return agroup
