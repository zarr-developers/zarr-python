from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING

from zarr.common import ZarrFormat
from zarr.group import AsyncGroup

if TYPE_CHECKING:
    from typing import Any, Literal
import pathlib
from dataclasses import dataclass, field

import pytest

from zarr.store import LocalStore, MemoryStore, StorePath
from zarr.store.remote import RemoteStore


async def parse_store(
    store: Literal["local", "memory", "remote"], path: str
) -> LocalStore | MemoryStore | RemoteStore:
    if store == "local":
        return await LocalStore.open(path, mode="w")
    if store == "memory":
        return await MemoryStore.open(mode="w")
    if store == "remote":
        return await RemoteStore.open(mode="w")
    raise AssertionError


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request):
    return request.param


# todo: harmonize this with local_store fixture
@pytest.fixture
async def store_path(tmpdir):
    store = await LocalStore.open(str(tmpdir), mode="w")
    p = StorePath(store)
    return p


@pytest.fixture(scope="function")
async def local_store(tmpdir):
    return await LocalStore.open(str(tmpdir), mode="w")


@pytest.fixture(scope="function")
async def remote_store():
    return await RemoteStore.open(mode="w")


@pytest.fixture(scope="function")
async def memory_store():
    return await MemoryStore.open(mode="w")


@pytest.fixture(scope="function")
async def store(request: str, tmpdir):
    param = request.param
    return await parse_store(param, str(tmpdir))


@dataclass
class AsyncGroupRequest:
    zarr_format: ZarrFormat
    store: Literal["local", "remote", "memory"]
    attributes: dict[str, Any] = field(default_factory=dict)


@pytest.fixture(scope="function")
async def async_group(request: pytest.FixtureRequest, tmpdir) -> AsyncGroup:
    param: AsyncGroupRequest = request.param

    store = await parse_store(param.store, str(tmpdir))
    agroup = await AsyncGroup.create(
        store,
        attributes=param.attributes,
        zarr_format=param.zarr_format,
        exists_ok=False,
    )
    return agroup


@pytest.fixture(params=["numpy", "cupy"])
def xp(request: pytest.FixtureRequest) -> Iterator[ModuleType]:
    """Fixture to parametrize over numpy-like libraries"""

    yield pytest.importorskip(request.param)
