from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, Verbosity, settings

from zarr import AsyncGroup, config
from zarr.store import LocalStore, MemoryStore, StorePath, ZipStore
from zarr.store.remote import RemoteStore

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from types import ModuleType
    from typing import Any, Literal

    from _pytest.compat import LEGACY_PATH

    from zarr.abc.store import Store
    from zarr.core.common import ChunkCoords, MemoryOrder, ZarrFormat


async def parse_store(
    store: Literal["local", "memory", "remote", "zip"], path: str
) -> LocalStore | MemoryStore | RemoteStore | ZipStore:
    if store == "local":
        return await LocalStore.open(path, mode="w")
    if store == "memory":
        return await MemoryStore.open(mode="w")
    if store == "remote":
        return await RemoteStore.open(url=path, mode="w")
    if store == "zip":
        return await ZipStore.open(path + "/zarr.zip", mode="w")
    raise AssertionError


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request: pytest.FixtureRequest) -> Any:
    return request.param


# todo: harmonize this with local_store fixture
@pytest.fixture
async def store_path(tmpdir: LEGACY_PATH) -> StorePath:
    store = await LocalStore.open(str(tmpdir), mode="w")
    p = StorePath(store)
    return p


@pytest.fixture(scope="function")
async def local_store(tmpdir: LEGACY_PATH) -> LocalStore:
    return await LocalStore.open(str(tmpdir), mode="w")


@pytest.fixture(scope="function")
async def remote_store(url: str) -> RemoteStore:
    return await RemoteStore.open(url, mode="w")


@pytest.fixture(scope="function")
async def memory_store() -> MemoryStore:
    return await MemoryStore.open(mode="w")


@pytest.fixture(scope="function")
async def zip_store(tmpdir: LEGACY_PATH) -> ZipStore:
    return await ZipStore.open(str(tmpdir / "zarr.zip"), mode="w")


@pytest.fixture(scope="function")
async def store(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> Store:
    param = request.param
    return await parse_store(param, str(tmpdir))


@dataclass
class AsyncGroupRequest:
    zarr_format: ZarrFormat
    store: Literal["local", "remote", "memory", "zip"]
    attributes: dict[str, Any] = field(default_factory=dict)


@pytest.fixture(scope="function")
async def async_group(request: pytest.FixtureRequest, tmpdir: LEGACY_PATH) -> AsyncGroup:
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

    if request.param == "cupy":
        request.node.add_marker(pytest.mark.gpu)

    yield pytest.importorskip(request.param)


@pytest.fixture(autouse=True)
def reset_config() -> Generator[None, None, None]:
    config.reset()
    yield
    config.reset()


@dataclass
class ArrayRequest:
    shape: ChunkCoords
    dtype: str
    order: MemoryOrder


@pytest.fixture
def array_fixture(request: pytest.FixtureRequest) -> npt.NDArray[Any]:
    array_request: ArrayRequest = request.param
    return (
        np.arange(np.prod(array_request.shape))
        .reshape(array_request.shape, order=array_request.order)
        .astype(array_request.dtype)
    )


settings.register_profile(
    "ci",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
settings.register_profile(
    "local",
    max_examples=300,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    verbosity=Verbosity.verbose,
)
