from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from botocore.session import Session
from hypothesis import HealthCheck, Verbosity, settings

from zarr import AsyncGroup, config
from zarr.abc.store import Store
from zarr.core.sync import sync
from zarr.storage import LocalStore, MemoryStore, StorePath, ZipStore
from zarr.storage.remote import RemoteStore

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator
    from typing import Any, Literal

    import botocore
    from _pytest.compat import LEGACY_PATH

    from zarr.core.common import ChunkCoords, MemoryOrder, ZarrFormat

s3fs = pytest.importorskip("s3fs")
requests = pytest.importorskip("requests")
moto_server = pytest.importorskip("moto.moto_server.threaded_moto_server")
moto = pytest.importorskip("moto")

# ### amended from s3fs ### #
test_bucket_name = "test"
secure_bucket_name = "test-secure"


async def parse_store(
    store: str,
    path: str,
    s3: s3fs.S3FileSystem,  # type: ignore[name-defined]
) -> LocalStore | MemoryStore | RemoteStore | ZipStore:
    """
    Take a string representation of a store and convert that string representation
    into the appropriate store object, which is then returned.
    """

    match store:
        case "local":
            return LocalStore(path, read_only=False)
        case "memory":
            return MemoryStore(read_only=False)
        case "remote":
            return RemoteStore(fs=s3, path=test_bucket_name, read_only=False)
        case "zip":
            return await ZipStore.open(path + "/zarr.zip", read_only=False, mode="w")

    raise AssertionError


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request: pytest.FixtureRequest) -> Any:
    """
    A pytest fixture that provides a parameterized path type.

    This fixture yields different types of path representations
    for testing purposes. The possible types are `str` and
    `pathlib.Path`. It can be used to test functions or methods
    that need to handle different path type inputs.

    Returns:
        The path type specified by the current parameter.
    """
    return request.param


# todo: harmonize this with local_store fixture
@pytest.fixture
async def store_path(tmpdir: LEGACY_PATH) -> StorePath:
    store = await LocalStore.open(str(tmpdir))
    return StorePath(store)


@pytest.fixture
async def store(
    request: pytest.FixtureRequest,
    tmpdir: LEGACY_PATH,
    s3: s3fs.S3FileSystem,  # type: ignore[name-defined]
) -> AsyncGenerator[Store, None]:
    param = request.param
    store_instance = await parse_store(param, str(tmpdir), s3)
    yield store_instance
    store_instance.close()


@pytest.fixture(params=["local", "memory", "zip"])
def sync_store(request: pytest.FixtureRequest, tmp_path: LEGACY_PATH, s3_base: str) -> Store:
    result = sync(parse_store(request.param, str(tmp_path), s3_base))
    if not isinstance(result, Store):
        raise TypeError("Wrong store class returned by test fixture! got " + result + " instead")
    return result


@dataclass
class AsyncGroupRequest:
    zarr_format: ZarrFormat
    store: Literal["local", "remote", "memory", "zip"]
    attributes: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
async def async_group(
    request: pytest.FixtureRequest, tmpdir: LEGACY_PATH, s3_base: str
) -> AsyncGroup:
    param: AsyncGroupRequest = request.param

    store = await parse_store(param.store, str(tmpdir), s3_base)
    return await AsyncGroup.from_store(
        store,
        attributes=param.attributes,
        zarr_format=param.zarr_format,
        exists_ok=False,
    )


@pytest.fixture(params=["numpy", "cupy"])
def xp(request: pytest.FixtureRequest) -> Any:
    """Fixture to parametrize over numpy-like libraries"""

    if request.param == "cupy":
        request.node.add_marker(pytest.mark.gpu)

    return pytest.importorskip(request.param)


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


@pytest.fixture(params=(2, 3), ids=["zarr2", "zarr3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    if request.param == 2:
        return 2
    elif request.param == 3:
        return 3
    msg = f"Invalid zarr format requested. Got {request.param}, expected on of (2,3)."
    raise ValueError(msg)


@pytest.fixture(scope="module")
def s3_base() -> Generator[str, None, None]:
    # writable local S3 system
    from moto.server import ThreadedMotoServer

    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server._server.server_address
    endpoint_url = f"http://{host}:{port}"

    yield endpoint_url
    server.stop()


def get_boto3_client(endpoint_url: str) -> botocore.client.BaseClient:
    # NB: we use the sync botocore client for setup
    session = Session()
    return session.create_client("s3", endpoint_url=endpoint_url, region_name="us-east-1")


@pytest.fixture(autouse=True)
def s3(s3_base: str) -> Generator[s3fs.S3FileSystem, None, None]:  # type: ignore[name-defined]
    """
    Quoting Martin Durant:
    pytest-asyncio creates a new event loop for each async test.
    When an async-mode s3fs instance is made from async, it will be assigned to the loop from
    which it is made. That means that if you use s3fs again from a subsequent test,
    you will have the same identical instance, but be running on a different loop - which fails.

    For the rest: it's very convenient to clean up the state of the store between tests,
    make sure we start off blank each time.

    https://github.com/zarr-developers/zarr-python/pull/1785#discussion_r1634856207
    """
    client = get_boto3_client(s3_base)
    client.create_bucket(Bucket=test_bucket_name, ACL="public-read")
    s3fs.S3FileSystem.clear_instance_cache()
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": s3_base}, asynchronous=True)
    session = sync(s3.set_session())
    s3.invalidate_cache()
    yield s3
    requests.post(f"{s3_base}/moto-api/reset")
    client.close()
    sync(session.close())


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
