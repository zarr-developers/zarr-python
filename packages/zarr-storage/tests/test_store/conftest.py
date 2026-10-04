from __future__ import annotations

import importlib
import os
import pathlib
import sys
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, settings
from zarr import config
from zarr.core.sync import sync

from zarr_storage.legacy import (
    FsspecStore,
    LatencyStore,
    LocalStore,
    MemoryStore,
    Store,
    StorePath,
    ZipStore,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Literal

    from zarr.core.common import ZarrFormat


@pytest.fixture(autouse=True)
def extracted_storage_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real Zarr arrays with the proposed storage re-exports.

    This is only a test harness for future runtime adoption. Rebind references
    to the old storage definitions in already imported Zarr modules, including
    aliases imported with ``from ... import ...``. No methods are mocked and
    importing zarr_storage itself never changes Zarr's runtime bindings.
    Pytest restores every binding at teardown; contract tests outside this
    directory continue to compare against the unmodified original classes.
    """
    module_pairs = {
        "zarr.abc.store": "zarr_storage.legacy._abc",
        "zarr.core._coalesce": "zarr_storage._coalesce",
        "zarr.testing.store": "zarr_storage.testing.store",
        "zarr.experimental.cache_store": "zarr_storage.legacy.experimental.cache_store",
    }
    for suffix in (
        "_common",
        "_fsspec",
        "_local",
        "_logging",
        "_memory",
        "_obstore",
        "_utils",
        "_wrapper",
        "_zip",
    ):
        module_pairs[f"zarr.storage.{suffix}"] = f"zarr_storage.legacy.{suffix}"
    replacements: dict[int, object] = {}
    for old_name, new_name in module_pairs.items():
        old = importlib.import_module(old_name)
        new = importlib.import_module(new_name)
        for name, value in vars(old).items():
            if (
                getattr(value, "__module__", None) == old_name
                or name in {"StoreLike", "ByteRequest"}
            ) and hasattr(new, name):
                replacements[id(value)] = getattr(new, name)
    for module_name, module in tuple(sys.modules.items()):
        if module_name == "zarr" or module_name.startswith("zarr."):
            for name, value in tuple(vars(module).items()):
                if id(value) in replacements:
                    monkeypatch.setattr(module, name, replacements[id(value)])


async def parse_store(
    store: Literal["local", "memory", "fsspec", "zip", "memory_get_latency"], path: str
) -> LocalStore | MemoryStore | FsspecStore | ZipStore | LatencyStore:
    if store == "local":
        return await LocalStore.open(path)
    if store == "memory":
        return await MemoryStore.open()
    if store == "fsspec":
        return await FsspecStore.open(url=path)
    if store == "zip":
        return await ZipStore.open(f"{path}/zarr.zip", mode="w")
    if store == "memory_get_latency":
        return LatencyStore(MemoryStore(), get_latency=0.0001, set_latency=0.0)
    raise AssertionError


@pytest.fixture(params=[str, pathlib.Path])
def path_type(request: pytest.FixtureRequest) -> Any:
    return request.param


@pytest.fixture
async def store_path(tmp_path: pathlib.Path) -> StorePath:
    store = await LocalStore.open(str(tmp_path))
    return StorePath(store)


@pytest.fixture
async def local_store(tmp_path: pathlib.Path) -> LocalStore:
    return await LocalStore.open(str(tmp_path))


@pytest.fixture
async def memory_store() -> MemoryStore:
    return await MemoryStore.open()


@pytest.fixture
async def zip_store(tmp_path: pathlib.Path) -> ZipStore:
    return await ZipStore.open(str(tmp_path / "zarr.zip"), mode="w")


@pytest.fixture
async def store(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> Store:
    param = request.param
    return await parse_store(param, str(tmp_path))


@pytest.fixture
async def store2(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> Store:
    """Fixture to create a second store for testing copy operations between stores"""
    param = request.param
    store2_path = tmp_path / "store2"
    store2_path.mkdir()
    return await parse_store(param, str(store2_path))


@pytest.fixture(params=["local", "memory", "zip"])
def sync_store(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> Store:
    result = sync(parse_store(request.param, str(tmp_path)))
    if not isinstance(result, Store):
        raise TypeError(f"Wrong store class returned by test fixture! got {result} instead")
    return result


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


@pytest.fixture(params=(2, 3), ids=["zarr2", "zarr3"])
def zarr_format(request: pytest.FixtureRequest) -> ZarrFormat:
    if request.param == 2:
        return 2
    elif request.param == 3:
        return 3
    msg = f"Invalid zarr format requested. Got {request.param}, expected on of (2,3)."
    raise ValueError(msg)


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--run-slow-hypothesis",
        action="store_true",
        default=False,
        help="run slow hypothesis tests",
    )


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if config.getoption("--run-slow-hypothesis"):
        return
    skip_slow_hyp = pytest.mark.skip(reason="need --run-slow-hypothesis option to run")
    for item in items:
        if "slow_hypothesis" in item.keywords:
            item.add_marker(skip_slow_hyp)


@pytest.fixture(scope="session")
def moto_server() -> Generator[str, None, None]:
    """Start a session-scoped moto S3 server and yield its endpoint URL.

    The server binds an ephemeral port (port=0), so the endpoint is only known at
    runtime; consumers must take it from this fixture rather than a constant. A fixed
    port deadlocks under pytest-xdist: session-scoped fixtures run once per *worker*, so
    concurrent workers race to bind the same port, and the losers block forever inside
    ThreadedMotoServer.start(), whose server thread dies on "Address already in use"
    before ever setting the ready event that start() waits on.

    importorskip lives inside the fixture so moto is only required when a test actually
    requests an S3 backend, not for the whole test session."""
    moto_server_mod = pytest.importorskip("moto.moto_server.threaded_moto_server")

    server = moto_server_mod.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    # moto needs *some* credentials present; use throwaway values if the environment has none.
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foo")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "foo")
    try:
        yield f"http://{host}:{port}/"
    finally:
        server.stop()


settings.register_profile(
    "storage",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "storage"))
