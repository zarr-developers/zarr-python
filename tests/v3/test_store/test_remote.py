from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import fsspec
import pytest
from botocore.session import Session
from upath import UPath

import zarr.api.asynchronous
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.core.sync import _collect_aiterator, sync
from zarr.storage import RemoteStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from collections.abc import Generator

    import botocore.client


s3fs = pytest.importorskip("s3fs")
requests = pytest.importorskip("requests")
moto_server = pytest.importorskip("moto.moto_server.threaded_moto_server")
moto = pytest.importorskip("moto")

# ### amended from s3fs ### #
test_bucket_name = "test"
secure_bucket_name = "test-secure"
port = 5555
endpoint_url = f"http://127.0.0.1:{port}/"


@pytest.fixture(scope="module")
def s3_base() -> Generator[None, None, None]:
    # writable local S3 system

    # This fixture is module-scoped, meaning that we can reuse the MotoServer across all tests
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foo"
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = "foo"

    yield
    server.stop()


def get_boto3_client() -> botocore.client.BaseClient:
    # NB: we use the sync botocore client for setup
    session = Session()
    return session.create_client("s3", endpoint_url=endpoint_url)


@pytest.fixture(autouse=True)
def s3(s3_base: None) -> Generator[s3fs.S3FileSystem, None, None]:
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
    client = get_boto3_client()
    client.create_bucket(Bucket=test_bucket_name, ACL="public-read")
    s3fs.S3FileSystem.clear_instance_cache()
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": endpoint_url})
    session = sync(s3.set_session())
    s3.invalidate_cache()
    yield s3
    requests.post(f"{endpoint_url}/moto-api/reset")
    client.close()
    sync(session.close())


# ### end from s3fs ### #


async def test_basic() -> None:
    store = RemoteStore.from_url(
        f"s3://{test_bucket_name}",
        mode="w",
        storage_options={"endpoint_url": endpoint_url, "anon": False},
    )
    assert await _collect_aiterator(store.list()) == ()
    assert not await store.exists("foo")
    data = b"hello"
    await store.set("foo", cpu.Buffer.from_bytes(data))
    assert await store.exists("foo")
    assert (await store.get("foo", prototype=default_buffer_prototype())).to_bytes() == data
    out = await store.get_partial_values(
        prototype=default_buffer_prototype(), key_ranges=[("foo", (1, None))]
    )
    assert out[0].to_bytes() == data[1:]


class TestRemoteStoreS3(StoreTests[RemoteStore, cpu.Buffer]):
    store_cls = RemoteStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, request) -> dict[str, str | bool]:
        fs, path = fsspec.url_to_fs(
            f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False
        )
        return {"fs": fs, "path": path, "mode": "r+"}

    @pytest.fixture
    def store(self, store_kwargs: dict[str, str | bool]) -> RemoteStore:
        return self.store_cls(**store_kwargs)

    async def get(self, store: RemoteStore, key: str) -> Buffer:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        new_fs = fsspec.filesystem(
            "s3", endpoint_url=store.fs.endpoint_url, anon=store.fs.anon, asynchronous=False
        )
        return self.buffer_cls.from_bytes(new_fs.cat(f"{store.path}/{key}"))

    async def set(self, store: RemoteStore, key: str, value: Buffer) -> None:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        new_fs = fsspec.filesystem(
            "s3", endpoint_url=store.fs.endpoint_url, anon=store.fs.anon, asynchronous=False
        )
        new_fs.write_bytes(f"{store.path}/{key}", value.to_bytes())

    def test_store_repr(self, store: RemoteStore) -> None:
        assert str(store) == "<RemoteStore(S3FileSystem, test)>"

    def test_store_supports_writes(self, store: RemoteStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: RemoteStore) -> None:
        assert not store.supports_partial_writes

    def test_store_supports_listing(self, store: RemoteStore) -> None:
        assert store.supports_listing

    async def test_remote_store_from_uri(
        self, store: RemoteStore, store_kwargs: dict[str, str | bool]
    ):
        storage_options = {
            "endpoint_url": endpoint_url,
            "anon": False,
        }

        meta = {"attributes": {"key": "value"}, "zarr_format": 3, "node_type": "group"}

        await store.set(
            "zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value"}

        meta["attributes"]["key"] = "value-2"
        await store.set(
            "directory-2/zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}/directory-2", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value-2"}

        meta["attributes"]["key"] = "value-3"
        await store.set(
            "directory-3/zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}", path="directory-3", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value-3"}

    def test_from_upath(self) -> None:
        path = UPath(f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False)
        result = RemoteStore.from_upath(path)
        assert result.fs.endpoint_url == endpoint_url
