from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest
from packaging.version import parse as parse_version

import zarr.api.asynchronous
from zarr.abc.store import OffsetByteRequest
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.core.sync import _collect_aiterator, sync
from zarr.storage import FsspecStore
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    from collections.abc import Generator

    import botocore.client


fsspec = pytest.importorskip("fsspec")
s3fs = pytest.importorskip("s3fs")
requests = pytest.importorskip("requests")
moto_server = pytest.importorskip("moto.moto_server.threaded_moto_server")
moto = pytest.importorskip("moto")
botocore = pytest.importorskip("botocore")

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
    session = botocore.session.Session()
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
    store = FsspecStore.from_url(
        f"s3://{test_bucket_name}/foo/spam/",
        storage_options={"endpoint_url": endpoint_url, "anon": False},
    )
    assert store.fs.asynchronous
    assert store.path == f"{test_bucket_name}/foo/spam"
    assert await _collect_aiterator(store.list()) == ()
    assert not await store.exists("foo")
    data = b"hello"
    await store.set("foo", cpu.Buffer.from_bytes(data))
    assert await store.exists("foo")
    assert (await store.get("foo", prototype=default_buffer_prototype())).to_bytes() == data
    out = await store.get_partial_values(
        prototype=default_buffer_prototype(), key_ranges=[("foo", OffsetByteRequest(1))]
    )
    assert out[0].to_bytes() == data[1:]


class TestFsspecStoreS3(StoreTests[FsspecStore, cpu.Buffer]):
    store_cls = FsspecStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, request) -> dict[str, str | bool]:
        fs, path = fsspec.url_to_fs(
            f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False, asynchronous=True
        )
        return {"fs": fs, "path": path}

    @pytest.fixture
    def store(self, store_kwargs: dict[str, str | bool]) -> FsspecStore:
        return self.store_cls(**store_kwargs)

    async def get(self, store: FsspecStore, key: str) -> Buffer:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        new_fs = fsspec.filesystem(
            "s3", endpoint_url=store.fs.endpoint_url, anon=store.fs.anon, asynchronous=False
        )
        return self.buffer_cls.from_bytes(new_fs.cat(f"{store.path}/{key}"))

    async def set(self, store: FsspecStore, key: str, value: Buffer) -> None:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        new_fs = fsspec.filesystem(
            "s3", endpoint_url=store.fs.endpoint_url, anon=store.fs.anon, asynchronous=False
        )
        new_fs.write_bytes(f"{store.path}/{key}", value.to_bytes())

    def test_store_repr(self, store: FsspecStore) -> None:
        assert str(store) == "<FsspecStore(S3FileSystem, test)>"

    def test_store_supports_writes(self, store: FsspecStore) -> None:
        assert store.supports_writes

    def test_store_supports_partial_writes(self, store: FsspecStore) -> None:
        assert not store.supports_partial_writes

    def test_store_supports_listing(self, store: FsspecStore) -> None:
        assert store.supports_listing

    async def test_fsspec_store_from_uri(self, store: FsspecStore) -> None:
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
        upath = pytest.importorskip("upath")
        path = upath.UPath(
            f"s3://{test_bucket_name}/foo/bar/",
            endpoint_url=endpoint_url,
            anon=False,
            asynchronous=True,
        )
        result = FsspecStore.from_upath(path)
        assert result.fs.endpoint_url == endpoint_url
        assert result.fs.asynchronous
        assert result.path == f"{test_bucket_name}/foo/bar"

    def test_init_raises_if_path_has_scheme(self, store_kwargs) -> None:
        # regression test for https://github.com/zarr-developers/zarr-python/issues/2342
        store_kwargs["path"] = "s3://" + store_kwargs["path"]
        with pytest.raises(
            ValueError, match="path argument to FsspecStore must not include scheme .*"
        ):
            self.store_cls(**store_kwargs)

    def test_init_warns_if_fs_asynchronous_is_false(self) -> None:
        fs, path = fsspec.url_to_fs(
            f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False, asynchronous=False
        )
        store_kwargs = {"fs": fs, "path": path}
        with pytest.warns(UserWarning, match=r".* was not created with `asynchronous=True`.*"):
            self.store_cls(**store_kwargs)

    async def test_empty_nonexistent_path(self, store_kwargs) -> None:
        # regression test for https://github.com/zarr-developers/zarr-python/pull/2343
        store_kwargs["path"] += "/abc"
        store = await self.store_cls.open(**store_kwargs)
        assert await store.is_empty("")


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
def test_wrap_sync_filesystem():
    """The local fs is not async so we should expect it to be wrapped automatically"""
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    store = FsspecStore.from_url("local://test/path")

    assert isinstance(store.fs, AsyncFileSystemWrapper)
    assert store.fs.async_impl


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
def test_no_wrap_async_filesystem():
    """An async fs should not be wrapped automatically; fsspec's https filesystem is such an fs"""
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    store = FsspecStore.from_url("https://test/path")

    assert not isinstance(store.fs, AsyncFileSystemWrapper)
    assert store.fs.async_impl
