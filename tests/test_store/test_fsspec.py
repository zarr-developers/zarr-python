from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from packaging.version import parse as parse_version

import zarr.api.asynchronous
from zarr import Array
from zarr.abc.store import OffsetByteRequest
from zarr.core.buffer import Buffer, cpu, default_buffer_prototype
from zarr.core.sync import _collect_aiterator, sync
from zarr.storage import FsspecStore
from zarr.storage._fsspec import _make_async
from zarr.testing.store import StoreTests

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Generator
    from pathlib import Path

    import botocore.client
    import s3fs

    from zarr.core.common import JSON


# Warning filter due to https://github.com/boto/boto3/issues/3889
pytestmark = [
    pytest.mark.filterwarnings(
        re.escape("ignore:datetime.datetime.utcnow() is deprecated:DeprecationWarning")
    ),
    # TODO: fix these warnings
    pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning"),
    pytest.mark.filterwarnings(
        "ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited:RuntimeWarning"
    ),
]

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
    buf = await store.get("foo", prototype=default_buffer_prototype())
    assert buf is not None
    assert buf.to_bytes() == data
    out = await store.get_partial_values(
        prototype=default_buffer_prototype(), key_ranges=[("foo", OffsetByteRequest(1))]
    )
    assert out[0] is not None
    assert out[0].to_bytes() == data[1:]


class TestFsspecStoreS3(StoreTests[FsspecStore, cpu.Buffer]):
    store_cls = FsspecStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self) -> dict[str, str | bool]:
        try:
            from fsspec import url_to_fs
        except ImportError:
            # before fsspec==2024.3.1
            from fsspec.core import url_to_fs
        fs, path = url_to_fs(
            f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False, asynchronous=True
        )
        return {"fs": fs, "path": path}

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> FsspecStore:
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

        meta: dict[str, JSON] = {
            "attributes": {"key": "value"},
            "zarr_format": 3,
            "node_type": "group",
        }

        await store.set(
            "zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value"}

        meta = {
            "attributes": {"key": "value-2"},
            "zarr_format": 3,
            "node_type": "group",
        }
        await store.set(
            "directory-2/zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}/directory-2", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value-2"}

        meta = {
            "attributes": {"key": "value-3"},
            "zarr_format": 3,
            "node_type": "group",
        }
        await store.set(
            "directory-3/zarr.json",
            self.buffer_cls.from_bytes(json.dumps(meta).encode()),
        )
        group = await zarr.api.asynchronous.open_group(
            store=f"s3://{test_bucket_name}", path="directory-3", storage_options=storage_options
        )
        assert dict(group.attrs) == {"key": "value-3"}

    @pytest.mark.skipif(
        parse_version(fsspec.__version__) < parse_version("2024.03.01"),
        reason="Prior bug in from_upath",
    )
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

    def test_init_raises_if_path_has_scheme(self, store_kwargs: dict[str, Any]) -> None:
        # regression test for https://github.com/zarr-developers/zarr-python/issues/2342
        store_kwargs["path"] = "s3://" + store_kwargs["path"]
        with pytest.raises(
            ValueError, match="path argument to FsspecStore must not include scheme .*"
        ):
            self.store_cls(**store_kwargs)

    def test_init_warns_if_fs_asynchronous_is_false(self) -> None:
        try:
            from fsspec import url_to_fs
        except ImportError:
            # before fsspec==2024.3.1
            from fsspec.core import url_to_fs
        fs, path = url_to_fs(
            f"s3://{test_bucket_name}", endpoint_url=endpoint_url, anon=False, asynchronous=False
        )
        store_kwargs = {"fs": fs, "path": path}
        with pytest.warns(UserWarning, match=r".* was not created with `asynchronous=True`.*"):
            self.store_cls(**store_kwargs)

    async def test_empty_nonexistent_path(self, store_kwargs: dict[str, Any]) -> None:
        # regression test for https://github.com/zarr-developers/zarr-python/pull/2343
        store_kwargs["path"] += "/abc"
        store = await self.store_cls.open(**store_kwargs)
        assert await store.is_empty("")

    async def test_delete_dir_unsupported_deletes(self, store: FsspecStore) -> None:
        store.supports_deletes = False
        with pytest.raises(
            NotImplementedError,
            match="This method is only available for stores that support deletes.",
        ):
            await store.delete_dir("test_prefix")


def array_roundtrip(store: FsspecStore) -> None:
    """
    Round trip an array using a Zarr store

    Args:
        store: FsspecStore
    """
    data = np.ones((3, 3))
    arr = zarr.create_array(store=store, overwrite=True, data=data)
    assert isinstance(arr, Array)
    # Read set values
    arr2 = zarr.open_array(store=store)
    assert isinstance(arr2, Array)
    np.testing.assert_array_equal(arr[:], data)


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
def test_wrap_sync_filesystem(tmp_path: pathlib.Path) -> None:
    """The local fs is not async so we should expect it to be wrapped automatically"""
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    store = FsspecStore.from_url(f"file://{tmp_path}", storage_options={"auto_mkdir": True})
    assert isinstance(store.fs, AsyncFileSystemWrapper)
    assert store.fs.async_impl
    array_roundtrip(store)


@pytest.mark.skipif(
    parse_version(fsspec.__version__) >= parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
def test_wrap_sync_filesystem_raises(tmp_path: pathlib.Path) -> None:
    """The local fs is not async so we should expect it to be wrapped automatically"""
    with pytest.raises(ImportError, match="The filesystem .*"):
        FsspecStore.from_url(f"file://{tmp_path}", storage_options={"auto_mkdir": True})


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
def test_no_wrap_async_filesystem() -> None:
    """An async fs should not be wrapped automatically; fsspec's s3 filesystem is such an fs"""
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    store = FsspecStore.from_url(
        f"s3://{test_bucket_name}/foo/spam/",
        storage_options={"endpoint_url": endpoint_url, "anon": False, "asynchronous": True},
        read_only=False,
    )
    assert not isinstance(store.fs, AsyncFileSystemWrapper)
    assert store.fs.async_impl
    array_roundtrip(store)


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
def test_open_fsmap_file(tmp_path: pathlib.Path) -> None:
    min_fsspec_with_async_wrapper = parse_version("2024.12.0")
    current_version = parse_version(fsspec.__version__)

    fs = fsspec.filesystem("file", auto_mkdir=True)
    mapper = fs.get_mapper(tmp_path)

    if current_version < min_fsspec_with_async_wrapper:
        # Expect ImportError for older versions
        with pytest.raises(
            ImportError,
            match=r"The filesystem .* is synchronous, and the required AsyncFileSystemWrapper is not available.*",
        ):
            array_roundtrip(mapper)
    else:
        # Newer versions should work
        array_roundtrip(mapper)


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
def test_open_fsmap_file_raises(tmp_path: pathlib.Path) -> None:
    fsspec = pytest.importorskip("fsspec.implementations.local")
    fs = fsspec.LocalFileSystem(auto_mkdir=False)
    mapper = fs.get_mapper(tmp_path)
    with pytest.raises(FileNotFoundError, match="No such file or directory: .*"):
        array_roundtrip(mapper)


@pytest.mark.parametrize("asynchronous", [True, False])
def test_open_fsmap_s3(asynchronous: bool) -> None:
    s3_filesystem = s3fs.S3FileSystem(
        asynchronous=asynchronous, endpoint_url=endpoint_url, anon=False
    )
    mapper = s3_filesystem.get_mapper(f"s3://{test_bucket_name}/map/foo/")
    array_roundtrip(mapper)


def test_open_s3map_raises() -> None:
    with pytest.raises(TypeError, match="Unsupported type for store_like:.*"):
        zarr.open(store=0, mode="w", shape=(3, 3))
    s3_filesystem = s3fs.S3FileSystem(asynchronous=True, endpoint_url=endpoint_url, anon=False)
    mapper = s3_filesystem.get_mapper(f"s3://{test_bucket_name}/map/foo/")
    with pytest.raises(
        ValueError, match="'path' was provided but is not used for FSMap store_like objects"
    ):
        zarr.open(store=mapper, path="bar", mode="w", shape=(3, 3))
    with pytest.raises(
        ValueError,
        match="'storage_options was provided but is not used for FSMap store_like objects",
    ):
        zarr.open(store=mapper, storage_options={"anon": True}, mode="w", shape=(3, 3))


@pytest.mark.parametrize("asynchronous", [True, False])
def test_make_async(asynchronous: bool) -> None:
    s3_filesystem = s3fs.S3FileSystem(
        asynchronous=asynchronous, endpoint_url=endpoint_url, anon=False
    )
    fs = _make_async(s3_filesystem)
    assert fs.asynchronous


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
async def test_delete_dir_wrapped_filesystem(tmp_path: Path) -> None:
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
    from fsspec.implementations.local import LocalFileSystem

    wrapped_fs = AsyncFileSystemWrapper(LocalFileSystem(auto_mkdir=True))
    store = FsspecStore(wrapped_fs, read_only=False, path=f"{tmp_path}/test/path")

    assert isinstance(store.fs, AsyncFileSystemWrapper)
    assert store.fs.asynchronous

    await store.set("zarr.json", cpu.Buffer.from_bytes(b"root"))
    await store.set("foo-bar/zarr.json", cpu.Buffer.from_bytes(b"root"))
    await store.set("foo/zarr.json", cpu.Buffer.from_bytes(b"bar"))
    await store.set("foo/c/0", cpu.Buffer.from_bytes(b"chunk"))
    await store.delete_dir("foo")
    assert await store.exists("zarr.json")
    assert await store.exists("foo-bar/zarr.json")
    assert not await store.exists("foo/zarr.json")
    assert not await store.exists("foo/c/0")


@pytest.mark.skipif(
    parse_version(fsspec.__version__) < parse_version("2024.12.0"),
    reason="No AsyncFileSystemWrapper",
)
async def test_with_read_only_auto_mkdir(tmp_path: Path) -> None:
    """
    Test that creating a read-only copy of a store backed by the local file system does not error
    if auto_mkdir is False.
    """

    store_w = FsspecStore.from_url(f"file://{tmp_path}", storage_options={"auto_mkdir": False})
    _ = store_w.with_read_only()
