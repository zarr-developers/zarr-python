import os

import fsspec
import pytest
from upath import UPath

from zarr.buffer import Buffer, cpu, default_buffer_prototype
from zarr.store import RemoteStore
from zarr.sync import sync
from zarr.testing.store import StoreTests

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
def s3_base():
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


def get_boto3_client():
    from botocore.session import Session

    # NB: we use the sync botocore client for setup
    session = Session()
    return session.create_client("s3", endpoint_url=endpoint_url)


@pytest.fixture(autouse=True, scope="function")
def s3(s3_base):
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


async def alist(it):
    out = []
    async for a in it:
        out.append(a)
    return out


async def test_basic():
    store = RemoteStore(f"s3://{test_bucket_name}", mode="w", endpoint_url=endpoint_url, anon=False)
    assert not await alist(store.list())
    assert not await store.exists("foo")
    data = b"hello"
    await store.set("foo", cpu.Buffer.from_bytes(data))
    assert await store.exists("foo")
    assert (await store.get("foo", prototype=default_buffer_prototype)).to_bytes() == data
    out = await store.get_partial_values(
        prototype=default_buffer_prototype, key_ranges=[("foo", (1, None))]
    )
    assert out[0].to_bytes() == data[1:]


class TestRemoteStoreS3(StoreTests[RemoteStore, cpu.Buffer]):
    store_cls = RemoteStore
    buffer_cls = cpu.Buffer

    @pytest.fixture(scope="function", params=("use_upath", "use_str"))
    def store_kwargs(self, request) -> dict[str, str | bool]:
        url = f"s3://{test_bucket_name}"
        anon = False
        mode = "w"
        if request.param == "use_upath":
            return {"mode": mode, "url": UPath(url, endpoint_url=endpoint_url, anon=anon)}
        elif request.param == "use_str":
            return {"url": url, "mode": mode, "anon": anon, "endpoint_url": endpoint_url}

        raise AssertionError

    @pytest.fixture(scope="function")
    def store(self, store_kwargs: dict[str, str | bool]) -> RemoteStore:
        url = store_kwargs["url"]
        mode = store_kwargs["mode"]
        if isinstance(url, UPath):
            out = self.store_cls(url=url, mode=mode)
        else:
            endpoint_url = store_kwargs["endpoint_url"]
            out = self.store_cls(url=url, asynchronous=True, mode=mode, endpoint_url=endpoint_url)
        return out

    def get(self, store: RemoteStore, key: str) -> Buffer:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        fs, _ = fsspec.url_to_fs(
            url=store._url,
            asynchronous=False,
            anon=store._fs.anon,
            endpoint_url=store._fs.endpoint_url,
        )
        return self.buffer_cls.from_bytes(fs.cat(f"{store.path}/{key}"))

    def set(self, store: RemoteStore, key: str, value: Buffer) -> None:
        #  make a new, synchronous instance of the filesystem because this test is run in sync code
        fs, _ = fsspec.url_to_fs(
            url=store._url,
            asynchronous=False,
            anon=store._fs.anon,
            endpoint_url=store._fs.endpoint_url,
        )
        fs.write_bytes(f"{store.path}/{key}", value.to_bytes())

    def test_store_repr(self, store: RemoteStore) -> None:
        assert str(store) == f"s3://{test_bucket_name}"

    def test_store_supports_writes(self, store: RemoteStore) -> None:
        assert True

    @pytest.mark.xfail
    def test_store_supports_partial_writes(self, store: RemoteStore) -> None:
        raise AssertionError

    def test_store_supports_listing(self, store: RemoteStore) -> None:
        assert True
