import os

import pytest

from zarr.buffer import Buffer, default_buffer_prototype
from zarr.store import RemoteStore
from zarr.store.core import _normalize_interval_index
from zarr.testing.store import StoreTests
from zarr.testing.utils import assert_bytes_equal

s3fs = pytest.importorskip("s3fs")
requests = pytest.importorskip("requests")
moto_server = pytest.importorskip("moto.moto_server.threaded_moto_server")
moto = pytest.importorskip("moto")

# ### amended from s3fs ### #
test_bucket_name = "test"
secure_bucket_name = "test-secure"
port = 5555
endpoint_uri = f"http://127.0.0.1:{port}/"


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
    return session.create_client("s3", endpoint_url=endpoint_uri)


@pytest.fixture()
def s3(s3_base):
    client = get_boto3_client()
    client.create_bucket(Bucket=test_bucket_name, ACL="public-read")
    s3fs.S3FileSystem.clear_instance_cache()
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs={"endpoint_url": endpoint_uri})
    s3.invalidate_cache()
    yield s3
    requests.post(f"{endpoint_uri}/moto-api/reset")


# ### end from s3fs ### #


async def alist(it):
    out = []
    async for a in it:
        out.append(a)
    return out


async def test_basic(s3):
    store = RemoteStore(f"s3://{test_bucket_name}", mode="w", endpoint_url=endpoint_uri, anon=False)
    assert not await alist(store.list())
    assert not await store.exists("foo")
    data = b"hello"
    await store.set("foo", Buffer.from_bytes(data))
    assert await store.exists("foo")
    assert (await store.get("foo")).to_bytes() == data
    out = await store.get_partial_values(
        prototype=default_buffer_prototype, key_ranges=[("foo", (1, None))]
    )
    assert out[0].to_bytes() == data[1:]


class TestRemoteStoreS3(StoreTests[RemoteStore]):
    store_cls = RemoteStore

    @pytest.fixture(scope="function")
    def store_kwargs(self) -> dict[str, str | bool]:
        return {
            "mode": "w",
            "endpoint_url": endpoint_uri,
            "anon": False,
            "url": f"s3://{test_bucket_name}",
        }

    @pytest.fixture(scope="function")
    def store(self, store_kwargs: dict[str, str | bool]) -> RemoteStore:
        return self.store_cls(**store_kwargs)

    def get(self, store: RemoteStore, key: str) -> Buffer:
        return Buffer.from_bytes(store._fs.get_mapper()[os.path.join(store.path, key)])

    def set(self, store: RemoteStore, key: str, value: Buffer) -> None:
        store._fs.get_mapper()[os.path.join(store.path, key)] = value.to_bytes()

    def test_store_repr(self, store: RemoteStore) -> None:
        assert str(store) == f"Remote fsspec store: {store.path}"

    def test_store_supports_writes(self, store: RemoteStore) -> None:
        assert True

    def test_store_supports_partial_writes(self, store: RemoteStore) -> None:
        assert True

    def test_store_supports_listing(self, store: RemoteStore) -> None:
        assert True

    @pytest.mark.parametrize("key", ["c/0", "foo/c/0.0", "foo/0/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    @pytest.mark.parametrize("byte_range", (None, (0, None), (1, None), (1, 2), (None, 1)))
    async def test_get(
        self,
        store: RemoteStore,
        key: str,
        data: bytes,
        byte_range: None | tuple[int | None, int | None],
        store_kwargs,
    ) -> None:
        """
        Ensure that data can be read from the store using the store.get method.
        """

        s3fs.S3FileSystem.clear_instance_cache()
        data_buf = Buffer.from_bytes(data)
        self.set(store, key, data_buf)
        store = self.store_cls(**store_kwargs)
        observed = await store.get(key, prototype=default_buffer_prototype, byte_range=byte_range)
        start, length = _normalize_interval_index(data_buf, interval=byte_range)
        expected = data_buf[start : start + length]
        assert_bytes_equal(observed, expected)
