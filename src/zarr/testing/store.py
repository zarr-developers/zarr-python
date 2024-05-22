from typing import Generic, TypeVar

import pytest

from zarr.abc.store import Store
from zarr.buffer import Buffer


def _normalize_byte_range(
    data: bytes, byte_range: None | tuple[int | None, int | None]
) -> tuple[int, int]:
    """
    Convert an implicit byte range into an explicit start and length
    """
    if byte_range is None:
        start = 0
        length = len(data)
    else:
        maybe_start, maybe_len = byte_range
        if maybe_start is None:
            start = 0
        else:
            start = maybe_start

        if maybe_len is None:
            length = len(data) - start
        else:
            length = maybe_len

    return (start, length)


S = TypeVar("S", bound=Store)


class StoreTests(Generic[S]):
    store_cls: type[S]

    @pytest.fixture(scope="function")
    def store(self) -> Store:
        return self.store_cls()

    def test_store_type(self, store: S) -> None:
        assert isinstance(store, Store)
        assert isinstance(store, self.store_cls)

    def test_store_repr(self, store: S) -> None:
        raise NotImplementedError

    def test_store_supports_writes(self, store: S) -> None:
        raise NotImplementedError

    def test_store_supports_partial_writes(self, store: S) -> None:
        raise NotImplementedError

    def test_store_supports_listing(self, store: S) -> None:
        raise NotImplementedError

    @pytest.mark.parametrize("key", ["c/0", "foo/c/0.0", "foo/0/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    @pytest.mark.parametrize("byte_range", (None, (0, None), (1, None), (1, 2), (None, 1)))
    async def test_set_get_bytes_roundtrip(
        self, store: S, key: str, data: bytes, byte_range: None | tuple[int | None, int | None]
    ) -> None:
        await store.set(key, Buffer.from_bytes(data))
        start, length = _normalize_byte_range(data, byte_range)
        expected = data[start : start + length]
        assert await store.get(key, byte_range=byte_range) == expected

    @pytest.mark.parametrize("key", ["foo/c/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    async def test_get_partial_values(self, store: S, key: str, data: bytes) -> None:
        # put all of the data
        await store.set(key, Buffer.from_bytes(data))
        # read back just part of it
        vals = await store.get_partial_values([(key, (0, 2))])
        assert vals == [data[0:2]]

        # read back multiple parts of it at once
        vals = await store.get_partial_values([(key, (0, 2)), (key, (2, 4))])
        assert vals == [data[0:2], data[2:4]]

    async def test_exists(self, store: S) -> None:
        assert not await store.exists("foo")
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        assert await store.exists("foo/zarr.json")

    async def test_delete(self, store: S) -> None:
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        assert await store.exists("foo/zarr.json")
        await store.delete("foo/zarr.json")
        assert not await store.exists("foo/zarr.json")

    async def test_list(self, store: S) -> None:
        assert [k async for k in store.list()] == []
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        keys = [k async for k in store.list()]
        assert keys == ["foo/zarr.json"], keys

        expected = ["foo/zarr.json"]
        for i in range(10):
            key = f"foo/c/{i}"
            expected.append(key)
            await store.set(
                f"foo/c/{i}", Buffer.from_bytes(i.to_bytes(length=3, byteorder="little"))
            )

    async def test_list_prefix(self, store: S) -> None:
        # TODO: we currently don't use list_prefix anywhere
        raise NotImplementedError

    async def test_list_dir(self, store: S) -> None:
        assert [k async for k in store.list_dir("")] == []
        assert [k async for k in store.list_dir("foo")] == []
        await store.set("foo/zarr.json", Buffer.from_bytes(b"bar"))
        await store.set("foo/c/1", Buffer.from_bytes(b"\x01"))

        keys = [k async for k in store.list_dir("foo")]
        assert set(keys) == set(["zarr.json", "c"]), keys

        keys = [k async for k in store.list_dir("foo/")]
        assert set(keys) == set(["zarr.json", "c"]), keys
