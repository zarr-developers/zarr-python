import pytest

from zarr.abc.store import Store


class StoreTests:
    store_cls: type[Store]

    @pytest.fixture(scope="function")
    def store(self) -> Store:
        return self.store_cls()

    def test_store_type(self, store: Store) -> None:
        assert isinstance(store, Store)
        assert isinstance(store, self.store_cls)

    def test_store_repr(self, store: Store) -> None:
        assert repr(store)

    def test_store_capabilities(self, store: Store) -> None:
        assert store.supports_writes
        assert store.supports_partial_writes
        assert store.supports_listing

    @pytest.mark.parametrize("key", ["c/0", "foo/c/0.0", "foo/0/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    async def test_set_get_bytes_roundtrip(self, store: Store, key: str, data: bytes) -> None:
        await store.set(key, data)
        assert await store.get(key) == data

    @pytest.mark.parametrize("key", ["foo/c/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    async def test_get_partial_values(self, store: Store, key: str, data: bytes) -> None:
        # put all of the data
        await store.set(key, data)
        # read back just part of it
        vals = await store.get_partial_values([(key, (0, 2))])
        assert vals == [data[0:2]]

        # read back multiple parts of it at once
        vals = await store.get_partial_values([(key, (0, 2)), (key, (2, 4))])
        assert vals == [data[0:2], data[2:4]]

    async def test_exists(self, store: Store) -> None:
        assert not await store.exists("foo")
        await store.set("foo/zarr.json", b"bar")
        assert await store.exists("foo/zarr.json")

    async def test_delete(self, store: Store) -> None:
        await store.set("foo/zarr.json", b"bar")
        assert await store.exists("foo/zarr.json")
        await store.delete("foo/zarr.json")
        assert not await store.exists("foo/zarr.json")

    async def test_list(self, store: Store) -> None:
        assert [k async for k in store.list()] == []
        await store.set("foo/zarr.json", b"bar")
        keys = [k async for k in store.list()]
        assert keys == ["foo/zarr.json"], keys

        expected = ["foo/zarr.json"]
        for i in range(10):
            key = f"foo/c/{i}"
            expected.append(key)
            await store.set(f"foo/c/{i}", i.to_bytes(length=3, byteorder="little"))

    async def test_list_prefix(self, store: Store) -> None:
        # TODO: we currently don't use list_prefix anywhere
        pass

    async def test_list_dir(self, store: Store) -> None:
        assert [k async for k in store.list_dir("")] == []
        assert [k async for k in store.list_dir("foo")] == []
        await store.set("foo/zarr.json", b"bar")
        await store.set("foo/c/1", b"\x01")

        keys = [k async for k in store.list_dir("foo")]
        assert keys == ["zarr.json", "c"], keys

        keys = [k async for k in store.list_dir("foo/")]
        assert keys == ["zarr.json", "c"], keys
