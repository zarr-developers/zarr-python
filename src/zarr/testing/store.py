import pickle
from typing import Any, Generic, TypeVar, cast

import pytest

from zarr.abc.store import AccessMode, Store
from zarr.core.buffer import Buffer, default_buffer_prototype
from zarr.core.common import AccessModeLiteral
from zarr.core.sync import _collect_aiterator
from zarr.storage._utils import _normalize_interval_index
from zarr.testing.utils import assert_bytes_equal

__all__ = ["StoreTests"]


S = TypeVar("S", bound=Store)
B = TypeVar("B", bound=Buffer)


class StoreTests(Generic[S, B]):
    store_cls: type[S]
    buffer_cls: type[B]

    async def set(self, store: S, key: str, value: Buffer) -> None:
        """
        Insert a value into a storage backend, with a specific key.
        This should not not use any store methods. Bypassing the store methods allows them to be
        tested.
        """
        raise NotImplementedError

    async def get(self, store: S, key: str) -> Buffer:
        """
        Retrieve a value from a storage backend, by key.
        This should not not use any store methods. Bypassing the store methods allows them to be
        tested.
        """

        raise NotImplementedError

    @pytest.fixture
    def store_kwargs(self) -> dict[str, Any]:
        return {"mode": "r+"}

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> Store:
        return await self.store_cls.open(**store_kwargs)

    def test_store_type(self, store: S) -> None:
        assert isinstance(store, Store)
        assert isinstance(store, self.store_cls)

    def test_store_eq(self, store: S, store_kwargs: dict[str, Any]) -> None:
        # check self equality
        assert store == store

        # check store equality with same inputs
        # asserting this is important for being able to compare (de)serialized stores
        store2 = self.store_cls(**store_kwargs)
        assert store == store2

    def test_serializable_store(self, store: S) -> None:
        foo = pickle.dumps(store)
        assert pickle.loads(foo) == store

    def test_store_mode(self, store: S, store_kwargs: dict[str, Any]) -> None:
        assert store.mode == AccessMode.from_literal("r+")
        assert not store.mode.readonly

        with pytest.raises(AttributeError):
            store.mode = AccessMode.from_literal("w")  # type: ignore[misc]

    @pytest.mark.parametrize("mode", ["r", "r+", "a", "w", "w-"])
    async def test_store_open_mode(
        self, store_kwargs: dict[str, Any], mode: AccessModeLiteral
    ) -> None:
        store_kwargs["mode"] = mode
        store = await self.store_cls.open(**store_kwargs)
        assert store._is_open
        assert store.mode == AccessMode.from_literal(mode)

    async def test_not_writable_store_raises(self, store_kwargs: dict[str, Any]) -> None:
        kwargs = {**store_kwargs, "mode": "r"}
        store = await self.store_cls.open(**kwargs)
        assert store.mode == AccessMode.from_literal("r")
        assert store.mode.readonly

        # set
        with pytest.raises(ValueError):
            await store.set("foo", self.buffer_cls.from_bytes(b"bar"))

        # delete
        with pytest.raises(ValueError):
            await store.delete("foo")

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
    @pytest.mark.parametrize("byte_range", [None, (0, None), (1, None), (1, 2), (None, 1)])
    async def test_get(
        self, store: S, key: str, data: bytes, byte_range: None | tuple[int | None, int | None]
    ) -> None:
        """
        Ensure that data can be read from the store using the store.get method.
        """
        data_buf = self.buffer_cls.from_bytes(data)
        await self.set(store, key, data_buf)
        observed = await store.get(key, prototype=default_buffer_prototype(), byte_range=byte_range)
        start, length = _normalize_interval_index(data_buf, interval=byte_range)
        expected = data_buf[start : start + length]
        assert_bytes_equal(observed, expected)

    async def test_get_many(self, store: S) -> None:
        """
        Ensure that multiple keys can be retrieved at once with the _get_many method.
        """
        keys = tuple(map(str, range(10)))
        values = tuple(f"{k}".encode() for k in keys)
        for k, v in zip(keys, values, strict=False):
            await self.set(store, k, self.buffer_cls.from_bytes(v))
        observed_buffers = await _collect_aiterator(
            store._get_many(
                zip(
                    keys,
                    (default_buffer_prototype(),) * len(keys),
                    (None,) * len(keys),
                    strict=False,
                )
            )
        )
        observed_kvs = sorted(((k, b.to_bytes()) for k, b in observed_buffers))  # type: ignore[union-attr]
        expected_kvs = sorted(((k, b) for k, b in zip(keys, values, strict=False)))
        assert observed_kvs == expected_kvs

    @pytest.mark.parametrize("key", ["zarr.json", "c/0", "foo/c/0.0", "foo/0/0"])
    @pytest.mark.parametrize("data", [b"\x01\x02\x03\x04", b""])
    async def test_set(self, store: S, key: str, data: bytes) -> None:
        """
        Ensure that data can be written to the store using the store.set method.
        """
        assert not store.mode.readonly
        data_buf = self.buffer_cls.from_bytes(data)
        await store.set(key, data_buf)
        observed = await self.get(store, key)
        assert_bytes_equal(observed, data_buf)

    async def test_set_many(self, store: S) -> None:
        """
        Test that a dict of key : value pairs can be inserted into the store via the
        `_set_many` method.
        """
        keys = ["zarr.json", "c/0", "foo/c/0.0", "foo/0/0"]
        data_buf = [self.buffer_cls.from_bytes(k.encode()) for k in keys]
        store_dict = dict(zip(keys, data_buf, strict=True))
        await store._set_many(store_dict.items())
        for k, v in store_dict.items():
            assert (await self.get(store, k)).to_bytes() == v.to_bytes()

    @pytest.mark.parametrize(
        "key_ranges",
        [
            [],
            [("zarr.json", (0, 1))],
            [("c/0", (0, 1)), ("zarr.json", (0, None))],
            [("c/0/0", (0, 1)), ("c/0/1", (None, 2)), ("c/0/2", (0, 3))],
        ],
    )
    async def test_get_partial_values(
        self, store: S, key_ranges: list[tuple[str, tuple[int | None, int | None]]]
    ) -> None:
        # put all of the data
        for key, _ in key_ranges:
            await self.set(store, key, self.buffer_cls.from_bytes(bytes(key, encoding="utf-8")))

        # read back just part of it
        observed_maybe = await store.get_partial_values(
            prototype=default_buffer_prototype(), key_ranges=key_ranges
        )

        observed: list[Buffer] = []
        expected: list[Buffer] = []

        for obs in observed_maybe:
            assert obs is not None
            observed.append(obs)

        for idx in range(len(observed)):
            key, byte_range = key_ranges[idx]
            result = await store.get(
                key, prototype=default_buffer_prototype(), byte_range=byte_range
            )
            assert result is not None
            expected.append(result)

        assert all(
            obs.to_bytes() == exp.to_bytes() for obs, exp in zip(observed, expected, strict=True)
        )

    async def test_exists(self, store: S) -> None:
        assert not await store.exists("foo")
        await store.set("foo/zarr.json", self.buffer_cls.from_bytes(b"bar"))
        assert await store.exists("foo/zarr.json")

    async def test_delete(self, store: S) -> None:
        await store.set("foo/zarr.json", self.buffer_cls.from_bytes(b"bar"))
        assert await store.exists("foo/zarr.json")
        await store.delete("foo/zarr.json")
        assert not await store.exists("foo/zarr.json")

    async def test_empty(self, store: S) -> None:
        assert await store.empty()
        await self.set(
            store, "key", self.buffer_cls.from_bytes(bytes("something", encoding="utf-8"))
        )
        assert not await store.empty()

    async def test_clear(self, store: S) -> None:
        await self.set(
            store, "key", self.buffer_cls.from_bytes(bytes("something", encoding="utf-8"))
        )
        await store.clear()
        assert await store.empty()

    async def test_list(self, store: S) -> None:
        assert await _collect_aiterator(store.list()) == ()
        prefix = "foo"
        data = self.buffer_cls.from_bytes(b"")
        store_dict = {
            prefix + "/zarr.json": data,
            **{prefix + f"/c/{idx}": data for idx in range(10)},
        }
        await store._set_many(store_dict.items())
        expected_sorted = sorted(store_dict.keys())
        observed = await _collect_aiterator(store.list())
        observed_sorted = sorted(observed)
        assert observed_sorted == expected_sorted

    async def test_list_prefix(self, store: S) -> None:
        """
        Test that the `list_prefix` method works as intended. Given a prefix, it should return
        all the keys in storage that start with this prefix. Keys should be returned with the shared
        prefix removed.
        """
        prefixes = ("", "a/", "a/b/", "a/b/c/")
        data = self.buffer_cls.from_bytes(b"")
        fname = "zarr.json"
        store_dict = {p + fname: data for p in prefixes}

        await store._set_many(store_dict.items())

        for prefix in prefixes:
            observed = tuple(sorted(await _collect_aiterator(store.list_prefix(prefix))))
            expected: tuple[str, ...] = ()
            for key in store_dict:
                if key.startswith(prefix):
                    expected += (key.removeprefix(prefix),)
            expected = tuple(sorted(expected))
            assert observed == expected

    async def test_list_dir(self, store: S) -> None:
        root = "foo"
        store_dict = {
            root + "/zarr.json": self.buffer_cls.from_bytes(b"bar"),
            root + "/c/1": self.buffer_cls.from_bytes(b"\x01"),
        }

        assert await _collect_aiterator(store.list_dir("")) == ()
        assert await _collect_aiterator(store.list_dir(root)) == ()

        await store._set_many(store_dict.items())

        keys_observed = await _collect_aiterator(store.list_dir(root))
        keys_expected = {k.removeprefix(root + "/").split("/")[0] for k in store_dict}

        assert sorted(keys_observed) == sorted(keys_expected)

        keys_observed = await _collect_aiterator(store.list_dir(root + "/"))
        assert sorted(keys_expected) == sorted(keys_observed)

    async def test_with_mode(self, store: S) -> None:
        data = b"0000"
        await self.set(store, "key", self.buffer_cls.from_bytes(data))
        assert (await self.get(store, "key")).to_bytes() == data

        for mode in ["r", "a"]:
            mode = cast(AccessModeLiteral, mode)
            clone = store.with_mode(mode)
            # await store.close()
            await clone._ensure_open()
            assert clone.mode == AccessMode.from_literal(mode)
            assert isinstance(clone, type(store))

            # earlier writes are visible
            result = await clone.get("key", default_buffer_prototype())
            assert result is not None
            assert result.to_bytes() == data

            # writes to original after with_mode is visible
            await self.set(store, "key-2", self.buffer_cls.from_bytes(data))
            result = await clone.get("key-2", default_buffer_prototype())
            assert result is not None
            assert result.to_bytes() == data

            if mode == "a":
                # writes to clone is visible in the original
                await clone.set("key-3", self.buffer_cls.from_bytes(data))
                result = await clone.get("key-3", default_buffer_prototype())
                assert result is not None
                assert result.to_bytes() == data

            else:
                with pytest.raises(ValueError, match="store mode"):
                    await clone.set("key-3", self.buffer_cls.from_bytes(data))

    async def test_set_if_not_exists(self, store: S) -> None:
        key = "k"
        data_buf = self.buffer_cls.from_bytes(b"0000")
        await self.set(store, key, data_buf)

        new = self.buffer_cls.from_bytes(b"1111")
        await store.set_if_not_exists("k", new)  # no error

        result = await store.get(key, default_buffer_prototype())
        assert result == data_buf

        await store.set_if_not_exists("k2", new)  # no error

        result = await store.get("k2", default_buffer_prototype())
        assert result == new
