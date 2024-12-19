from __future__ import annotations

import asyncio
import pickle
from typing import TYPE_CHECKING, Generic, TypeVar

from zarr.storage.wrapper import WrapperStore

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.store import ByteRangeRequest
    from zarr.core.buffer.core import BufferPrototype

import pytest

from zarr.abc.store import ByteRangeRequest, Store
from zarr.core.buffer import Buffer, default_buffer_prototype
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
        return {"read_only": False}

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

    def test_store_read_only(self, store: S) -> None:
        assert not store.read_only

        with pytest.raises(AttributeError):
            store.read_only = False  # type: ignore[misc]

    @pytest.mark.parametrize("read_only", [True, False])
    async def test_store_open_read_only(
        self, store_kwargs: dict[str, Any], read_only: bool
    ) -> None:
        store_kwargs["read_only"] = read_only
        store = await self.store_cls.open(**store_kwargs)
        assert store._is_open
        assert store.read_only == read_only

    async def test_read_only_store_raises(self, store_kwargs: dict[str, Any]) -> None:
        kwargs = {**store_kwargs, "read_only": True}
        store = await self.store_cls.open(**kwargs)
        assert store.read_only

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
        self, store: S, key: str, data: bytes, byte_range: tuple[int | None, int | None] | None
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
        assert not store.read_only
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
        if not store.supports_deletes:
            pytest.skip("store does not support deletes")
        await store.set("foo/zarr.json", self.buffer_cls.from_bytes(b"bar"))
        assert await store.exists("foo/zarr.json")
        await store.delete("foo/zarr.json")
        assert not await store.exists("foo/zarr.json")

    async def test_delete_dir(self, store: S) -> None:
        if not store.supports_deletes:
            pytest.skip("store does not support deletes")
        await store.set("zarr.json", self.buffer_cls.from_bytes(b"root"))
        await store.set("foo-bar/zarr.json", self.buffer_cls.from_bytes(b"root"))
        await store.set("foo/zarr.json", self.buffer_cls.from_bytes(b"bar"))
        await store.set("foo/c/0", self.buffer_cls.from_bytes(b"chunk"))
        await store.delete_dir("foo")
        assert await store.exists("zarr.json")
        assert await store.exists("foo-bar/zarr.json")
        assert not await store.exists("foo/zarr.json")
        assert not await store.exists("foo/c/0")

    async def test_is_empty(self, store: S) -> None:
        assert await store.is_empty("")
        await self.set(
            store, "foo/bar", self.buffer_cls.from_bytes(bytes("something", encoding="utf-8"))
        )
        assert not await store.is_empty("")
        assert await store.is_empty("fo")
        assert not await store.is_empty("foo/")
        assert not await store.is_empty("foo")
        assert await store.is_empty("spam/")

    async def test_clear(self, store: S) -> None:
        await self.set(
            store, "key", self.buffer_cls.from_bytes(bytes("something", encoding="utf-8"))
        )
        await store.clear()
        assert await store.is_empty("")

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
        all the keys in storage that start with this prefix.
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
                    expected += (key,)
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


class LatencyStore(WrapperStore[Store]):
    """
    A wrapper class that takes any store class in its constructor and
    adds latency to the `set` and `get` methods. This can be used for
    performance testing.
    """

    get_latency: float
    set_latency: float

    def __init__(self, cls: Store, *, get_latency: float = 0, set_latency: float = 0) -> None:
        self.get_latency = float(get_latency)
        self.set_latency = float(set_latency)
        self._store = cls

    async def set(self, key: str, value: Buffer) -> None:
        """
        Add latency to the ``set`` method.

        Calls ``asyncio.sleep(self.set_latency)`` before invoking the wrapped ``set`` method.

        Parameters
        ----------
        key : str
            The key to set
        value : Buffer
            The value to set

        Returns
        -------
        None
        """
        await asyncio.sleep(self.set_latency)
        await self._store.set(key, value)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRangeRequest | None = None
    ) -> Buffer | None:
        """
        Add latency to the ``get`` method.

        Calls ``asyncio.sleep(self.get_latency)`` before invoking the wrapped ``get`` method.

        Parameters
        ----------
        key : str
            The key to get
        prototype : BufferPrototype
            The BufferPrototype to use.
        byte_range : ByteRangeRequest, optional
            An optional byte range.

        Returns
        -------
        buffer : Buffer or None
        """
        await asyncio.sleep(self.get_latency)
        return await self._store.get(key, prototype=prototype, byte_range=byte_range)
