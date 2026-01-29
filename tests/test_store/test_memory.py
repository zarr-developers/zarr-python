from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

import zarr
from zarr.core.buffer import Buffer, cpu, gpu
from zarr.core.sync import sync
from zarr.errors import ZarrUserWarning
from zarr.storage import GpuMemoryStore, ManagedMemoryStore, MemoryStore
from zarr.testing.store import StoreTests
from zarr.testing.utils import gpu_test

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import ZarrFormat


# TODO: work out where this warning is coming from and fix it
@pytest.mark.filterwarnings(
    re.escape("ignore:coroutine 'ClientCreatorContext.__aexit__' was never awaited")
)
class TestMemoryStore(StoreTests[MemoryStore, cpu.Buffer]):
    store_cls = MemoryStore
    buffer_cls = cpu.Buffer

    async def set(self, store: MemoryStore, key: str, value: Buffer) -> None:
        store._store_dict[key] = value

    async def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(params=[None, True])
    def store_kwargs(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        kwargs: dict[str, Any]
        if request.param is True:
            kwargs = {"store_dict": {}}
        else:
            kwargs = {"store_dict": None}
        return kwargs

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> MemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: MemoryStore) -> None:
        assert str(store) == f"memory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: MemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: MemoryStore) -> None:
        assert store.supports_listing

    async def test_list_prefix(self, store: MemoryStore) -> None:
        assert True

    @pytest.mark.parametrize("dtype", ["uint8", "float32", "int64"])
    @pytest.mark.parametrize("zarr_format", [2, 3])
    async def test_deterministic_size(
        self, store: MemoryStore, dtype: npt.DTypeLike, zarr_format: ZarrFormat
    ) -> None:
        a = zarr.empty(
            store=store,
            shape=(3,),
            chunks=(1000,),
            dtype=dtype,
            zarr_format=zarr_format,
            overwrite=True,
        )
        a[...] = 1
        a.resize((1000,))

        np.testing.assert_array_equal(a[:3], 1)
        np.testing.assert_array_equal(a[3:], 0)

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    async def test_get_bytes_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_bytes works with prototype=None."""
        data = b"hello world"
        key = "test_key"
        await self.set(store, key, self.buffer_cls.from_bytes(data))

        result = await store._get_bytes(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    def test_get_bytes_sync_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_bytes_sync works with prototype=None."""
        data = b"hello world"
        key = "test_key"
        sync(self.set(store, key, self.buffer_cls.from_bytes(data)))

        result = store._get_bytes_sync(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    async def test_get_json_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_json works with prototype=None."""
        data = {"foo": "bar", "number": 42}
        key = "test.json"
        await self.set(store, key, self.buffer_cls.from_bytes(json.dumps(data).encode()))

        result = await store._get_json(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    def test_get_json_sync_with_prototype_none(
        self, store: MemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_json_sync works with prototype=None."""
        data = {"foo": "bar", "number": 42}
        key = "test.json"
        sync(self.set(store, key, self.buffer_cls.from_bytes(json.dumps(data).encode())))

        result = store._get_json_sync(key, prototype=buffer_cls)
        assert result == data


# TODO: fix this warning
@pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning")
@gpu_test
class TestGpuMemoryStore(StoreTests[GpuMemoryStore, gpu.Buffer]):
    store_cls = GpuMemoryStore
    buffer_cls = gpu.Buffer

    async def set(self, store: GpuMemoryStore, key: str, value: gpu.Buffer) -> None:  # type: ignore[override]
        store._store_dict[key] = value

    async def get(self, store: MemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture(params=[None, True])
    def store_kwargs(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        kwargs: dict[str, Any]
        if request.param is True:
            kwargs = {"store_dict": {}}
        else:
            kwargs = {"store_dict": None}
        return kwargs

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> GpuMemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: GpuMemoryStore) -> None:
        assert str(store) == f"gpumemory://{id(store._store_dict)}"

    def test_store_supports_writes(self, store: GpuMemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: GpuMemoryStore) -> None:
        assert store.supports_listing

    async def test_list_prefix(self, store: GpuMemoryStore) -> None:
        assert True

    def test_dict_reference(self, store: GpuMemoryStore) -> None:
        store_dict: dict[str, Any] = {}
        result = GpuMemoryStore(store_dict=store_dict)
        assert result._store_dict is store_dict

    def test_from_dict(self) -> None:
        d = {
            "a": gpu.Buffer.from_bytes(b"aaaa"),
            "b": cpu.Buffer.from_bytes(b"bbbb"),
        }
        msg = "Creating a zarr.buffer.gpu.Buffer with an array that does not support the __cuda_array_interface__ for zero-copy transfers, falling back to slow copy based path"
        with pytest.warns(ZarrUserWarning, match=msg):
            result = GpuMemoryStore.from_dict(d)
        for v in result._store_dict.values():
            assert type(v) is gpu.Buffer


class TestManagedMemoryStore(StoreTests[ManagedMemoryStore, cpu.Buffer]):
    store_cls = ManagedMemoryStore
    buffer_cls = cpu.Buffer

    async def set(self, store: ManagedMemoryStore, key: str, value: Buffer) -> None:
        store._store_dict[key] = value

    async def get(self, store: ManagedMemoryStore, key: str) -> Buffer:
        return store._store_dict[key]

    @pytest.fixture
    def store_kwargs(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        # Use a unique name per test to avoid sharing state between tests
        # but ensure the name is deterministic for equality tests
        # Replace '/' with '-' since store names cannot contain '/'
        sanitized_name = request.node.name.replace("/", "-")
        return {"name": f"test-{sanitized_name}"}

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> ManagedMemoryStore:
        return self.store_cls(**store_kwargs)

    def test_store_repr(self, store: ManagedMemoryStore) -> None:
        assert str(store) == f"memory://{store.name}"

    async def test_serializable_store(self, store: ManagedMemoryStore) -> None:
        """
        Test pickling semantics for ManagedMemoryStore.

        When pickled and unpickled within the same process (where the original
        store still exists in the registry), the unpickled store reconnects to
        the same backing dict.
        """
        import pickle

        # Add some data to the store
        await store.set("test-key", self.buffer_cls.from_bytes(b"test-value"))

        # Pickle and unpickle the store
        pickled = pickle.dumps(store)
        store2 = pickle.loads(pickled)

        # The unpickled store should reconnect to the same backing dict
        assert store2._store_dict is store._store_dict
        assert store2.name == store.name
        assert store2.path == store.path
        assert store2.read_only == store.read_only

        # The data should be accessible
        result = await store2.get("test-key")
        assert result is not None
        assert result.to_bytes() == b"test-value"

    async def test_pickle_with_path(self) -> None:
        """Test that path is preserved through pickle round-trip."""
        import pickle

        store = ManagedMemoryStore(name="pickle-path-test", path="some/path")
        await store.set("key", self.buffer_cls.from_bytes(b"value"))

        pickled = pickle.dumps(store)
        store2 = pickle.loads(pickled)

        assert store2.path == "some/path"
        assert store2._store_dict is store._store_dict

        # Check that operations use the path correctly
        result = await store2.get("key")
        assert result is not None
        assert result.to_bytes() == b"value"

    def test_pickle_after_gc(self) -> None:
        """
        Test that unpickling after the original store is garbage collected
        creates a new empty store with the same name (in the same process).
        """
        import gc
        import pickle

        # Create a store with a unique name and pickle it
        store = ManagedMemoryStore(name="gc-pickle-test")
        store._store_dict["key"] = self.buffer_cls.from_bytes(b"value")
        pickled = pickle.dumps(store)

        # Delete the store and garbage collect
        del store
        gc.collect()

        # Unpickling should create a new store with an empty dict
        store2 = pickle.loads(pickled)
        assert store2.name == "gc-pickle-test"
        # The dict is empty because the original was garbage collected
        assert len(store2._store_dict) == 0

    async def test_cross_process_detection(self) -> None:
        """
        Test that using a ManagedMemoryStore in a different process raises an error.

        This prevents silent data loss when a store is pickled and unpickled
        in a different process (e.g., with multiprocessing).
        """
        import pickle

        store = ManagedMemoryStore(name="cross-process-test")
        await store.set("key", self.buffer_cls.from_bytes(b"value"))

        # Simulate unpickling in a different process by manipulating _created_pid
        pickled = pickle.dumps(store)
        store2 = pickle.loads(pickled)

        # Manually change the created_pid to simulate a different process
        store2._created_pid = store2._created_pid + 1

        # All operations should raise RuntimeError
        with pytest.raises(RuntimeError, match="was created in process"):
            await store2.get("key")

        with pytest.raises(RuntimeError, match="was created in process"):
            await store2.set("key", self.buffer_cls.from_bytes(b"value"))

        with pytest.raises(RuntimeError, match="was created in process"):
            await store2.exists("key")

        with pytest.raises(RuntimeError, match="was created in process"):
            await store2.delete("key")

        with pytest.raises(RuntimeError, match="was created in process"):
            [k async for k in store2.list()]

        with pytest.raises(RuntimeError, match="was created in process"):
            [k async for k in store2.list_prefix("")]

        with pytest.raises(RuntimeError, match="was created in process"):
            [k async for k in store2.list_dir("")]

    def test_store_supports_writes(self, store: ManagedMemoryStore) -> None:
        assert store.supports_writes

    def test_store_supports_listing(self, store: ManagedMemoryStore) -> None:
        assert store.supports_listing

    async def test_list_prefix(self, store: MemoryStore) -> None:
        assert True

    @pytest.mark.parametrize("dtype", ["uint8", "float32", "int64"])
    @pytest.mark.parametrize("zarr_format", [2, 3])
    async def test_deterministic_size(
        self, store: MemoryStore, dtype: npt.DTypeLike, zarr_format: ZarrFormat
    ) -> None:
        a = zarr.empty(
            store=store,
            shape=(3,),
            chunks=(1000,),
            dtype=dtype,
            zarr_format=zarr_format,
            overwrite=True,
        )
        a[...] = 1
        a.resize((1000,))

        np.testing.assert_array_equal(a[:3], 1)
        np.testing.assert_array_equal(a[3:], 0)

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    async def test_get_bytes_with_prototype_none(
        self, store: ManagedMemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_bytes works with prototype=None."""
        data = b"hello world"
        key = "test_key"
        await self.set(store, key, self.buffer_cls.from_bytes(data))

        result = await store._get_bytes(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    def test_get_bytes_sync_with_prototype_none(
        self, store: ManagedMemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_bytes_sync works with prototype=None."""
        data = b"hello world"
        key = "test_key"
        sync(self.set(store, key, self.buffer_cls.from_bytes(data)))

        result = store._get_bytes_sync(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    async def test_get_json_with_prototype_none(
        self, store: ManagedMemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_json works with prototype=None."""
        data = {"foo": "bar", "number": 42}
        key = "test.json"
        await self.set(store, key, self.buffer_cls.from_bytes(json.dumps(data).encode()))

        result = await store._get_json(key, prototype=buffer_cls)
        assert result == data

    @pytest.mark.parametrize("buffer_cls", [None, cpu.buffer_prototype])
    def test_get_json_sync_with_prototype_none(
        self, store: ManagedMemoryStore, buffer_cls: None | BufferPrototype
    ) -> None:
        """Test that get_json_sync works with prototype=None."""
        data = {"foo": "bar", "number": 42}
        key = "test.json"
        sync(self.set(store, key, self.buffer_cls.from_bytes(json.dumps(data).encode())))

        result = store._get_json_sync(key, prototype=buffer_cls)
        assert result == data

    def test_from_url(self, store: ManagedMemoryStore) -> None:
        """Test that from_url creates a store sharing the same dict."""
        url = str(store)
        store2 = ManagedMemoryStore.from_url(url)
        assert store2._store_dict is store._store_dict

    def test_from_url_with_path(self, store: ManagedMemoryStore) -> None:
        """Test that from_url extracts path component from URL."""
        url = str(store) + "/some/path"
        store2 = ManagedMemoryStore.from_url(url)
        assert store2._store_dict is store._store_dict
        assert store2.path == "some/path"
        assert str(store2) == url

    def test_from_url_invalid(self) -> None:
        """Test that from_url raises ValueError for non-existent store."""
        with pytest.raises(ValueError, match="Memory store not found"):
            ManagedMemoryStore.from_url("memory://nonexistent-store")

    def test_from_url_not_memory_scheme(self) -> None:
        """Test that from_url raises ValueError for non-memory URLs."""
        with pytest.raises(ValueError, match="Memory store not found"):
            ManagedMemoryStore.from_url("file:///tmp/test")

    def test_named_store(self) -> None:
        """Test that stores can be created with explicit names."""
        store = ManagedMemoryStore(name="my-test-store")
        assert store.name == "my-test-store"
        assert str(store) == "memory://my-test-store"

    def test_named_store_shares_dict(self) -> None:
        """Test that creating a store with the same name shares the dict."""
        store1 = ManagedMemoryStore(name="shared-store")
        store2 = ManagedMemoryStore(name="shared-store")
        assert store1._store_dict is store2._store_dict
        assert store1.name == store2.name

    def test_auto_generated_name(self) -> None:
        """Test that stores get auto-generated names when none provided."""
        store = ManagedMemoryStore()
        assert store.name is not None
        assert str(store) == f"memory://{store.name}"

    def test_with_read_only_shares_dict(self, store: ManagedMemoryStore) -> None:
        """Test that with_read_only creates a store sharing the same dict."""
        store2 = store.with_read_only(True)
        assert store2._store_dict is store._store_dict
        assert store2.read_only is True
        assert store.read_only is False

    def test_with_read_only_preserves_path(self) -> None:
        """Test that with_read_only preserves the path."""
        store = ManagedMemoryStore(name="path-test", path="some/path")
        store2 = store.with_read_only(True)
        assert store2.path == "some/path"
        assert store2._store_dict is store._store_dict

    async def test_path_prefix_operations(self) -> None:
        """Test that store operations use the path prefix correctly."""
        store = ManagedMemoryStore(name="prefix-test")
        store_with_path = ManagedMemoryStore.from_url("memory://prefix-test/subdir")

        # Write via store_with_path
        await store_with_path.set("key", self.buffer_cls.from_bytes(b"value"))

        # The key should be stored with the prefix in the underlying dict
        assert "subdir/key" in store._store_dict
        assert "key" not in store._store_dict

        # Read via store_with_path should work
        result = await store_with_path.get("key")
        assert result is not None
        assert result.to_bytes() == b"value"

        # Read via store without path should use full key
        result2 = await store.get("subdir/key")
        assert result2 is not None
        assert result2.to_bytes() == b"value"

    async def test_path_list_operations(self) -> None:
        """Test that list operations filter by path prefix."""
        store = ManagedMemoryStore(name="list-test")

        # Set up some keys at different paths
        await store.set("a/key1", self.buffer_cls.from_bytes(b"v1"))
        await store.set("a/key2", self.buffer_cls.from_bytes(b"v2"))
        await store.set("b/key3", self.buffer_cls.from_bytes(b"v3"))

        # Create a store with path "a"
        store_a = ManagedMemoryStore.from_url("memory://list-test/a")

        # list() should only return keys under "a", without the "a/" prefix
        keys = [k async for k in store_a.list()]
        assert sorted(keys) == ["key1", "key2"]

    async def test_path_exists(self) -> None:
        """Test that exists() uses the path prefix."""
        store = ManagedMemoryStore(name="exists-test")
        await store.set("prefix/key", self.buffer_cls.from_bytes(b"value"))

        store_with_path = ManagedMemoryStore.from_url("memory://exists-test/prefix")
        assert await store_with_path.exists("key")
        assert not await store_with_path.exists("prefix/key")

    def test_path_normalization(self) -> None:
        """Test that paths are normalized."""
        store1 = ManagedMemoryStore(name="norm-test", path="a/b/")
        store2 = ManagedMemoryStore(name="norm-test", path="/a/b")
        store3 = ManagedMemoryStore(name="norm-test", path="a//b")
        assert store1.path == "a/b"
        assert store2.path == "a/b"
        assert store3.path == "a/b"

    def test_name_cannot_contain_slash(self) -> None:
        """Test that store names cannot contain '/'."""
        with pytest.raises(ValueError, match="cannot contain '/'"):
            ManagedMemoryStore(name="foo/bar")

    def test_garbage_collection(self) -> None:
        """Test that the dict is garbage collected when no stores reference it."""
        import gc

        store = ManagedMemoryStore()
        url = str(store)

        # URL should resolve while store exists
        store2 = ManagedMemoryStore.from_url(url)
        assert store2._store_dict is store._store_dict

        # Delete both stores
        del store
        del store2
        gc.collect()

        # URL should no longer resolve
        with pytest.raises(ValueError, match="garbage collected"):
            ManagedMemoryStore.from_url(url)
