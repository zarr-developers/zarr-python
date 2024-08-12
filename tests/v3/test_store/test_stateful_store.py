# Stateful tests for arbitrary Zarr stores.

import asyncio

import hypothesis.strategies as st
from hypothesis import note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

from zarr.abc.store import Store as StoreABC
from zarr.buffer import Buffer, default_buffer_prototype
from zarr.store import LocalStore, MemoryStore, RemoteStore

# from strategies_store import StoreStatefulStrategies, key_ranges
from zarr.testing.strategies import key_ranges, paths

# zarr spec: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html


class SyncStoreWrapper():
    def __init__(self, store):
        """Store to hold async functions that map to StoreABC abstract methods"""
        self.store = store
        # Unfortunately, hypothesis' stateful testing infra does not support asyncio

    # So we redefine sync versions of the Store API.
    # https://github.com/HypothesisWorks/hypothesis/issues/3712#issuecomment-1668999041
    def set(self, key, data_buffer):  # buffer is value
        return asyncio.run(self.store.set(key, data_buffer))

    async def list(self):
        paths = [path async for path in self.store.list()]
        # note(f'(store set) store {paths=}, type {type(paths)=}, {len(paths)=}')
        return paths

    async def get(
        self,
        key,
    ):
        obs = await self.store.get(key, prototype=default_buffer_prototype())
        return obs

    async def get_partial_values(self, key_ranges):
        obs_maybe = await self.store.get_partial_values(
            prototype=default_buffer_prototype(), key_ranges=key_ranges
        )
        return obs_maybe

    async def delete(self, path):  # path is key
        await self.store.delete(path)

    async def empty(self):
        await self.store.empty()

    async def clear(self):
        await self.store.clear()

    async def exists(self, key):
        raise NotImplementedError

    async def list_dir(self, prefix):
        raise NotImplementedError

    async def list_prefix(self, prefix: str):
        raise NotImplementedError

    async def set_partial_values(self, key_start_values):
        raise NotImplementedError

    async def supports_listing(self):
        raise NotImplementedError

    async def supports_partial_writes(self):
        raise NotImplementedError

    async def supports_writes(self):
        raise NotImplementedError


class ZarrStoreStateMachine(RuleBasedStateMachine):
    def __init__(self):  # look into using run_machine_as_test()
        super().__init__()
        self.model = {}
        self.store = SyncStoreWrapper(MemoryStore(mode="w"))
        # self.store = MemoryStore(mode='w')

    @rule(key=paths, data=st.binary(min_size=0, max_size=100))
    def set(self, key: str, data: bytes) -> None:
        note(f"(set) Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        asyncio.run(self.sync_wrapper.set(key, data_buf))
        self.model[key] = data_buf  # this was data

    @invariant()
    def check_paths_equal(self) -> None:
        note("Checking that paths are equal")
        paths = asyncio.run(self.sync_wrapper.list())
        assert list(self.model.keys()) == paths
        note("Checking values equal")
        for key, _val in self.model.items():
            store_item = asyncio.run(self.sync_wrapper.get(key)).to_bytes()
            # note(f'(inv) model item: {self.model[key]}')
            # note(f'(inv) {store_item=}')
            assert self.model[key].to_bytes() == store_item

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    # @rule(key=keys_bundle)
    def get(self, data) -> None:
        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        store_value = asyncio.run(self.sync_wrapper.get(key))
        # to bytes here necessary (on model and store) because data_buf set to model in set()
        assert self.model[key].to_bytes() == store_value.to_bytes()

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get_partial_values(self, data) -> None:
        key_st = st.sampled_from(sorted(self.model.keys()))
        key_range = data.draw(key_ranges(keys=key_st))

        obs_maybe = asyncio.run(self.sync_wrapper.get_partial_values(key_range))
        observed = []

        for obs in obs_maybe:
            assert obs is not None
            observed.append(obs.to_bytes())

        model_vals_ls = []

        for key, byte_range in key_range:
            model_vals = self.model[key]
            start = byte_range[0] or 0
            step = byte_range[1]
            stop = start + step if step is not None else None
            model_vals_ls.append(model_vals[start:stop])

        assert all(
            obs == exp.to_bytes() for obs, exp in zip(observed, model_vals_ls, strict=True)
        ), (
            observed,
            model_vals_ls,
        )

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def delete(self, data) -> None:
        path_st = data.draw(st.sampled_from(sorted(self.model.keys())))
        note(f"(delete) Deleting {path_st=}")

        asyncio.run(self.sync_wrapper.delete(path_st))
        del self.model[path_st]

    @rule(key=paths, data=st.binary(min_size=0, max_size=100))
    def clear(self, key: str, data: bytes):
        """clear() is in zarr/store/memory.py
        it calls clear on self._store_dict
        clear() is dict method that removes all key-val pairs from dict
        """
        assert not self.store.mode.readonly
        note("(clear)")
        asyncio.run(self.sync_wrapper.clear())
        self.model.clear()

        assert len(self.model.keys()) == 0

    # @rule()
    # def empty(self, data) -> None:
    #    """empty checks if a store is empty or not
    #    return true if self._store_dict doesn't exist
    #    return false if self._store_dict exists"""
    #    note("(empty)")

    #    asyncio.run(self.sync_wrapper.empty())
    #    assert self.store.empty()


StatefulStoreTest = ZarrStoreStateMachine.TestCase
