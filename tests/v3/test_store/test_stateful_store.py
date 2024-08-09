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

from zarr.buffer import Buffer, default_buffer_prototype
from zarr.store import MemoryStore

# from strategies_store import StoreStatefulStrategies, key_ranges
from zarr.strategies import key_ranges, paths

# zarr spec: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html


# NOTE: all methods should be called on self.store, assertions should compare self.model and self.store._store_dict
class ZarrStoreStateMachine(RuleBasedStateMachine):
    def __init__(self):  # look into using run_machine_as_test()
        super().__init__()
        self.model = {}
        self.store = MemoryStore(mode="w")

    # Unfortunately, hypothesis' stateful testing infra does not support asyncio
    # So we redefine sync versions of the Store API.
    # https://github.com/HypothesisWorks/hypothesis/issues/3712#issuecomment-1668999041
    async def store_set(self, key, data_buffer):
        await self.store.set(key, data_buffer)

    async def store_list(self):
        paths = [path async for path in self.store.list()]
        # note(f'(store set) store {paths=}, type {type(paths)=}, {len(paths)=}')
        return paths

    async def get_key(self, key):
        obs = await self.store.get(key, prototype=default_buffer_prototype())
        return obs

    async def get_partial(self, key_ranges):
        obs_maybe = await self.store.get_partial_values(
            prototype=default_buffer_prototype(), key_ranges=key_ranges
        )
        return obs_maybe

    async def delete_key(self, path):
        await self.store.delete(path)

    async def store_empty(self):
        await self.store.empty()

    async def store_clear(self):
        await self.store.clear()

    # async def listdir(self, group): #does listdir take group?

    #    dir_ls = await self.store.list_dir(group)

    #    return dir_ls
    # ------

    @rule(key=paths, data=st.binary(min_size=0, max_size=100))
    def set(self, key: str, data: bytes) -> None:
        note(f"(set) Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        asyncio.run(self.store_set(key, data_buf))
        self.model[key] = data_buf  # this was data

    @invariant()
    def check_paths_equal(self) -> None:
        note("Checking that paths are equal")
        paths = asyncio.run(self.store_list())
        # note(f'(check paths equal) {self.model=}, {self.store._store_dict=}')
        assert list(self.model.keys()) == paths
        assert len(self.model.keys()) == len(self.store)
        assert self.model == self.store._store_dict

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    # @rule(key=keys_bundle)
    def get(self, data) -> None:
        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        store_value = asyncio.run(self.get_key(key))
        # note(f'(get) {self.model[key]=}, {store_value}')
        # to bytes here necessary (on model and store) because data_buf set to model in set()
        assert self.model[key].to_bytes() == store_value.to_bytes()

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get_partial_values(self, data) -> None:
        key_st = st.sampled_from(sorted(self.model.keys()))
        key_range = data.draw(key_ranges(keys=key_st))

        obs_maybe = asyncio.run(self.get_partial(key_range))
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

        asyncio.run(self.delete_key(path_st))

        del self.model[path_st]

        # property tests
        assert self.model.keys() == self.store._store_dict.keys()
        assert path_st not in list(self.model.keys())

    @rule(key=paths, data=st.binary(min_size=0, max_size=100))
    def clear(self, key: str, data: bytes):
        """clear() is in zarr/store/memory.py
        it calls clear on self._store_dict
        clear() is dict method that removes all key-val pairs from dict
        """
        assert not self.store.mode.readonly
        note("(clear)")
        asyncio.run(self.store_clear())
        self.model.clear()

        assert self.model == self.store._store_dict
        assert len(self.model.keys()) == 0

    def empty(self, data) -> None:
        """empty checks if a store is empty or not
        return true if self._store_dict doesn't exist
        return false if self._store_dict exists"""
        note("(empty)")

        asyncio.run(self.store_clear())
        assert self.store.empty()

    # @precondition(lambda self: len(self.model.keys()) > 0)
    # @rule(key = paths, data=st.binary(min_size=0, max_size=100))
    # def listdir(self, key, data) -> None:
    #    '''list_dir - Retrieve all keys and prefixes with a given prefix
    #    and which do not contain the character “/” after the given prefix.
    #    '''

    # assert not self.store.mode.readonly
    # data_buf = Buffer.from_bytes(data)
    # set keys on store
    # asyncio.run(self.store_set(key, data_buf))
    # set same keys on model
    # self.model[key] = data
    # list keys on store
    # asyncio.run(self.listdir(key))

    # self.model.listed_keys = list(self.model.keys())
    # for store, set random keys

    # for store, list all keys
    # for model, list all keys


StatefulStoreTest = ZarrStoreStateMachine.TestCase


# @invariant()
# def check_delete(self) -> None:
#    '''this doesn't actually do anything'''
#    note('(check_delete)')
# can/should this be the same as the invariant for set?
#    paths = asyncio.run(self.store_list())
#    note(f"After delete, checking that paths are equal, {paths=}, model={list(self.model.keys())}")
#    assert list(self.model.keys()) == paths
#    assert len(list(self.model.keys())) == len(paths)
# maybe add assertion that path_st not in keys? but this would req paths_st from delete()..


# @invariant()
# listdir not working
# def check_listdir(self):

#    store_keys = asyncio.run(self.list_dir())#need to pass a group to this
#    model_keys = self.model.listed_keys
#    assert store_keys == model_keys
