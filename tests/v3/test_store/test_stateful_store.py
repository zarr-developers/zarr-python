# Stateful tests for arbitrary Zarr stores.

import asyncio
import string

import hypothesis.strategies as st
from hypothesis import note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)
from hypothesis.strategies import SearchStrategy

from zarr.buffer import Buffer, default_buffer_prototype
from zarr.store import MemoryStore

from strategies_store import StoreStatefulStrategies, key_ranges


strategies = StoreStatefulStrategies()

class ZarrStoreStateMachine(RuleBasedStateMachine):
    #TODO add arg/class for store type
    def __init__(self):
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
        return paths

    async def get_key(self, key):
        obs = await self.store.get(key, prototype=default_buffer_prototype())
        return obs.to_bytes()

    async def get_partial(self, key_ranges):
        # read back part
        obs_maybe = await self.store.get_partial_values(
            prototype=default_buffer_prototype(), key_ranges=key_ranges
        )
        return obs_maybe

    async def delete_key(self, path):

        await self.store.delete(path)


    # ------

    @rule(key=strategies.key_st, data=strategies.data_st)#st.binary(min_size=0, max_size=100))
    def set(self, key: str, data: bytes) -> None:
        note(f"Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        asyncio.run(self.store_set(key, data_buf))
        # TODO: does model need to contain Buffer or just data?
        self.model[key] = data

    @invariant()
    def check_paths_equal(self) -> None:
        note("Checking that paths are equal")
        paths = asyncio.run(self.store_list())
        assert list(self.model.keys()) == paths

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get(self, data) -> None:

        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        store_value = asyncio.run(self.get_key(key))
        assert self.model[key] == store_value

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

        for idx in range(len(observed)):
            key, byte_range = key_range[idx]
            model_vals = self.model[key]
            start = byte_range[0] or 0
            step = byte_range[1]
            stop = start + step if step is not None else None
            model_vals_partial = model_vals[start:stop]
            model_vals_ls.append(model_vals_partial)

        assert all(obs == exp for obs, exp in zip(observed, model_vals_ls, strict=True)), (
            observed,
            model_vals_ls,
        )

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def delete(self,data) -> None:

        path_st = data.draw(st.sampled_from(sorted(self.model.keys())))
        note(f'Deleting {path_st=}')

        asyncio.run(self.delete_key(path_st))

        del self.model[path_st]

    @invariant()
    def check_delete(self) -> None:
        #can/should this be the same as the invariant for set? 
        paths = asyncio.run(self.store_list())
        note(f"Checking that paths are equal, {paths=}, model={self.model.keys()}")
        assert list(self.model.keys()) == paths



StatefulStoreTest = ZarrStoreStateMachine.TestCase
