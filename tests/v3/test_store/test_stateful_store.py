# Stateful tests for arbitrary Zarr stores.

import pytest
import string
import hypothesis.strategies as st
import zarr
import asyncio
from hypothesis import note
from hypothesis.stateful import (
    Bundle,
    HealthCheck,
    RuleBasedStateMachine,
    Settings,
    consumes,       
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from zarr.store import MemoryStore
from zarr.buffer import Buffer, default_buffer_prototype
from zarr.store.utils import _normalize_interval_index
from zarr.testing.utils import assert_bytes_equal


class ZarrStoreStateMachine(RuleBasedStateMachine):
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
    
    async def get_key(self, key ):
        obs = await self.store.get(key, prototype= default_buffer_prototype())
        return obs.to_bytes()
    
    async def get_partial(self, key_ranges):

        #read back part
        obs_maybe = await self.store.get_partial_values(
            prototype=default_buffer_prototype(), key_ranges=key_ranges
        )

        return obs_maybe 
    
    # ------

    # rules for get, set
    #strategy for key
    group_st = st.text(alphabet=string.ascii_letters + string.digits,min_size=1, max_size=10)
    middle_st = st.one_of(st.just('c'), st.integers(min_value=0, max_value=100).map(str))
    end_st = st.one_of(st.just('zarr.json'), st.integers(min_value=0, max_value=10).map(str))
    key_st = st.tuples(group_st, middle_st, end_st).map('/'.join)
    
    #strategy for data
    int_st = st.integers(min_value=0, max_value=255)
    data_st = st.lists(int_st, min_size=0).map(bytes)

    #strategy for key_ranges
    inner_tuple_st = st.tuples(st.one_of(st.integers(), st.none()), st.one_of(st.integers(), st.none()))
    key_range_st = st.tuples(st.one_of(key_st), inner_tuple_st)

    @rule(key=key_st, data=data_st)
    def set(self, key:str, data: bytes) -> None:
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
    @rule(data = st.data())
    def get(self, data) -> None:

        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        
        store_value = asyncio.run(self.get_key(key))
        assert self.model[key] == store_value
    
 
    @rule(key_ranges = key_range_st)
    def get_partial_data(self, key_ranges) -> None:

        #for all keys in range, set key, buffer obj as pair to store
        for key, _ in key_ranges:
            self.store.set(key, Buffer.from_bytes(bytes(key, encoding='utf-8')))

        #read back part
        obs_maybe = asyncio.run(self.get_partial(key_ranges))

        observed = []

        for obs in obs_maybe:
            assert result is not None
            observed.append(obs)
            
        self.store.partial_vals = observed #this is wrong

    @invariant()
    def check_partial_values(self) -> None:

        model_vals = []
        for idx in range(len(self.store.partial_vals)):

            key, byte_range = key_ranges[idx]
            model_vals = self.model[key]

        assert all(
            obs.to_bytes() == exp.to_bytes() for obs, exp in zip(self.store.partial_vals, model_vals, strict=True)
        )

#ZarrStoreStateMachine.TestCase.settings = settings()#max_examples=300, deadline=None)
StatefulStoreTest = ZarrStoreStateMachine.TestCase

