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
    # ------

    # rules for get, set
    group_st = st.text(alphabet=string.ascii_letters + string.digits,min_size=1, max_size=10)
    middle_st = st.one_of(st.just('c'), st.integers(min_value=0, max_value=100).map(str))
    end_st = st.one_of(st.just('zarr.json'), st.integers(min_value=0, max_value=10).map(str))
    key_st = st.tuples(group_st, middle_st, end_st).map('/'.join)
    
    int_st = st.integers(min_value=0, max_value=255)
    data_st = st.lists(int_st, min_size=0).map(bytes)

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
    @rule(data = st.data())#key=st.just("a"), data=st.just(b"0"))
    def get(self, data) -> None:

        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        
        store_value = asyncio.run(self.get_key(key))
        assert self.model[key] == store_value
    
 



#ZarrStoreStateMachine.TestCase.settings = settings()#max_examples=300, deadline=None)
StatefulStoreTest = ZarrStoreStateMachine.TestCase

