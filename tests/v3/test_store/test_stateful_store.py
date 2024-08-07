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
        note(f'async store key range: {key_ranges}')
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
    #data_st = st.lists(int_st, min_size=0).map(bytes)
    data_st = st.just([1,2,3,4]).map(bytes)

    #strategy for key_ranges
    inner_tuple_st = st.tuples(st.one_of(st.integers(), st.none()), st.one_of(st.integers(), st.none()))
    key_range_st = st.lists(st.tuples(st.one_of(key_st), inner_tuple_st))

    @rule(key=key_st, data=data_st)
    def set(self, key:str, data: bytes) -> None:
        #note(f"rule(set): Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        asyncio.run(self.store_set(key, data_buf))
        # TODO: does model need to contain Buffer or just data?
        self.model[key] = data

    @invariant()
    def check_paths_equal(self) -> None:
        #note("inv: Checking that paths are equal")
        paths = asyncio.run(self.store_list())
        assert list(self.model.keys()) == paths


    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data = st.data())
    def get(self, data) -> None:

        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        
        store_value = asyncio.run(self.get_key(key))
        assert self.model[key] == store_value
    
    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data = st.data())
    def get_partial_values(self, data) -> None:
        '''notes on what get_partial_values() does:
        - takes self, key_ranges (list of tuples), BufferPrototype
        - for key, byte_range in key_ranges, 
            - check that key is str
            - make path (path = self.root / key)
            - make tuple: (_get, path, prototype, byte_range)
            - append tuple to args list 
            - pass args list to: await concurrent_map()
            in concurrent_map():
            - if limit=None, call asyncio.gather() and pass _get(item) for each item in args <- i think? 
                        a bit funny bc each item of args list is (_get, path, prototype, byte_range), so _get is in that item ?
            - if limit != None, make asyncio.semaphore(limit) <- a synchronization primitive
                - runs same get call on items eventually but with some async stuff
        '''

        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        val = self.model[key]
        vals_len = len(self.model[key])if self.model[key] != b'\x00' else 0
        note(f"1 rule(get_partial), store: self.model['{key}'] == {self.model[key]}")
        note(f'2 rule(get_partial), store: len(self.model[{key}]) == {vals_len}')
        #byte_range = data.draw(st.tuples(st.none() | st.integers(min_value=0, max_value=vals_len), st.none()  | st.integers(min_value = 0, max_value=vals_len)))
        #use this hardcoded byte_range to test get_partial- seems like errors when byte range None,None
        byte_range = data.draw(st.tuples(st.none(), st.none()))
        key_range = [(key, byte_range)]
        note(f'3 rule(get_partial), store: byte range: {byte_range}, vals: {self.model[key]}')#, start/stop: {byte_range[0]}/{byte_range[0]+byte_range[1]}')

        #read back part
        obs_maybe = asyncio.run(self.get_partial(key_range))
        observed = []

        for obs in obs_maybe:
            assert obs is not None
            observed.append(obs.to_bytes())
        note(f'4 rule(get_partial), store: obs result of get_partial to bytes : {observed}')
    
        #return observed
        model_vals_ls = []

        for idx in range(len(observed)):

            key, byte_range = key_range[idx]
            model_vals = self.model[key]
            start = byte_range[0] #or 0
            step = byte_range[1] #or 0 
            def calc_stop(start, step):
                '''it looks like the behavior of get_partial_values() 
                when byte_range = (None,None) is to return all,
                to match this with model, calc byte range with this instead of 
                start + step'''
                if start is None and step is not None :
                    stop = 0 + step
                    return stop
                elif start is not None and step is None:
                    return None  
                elif start is None and step is None:
                    return None
                else:
                    return start + step
            stop = calc_stop(start, step)
            note(f'model start: {start}, stop: {stop}')
            #stop = start + step
            model_vals_partial = model_vals[start:stop]
            note(f'5 rule(get_partial),model (pre get_partial) Key: {key}, byte range: {byte_range}, vals: {model_vals}')

            model_vals_ls.append(model_vals_partial)
            note(f'6 rule(get_partial), model (results of get_partial) vals: {model_vals_ls}')
        
        assert all(
            obs == exp for obs, exp in zip(observed, model_vals_ls, strict=True)
        ), (observed, model_vals_ls)

        

#ZarrStoreStateMachine.TestCase.settings = settings()#max_examples=300, deadline=None)
StatefulStoreTest = ZarrStoreStateMachine.TestCase

