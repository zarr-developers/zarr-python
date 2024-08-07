# Stateful tests for arbitrary Zarr stores.

import asyncio
import string

import hypothesis.strategies as st
from hypothesis import note
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)
from hypothesis.strategies import SearchStrategy

from zarr.buffer import Buffer, default_buffer_prototype
from zarr.store import MemoryStore

group_st = st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10)
key_st = st.lists(group_st, min_size=1, max_size=5).map("/".join)


def key_ranges(keys: SearchStrategy = key_st):
    byte_ranges = st.tuples(
        st.none() | st.integers(min_value=0), st.none() | st.integers(min_value=0)
    )
    key_tuple = st.tuples(keys, byte_ranges)
    key_range_st = st.lists(key_tuple, min_size=1, max_size=10)
    return key_range_st


class ZarrStoreStateMachine(RuleBasedStateMachine):
    keys_bundle = Bundle("keys_bundle")

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
        note(f"async store key range: {key_ranges}")
        return obs_maybe

    # ------

    # rules for get, set
    # strategy for key
    # check out st.lists
    # check out deepak's PR for additional rules in strategies
    # key_st = st.lists(group_st, min_size=1, max_size=5).map('/'.join)

    # key_range_st = key_ranges()
    # strategy for key_ranges
    # inner_tuple_st = st.tuples(st.one_of(st.integers(), st.none()), st.one_of(st.integers(), st.none()))
    # key_range_st = st.lists(st.tuples(st.one_of(key_st), inner_tuple_st))

    @rule(key=key_st, data=st.binary(min_size=0, max_size=100), target=keys_bundle)
    def set(self, key: str, data: bytes) -> None:
        note(f"rule(set): Setting {key=} with {data=}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        asyncio.run(self.store_set(key, data_buf))
        # TODO: does model need to contain Buffer or just data?
        self.model[key] = data
        return key

    @invariant()
    def check_paths_equal(self) -> None:
        # note("inv: Checking that paths are equal")
        paths = asyncio.run(self.store_list())
        assert list(self.model.keys()) == paths

    @rule(key=keys_bundle)
    def get(self, key) -> None:
        note(f"get {key=}")
        store_value = asyncio.run(self.get_key(key))
        assert self.model[key] == store_value

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get_partial_values(self, data) -> None:
        """notes on what get_partial_values() does:
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
        """
        key_st = st.sampled_from(sorted(self.model.keys()))
        key_range = data.draw(key_ranges(keys=key_st))

        # val = self.model[key]
        # note(f'{val=}')
        # vals_len = len(self.model[key])
        # byte_range = data.draw(st.tuples(st.none() | st.integers(min_value=0), st.none()  | st.integers(min_value = 0)))
        # key_range = [(key, byte_range) for key, byte_range in ]
        # key_range = [(key, (byte_range))]
        # want key_range (list of tuples):  [(key, (start, step)), (key, (start, step))....]

        # note(f'get_partial (store) {key=}, {byte_range=}, vals: {self.model[key]}')

        # read back part
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


# ZarrStoreStateMachine.TestCase.settings = settings()#max_examples=300, deadline=None)
StatefulStoreTest = ZarrStoreStateMachine.TestCase
