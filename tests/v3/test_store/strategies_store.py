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

class StoreStatefulStrategies():
    def __init__(self):
        self.group_st = st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10)
        self.key_st =  st.lists(self.group_st, min_size=1, max_size=5).map("/".join)
        self.data_st = st.binary(min_size=0, max_size=100)

def key_ranges(keys: SearchStrategy | None = None):
    if keys is None:
        group_st = st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10)
        keys = st.lists(group_st, min_size=1, max_size=5).map("/".join)
    byte_ranges = st.tuples(
        st.none() | st.integers(min_value=0), st.none() | st.integers(min_value=0)
    )
    key_tuple = st.tuples(keys, byte_ranges)
    key_range_st = st.lists(key_tuple, min_size=1, max_size=10)
    return key_range_st