# Stateful tests for arbitrary Zarr stores.

import pytest
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
    # ------

    # rules for get, set
    #TODO: st.just
    @rule(key=st.just("a"), data=st.just(b"0"))
    def set(self, key:str, data: bytes) -> None:
        note(f"Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        asyncio.run(self.store_set(key, data_buf))
        # TODO: does model need to contain Buffer or just data?
        self.model[key] = data_buf

    @invariant()
    def check_paths_equal(self) -> None:
        note("Checking that paths are equal")
        paths = asyncio.run(self.store_list())
        assert list(self.model.keys()) == paths

    #@rule()
    #def get(self, key:str, data:bytes) -> None:
    #    data_buf = Buffer.from_bytes(data)

    #    self.set(self.store, key, data_buf)

    #    self.model[key] = 



#ZarrStoreStateMachine.TestCase.settings = settings()#max_examples=300, deadline=None)
StatefulStoreTest = ZarrStoreStateMachine.TestCase

