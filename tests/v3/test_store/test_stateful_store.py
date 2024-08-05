import hypothesis.strategies as st
import zarr
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

    #rules for get, set
    @rule()
    def set(self, key:str, data: bytes) -> None:

        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        self.store.set(key, data_buf)
        self.model[key] = data_buf
    
    @invariant()
    def check_paths_equal(self):
        assert self.model.keys() == self.store.list() #check zarr syntax

    #@rule()
    #def get(self, key:str, data:bytes) -> None:
    #    data_buf = Buffer.from_bytes(data)

    #    self.set(self.store, key, data_buf)

    #    self.model[key] = 



#ZarrStoreStateMachine.TestCase.settings = settings()#max_examples=300, deadline=None)
StatefulStoreTest = ZarrStoreStateMachine.TestCase

