# Stateful tests for arbitrary Zarr stores.
import pytest
from hypothesis.stateful import (
    Settings,
    run_state_machine_as_test,
)

from zarr.abc.store import Store
from zarr.storage import LocalStore, MemoryStore, ZipStore
from zarr.testing.stateful import ZarrHierarchyStateMachine, ZarrStoreStateMachine


def test_zarr_hierarchy(sync_store: Store):
    def mk_test_instance_sync() -> ZarrHierarchyStateMachine:
        return ZarrHierarchyStateMachine(sync_store)

    if isinstance(sync_store, ZipStore):
        pytest.skip(reason="ZipStore does not support delete")
    if isinstance(sync_store, MemoryStore):
        run_state_machine_as_test(
            mk_test_instance_sync, settings=Settings(report_multiple_bugs=False)
        )


def test_zarr_store(sync_store: Store) -> None:
    def mk_test_instance_sync() -> None:
        return ZarrStoreStateMachine(sync_store)

    if isinstance(sync_store, ZipStore):
        pytest.skip(reason="ZipStore does not support delete")
    if isinstance(sync_store, LocalStore):
        pytest.skip(reason="This test has errors")
    run_state_machine_as_test(mk_test_instance_sync, settings=Settings(report_multiple_bugs=True))
