# Stateful tests for arbitrary Zarr stores.
import pytest
from hypothesis.stateful import (
    run_state_machine_as_test,
)

from zarr.abc.store import Store
from zarr.storage import ZipStore
from zarr.testing.stateful import ZarrHierarchyStateMachine, ZarrStoreStateMachine

pytestmark = pytest.mark.slow_hypothesis


def test_zarr_hierarchy(sync_store: Store):
    def mk_test_instance_sync() -> ZarrHierarchyStateMachine:
        return ZarrHierarchyStateMachine(sync_store)

    if isinstance(sync_store, ZipStore):
        pytest.skip(reason="ZipStore does not support delete")

    run_state_machine_as_test(mk_test_instance_sync)


def test_zarr_store(sync_store: Store) -> None:
    def mk_test_instance_sync() -> None:
        return ZarrStoreStateMachine(sync_store)

    if isinstance(sync_store, ZipStore):
        pytest.skip(reason="ZipStore does not support delete")

    run_state_machine_as_test(mk_test_instance_sync)
