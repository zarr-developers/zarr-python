# Stateful tests for arbitrary Zarr stores.
import pytest
from hypothesis.stateful import (
    run_state_machine_as_test,
)

from zarr.abc.store import Store
from zarr.storage import LocalStore, ZipStore
from zarr.testing.stateful import ZarrHierarchyStateMachine, ZarrStoreStateMachine

pytestmark = [
    pytest.mark.slow_hypothesis,
    # TODO: work out where this warning is coming from and fix
    pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning"),
]


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
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

    if isinstance(sync_store, LocalStore):
        # This test uses arbitrary keys, which are passed to `set` and `delete`.
        # It assumes that `set` and `delete` are the only two operations that modify state.
        # But LocalStore, directories can hang around even after a key is delete-d.
        pytest.skip(reason="Test isn't suitable for LocalStore.")
    run_state_machine_as_test(mk_test_instance_sync)
