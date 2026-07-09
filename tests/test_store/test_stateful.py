# Stateful tests for arbitrary Zarr stores.
from collections.abc import Generator

import pytest
from hypothesis.stateful import (
    run_state_machine_as_test,
)

import zarr
from zarr.abc.store import Store
from zarr.storage import LocalStore
from zarr.testing.stateful import ZarrHierarchyStateMachine, ZarrStoreStateMachine

pytestmark = [
    pytest.mark.slow_hypothesis,
    # TODO: work out where this warning is coming from and fix
    pytest.mark.filterwarnings("ignore:Unclosed client session:ResourceWarning"),
]


@pytest.fixture(autouse=True)
def _enable_rectilinear_chunks() -> Generator[None, None, None]:
    """Enable rectilinear chunks since strategies may generate them."""
    with zarr.config.set({"array.rectilinear_chunks": True}):
        yield


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_zarr_hierarchy(sync_store: Store) -> None:
    def mk_test_instance_sync() -> ZarrHierarchyStateMachine:
        return ZarrHierarchyStateMachine(sync_store)

    # ZipStore now supports soft-delete — skip removed
    run_state_machine_as_test(mk_test_instance_sync)  # type: ignore[no-untyped-call]


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_zarr_store(sync_store: Store) -> None:
    def mk_test_instance_sync() -> ZarrStoreStateMachine:
        return ZarrStoreStateMachine(sync_store)

    # ZipStore now supports soft-delete — skip removed

    if isinstance(sync_store, LocalStore):
        pytest.skip(reason="Test isn't suitable for LocalStore.")
    run_state_machine_as_test(mk_test_instance_sync)  # type: ignore[no-untyped-call]
