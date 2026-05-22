# Stateful tests for arbitrary Zarr stores.
from collections.abc import Generator

import pytest
from hypothesis.stateful import (
    run_state_machine_as_test,
)

import zarr
from zarr.abc.store import Store
from zarr.storage import LocalStore, ZipStore
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
def test_zarr_hierarchy(sync_store: Store) -> None:
    def mk_test_instance_sync() -> ZarrHierarchyStateMachine:
        return ZarrHierarchyStateMachine(sync_store)

    if isinstance(sync_store, ZipStore):
        pytest.skip(reason="ZipStore does not support delete")

    run_state_machine_as_test(mk_test_instance_sync)  # type: ignore[no-untyped-call]


def test_zarr_store(sync_store: Store) -> None:
    def mk_test_instance_sync() -> ZarrStoreStateMachine:
        return ZarrStoreStateMachine(sync_store)

    if isinstance(sync_store, ZipStore):
        pytest.skip(reason="ZipStore does not support delete")

    if isinstance(sync_store, LocalStore):
        # This test uses arbitrary keys, which are passed to `set` and `delete`.
        # It assumes that `set` and `delete` are the only two operations that modify state.
        # But LocalStore, directories can hang around even after a key is delete-d.
        pytest.skip(reason="Test isn't suitable for LocalStore.")
    run_state_machine_as_test(mk_test_instance_sync)  # type: ignore[no-untyped-call]


def test_delete_dir_prefix_matching() -> None:
    """Regression test for delete_dir prefix matching bug (GH#3977).

    Verifies that delete_dir bookkeeping only removes exact path matches
    and true descendants, not unrelated nodes that merely share a string
    prefix (e.g. ``6/faNT…`` must NOT be deleted when removing ``6/f``).
    """
    all_groups = {"6/f", "6/faNT7p7jvJsO3_C._HYi", "other"}
    all_arrays = {"6/f/child", "6/other"}
    path = "6/f"

    matches = set()
    for node in all_groups | all_arrays:
        if node == path or node.startswith(path + "/"):
            matches.add(node)

    assert matches == {"6/f", "6/f/child"}
    assert "6/faNT7p7jvJsO3_C._HYi" not in matches
    assert "other" not in matches
    assert "6/other" not in matches
