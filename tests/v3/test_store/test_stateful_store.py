# Stateful tests for arbitrary Zarr stores.
import hypothesis.strategies as st
import pytest
from hypothesis import assume, note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    Settings,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from hypothesis.strategies import DataObject

import zarr
from zarr.abc.store import AccessMode, Store
from zarr.core.buffer import BufferPrototype, cpu, default_buffer_prototype
from zarr.storage import LocalStore, ZipStore
from zarr.testing.strategies import key_ranges
from zarr.testing.strategies import keys as zarr_keys

MAX_BINARY_SIZE = 100


class SyncStoreWrapper(zarr.core.sync.SyncMixin):
    def __init__(self, store: Store) -> None:
        """Synchronous Store wrapper

        This class holds synchronous methods that map to async methods of Store classes.
        The synchronous wrapper is needed because hypothesis' stateful testing infra does
        not support asyncio so we redefine sync versions of the Store API.
        https://github.com/HypothesisWorks/hypothesis/issues/3712#issuecomment-1668999041
        """
        self.store = store

    @property
    def mode(self) -> AccessMode:
        return self.store.mode

    def set(self, key: str, data_buffer: zarr.core.buffer.Buffer) -> None:
        return self._sync(self.store.set(key, data_buffer))

    def list(self) -> list:
        return self._sync_iter(self.store.list())

    def get(self, key: str, prototype: BufferPrototype) -> zarr.core.buffer.Buffer:
        return self._sync(self.store.get(key, prototype=prototype))

    def get_partial_values(
        self, key_ranges: list, prototype: BufferPrototype
    ) -> zarr.core.buffer.Buffer:
        return self._sync(self.store.get_partial_values(prototype=prototype, key_ranges=key_ranges))

    def delete(self, path: str) -> None:
        return self._sync(self.store.delete(path))

    def empty(self) -> bool:
        return self._sync(self.store.empty())

    def clear(self) -> None:
        return self._sync(self.store.clear())

    def exists(self, key) -> bool:
        return self._sync(self.store.exists(key))

    def list_dir(self, prefix):
        raise NotImplementedError

    def list_prefix(self, prefix: str):
        raise NotImplementedError

    def set_partial_values(self, key_start_values):
        raise NotImplementedError

    @property
    def supports_listing(self) -> bool:
        return self.store.supports_listing

    @property
    def supports_partial_writes(self) -> bool:
        return self.supports_partial_writes

    @property
    def supports_writes(self) -> bool:
        return self.store.supports_writes


class ZarrStoreStateMachine(RuleBasedStateMachine):
    """ "
    Zarr store state machine

        This is a subclass of a Hypothesis RuleBasedStateMachine.
        It is testing a framework to ensure that the state of a Zarr store matches
        an expected state after a set of random operations. It contains a store
        (currently, a Zarr MemoryStore) and a model, a simplified version of a
        zarr store (in this case, a dict). It also contains rules which represent
        actions that can be applied to a zarr store. Rules apply an action to both
        the store and the model, and invariants assert that the state of the model
        is equal to the state of the store. Hypothesis then generates sequences of
        rules, running invariants after each rule. It raises an error if a sequence
        produces discontinuity between state of the model and state of the store
        (ie. an invariant is violated).
        https://hypothesis.readthedocs.io/en/latest/stateful.html
    """

    def __init__(self, store: Store) -> None:
        super().__init__()
        self.model: dict[str, bytes] = {}
        self.store = SyncStoreWrapper(store)
        self.prototype = default_buffer_prototype()

    @initialize()
    def init_store(self):
        self.store.clear()

    @rule(key=zarr_keys, data=st.binary(min_size=0, max_size=MAX_BINARY_SIZE))
    def set(self, key: str, data: DataObject) -> None:
        note(f"(set) Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = cpu.Buffer.from_bytes(data)
        self.store.set(key, data_buf)
        self.model[key] = data_buf

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(key=zarr_keys, data=st.data())
    def get(self, key: str, data: DataObject) -> None:
        key = data.draw(
            st.sampled_from(sorted(self.model.keys()))
        )  # hypothesis wants to sample from sorted list
        note("(get)")
        store_value = self.store.get(key, self.prototype)
        # to bytes here necessary because data_buf set to model in set()
        assert self.model[key].to_bytes() == (store_value.to_bytes())

    @rule(key=zarr_keys, data=st.data())
    def get_invalid_zarr_keys(self, key: str, data: DataObject) -> None:
        note("(get_invalid)")
        assume(key not in self.model)
        assert self.store.get(key, self.prototype) is None

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get_partial_values(self, data: DataObject) -> None:
        key_range = data.draw(
            key_ranges(keys=st.sampled_from(sorted(self.model.keys())), max_size=MAX_BINARY_SIZE)
        )
        note(f"(get partial) {key_range=}")
        obs_maybe = self.store.get_partial_values(key_range, self.prototype)
        observed = []

        for obs in obs_maybe:
            assert obs is not None
            observed.append(obs.to_bytes())

        model_vals_ls = []

        for key, byte_range in key_range:
            start = byte_range[0] or 0
            step = byte_range[1]
            stop = start + step if step is not None else None
            model_vals_ls.append(self.model[key][start:stop])

        assert all(
            obs == exp.to_bytes() for obs, exp in zip(observed, model_vals_ls, strict=True)
        ), (
            observed,
            model_vals_ls,
        )

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def delete(self, data: DataObject) -> None:
        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        note(f"(delete) Deleting {key=}")

        self.store.delete(key)
        del self.model[key]

    @rule()
    def clear(self) -> None:
        assert not self.store.mode.readonly
        note("(clear)")
        self.store.clear()
        self.model.clear()

        assert self.store.empty()

        assert len(self.model.keys()) == len(list(self.store.list())) == 0

    @rule()
    # Local store can be non-empty when there are subdirectories but no files
    @precondition(lambda self: not isinstance(self.store.store, LocalStore))
    def empty(self) -> None:
        note("(empty)")

        # make sure they either both are or both aren't empty (same state)
        assert self.store.empty() == (not self.model)

    @rule(key=zarr_keys)
    def exists(self, key: str) -> None:
        note("(exists)")

        assert self.store.exists(key) == (key in self.model)

    @invariant()
    def check_paths_equal(self) -> None:
        note("Checking that paths are equal")
        paths = sorted(self.store.list())

        assert sorted(self.model.keys()) == paths

    @invariant()
    def check_vals_equal(self) -> None:
        note("Checking values equal")
        for key, val in self.model.items():
            store_item = self.store.get(key, self.prototype).to_bytes()
            assert val.to_bytes() == store_item

    @invariant()
    def check_num_zarr_keys_equal(self) -> None:
        note("check num zarr_keys equal")

        assert len(self.model) == len(list(self.store.list()))

    @invariant()
    def check_zarr_keys(self) -> None:
        keys = list(self.store.list())

        if not keys:
            assert self.store.empty() is True

        else:
            assert self.store.empty() is False

            for key in keys:
                assert self.store.exists(key) is True
        note("checking keys / exists / empty")


def test_zarr_hierarchy(sync_store: Store) -> None:
    def mk_test_instance_sync() -> None:
        return ZarrStoreStateMachine(sync_store)

    if isinstance(sync_store, ZipStore):
        pytest.skip(reason="ZipStore does not support delete")
    if isinstance(sync_store, LocalStore):
        pytest.skip(reason="This test has errors")
    run_state_machine_as_test(mk_test_instance_sync, settings=Settings(report_multiple_bugs=True))
