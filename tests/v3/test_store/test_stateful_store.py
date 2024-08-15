# Stateful tests for arbitrary Zarr stores.


import hypothesis.strategies as st
from hypothesis import assume, note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

import zarr
from zarr.abc.store import AccessMode, Store
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.store import MemoryStore
from zarr.testing.strategies import key_ranges, paths


class SyncStoreWrapper(zarr.core.sync.SyncMixin):
    def __init__(self, store: Store):
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

    def set(self, key, data_buffer) -> None:
        return sync(self.store.set(key, data_buffer))

    def list(self) -> list:
    return self._sync_iter(self.store.list())

    def get(self, key, prototype: BufferPrototype) -> zarr.core.buffer.Buffer:
        obs = self._sync(self.store.get(key, prototype=prototype))
        return obs

    def get_partial_values(self, key_ranges, prototype: BufferPrototype) -> zarr.core.buffer.Buffer:
        obs_partial = self._sync(
            self.store.get_partial_values(prototype=prototype, key_ranges=key_ranges)
        )
        return obs_partial

    def delete(self, path) -> None:
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

    def __init__(self):
        super().__init__()
        self.model: dict[str, bytes] = {}
        self.store = SyncStoreWrapper(MemoryStore(mode="w"))
        self.prototype = default_buffer_prototype()

    @rule(key=paths, data=st.binary(min_size=0, max_size=100))
    def set(self, key: str, data: bytes) -> None:
        note(f"(set) Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        self.store.set(key, data_buf)
        self.model[key] = data_buf

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(key=paths, data=st.data())
    def get(self, key, data) -> None:
        key = data.draw(
            st.sampled_from(sorted(self.model.keys()))
        )  # hypothesis wants to sample from sorted list
        note("(get)")
        store_value = self.store.get(key, self.prototype)
        # to bytes here necessary because data_buf set to model in set()
        assert self.model[key].to_bytes() == (store_value.to_bytes())

    @rule(key=paths, data=st.data())
    def get_invalid_keys(self, key, data) -> None:
        note("(get_invalid)")
        assume(key not in self.model.keys())
        assert self.store.get(key, self.prototype) is None

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get_partial_values(self, data) -> None:
        key_range = data.draw(key_ranges(keys=st.sampled_from(sorted(self.model.keys()))))
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
    def delete(self, data) -> None:
        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        note(f"(delete) Deleting {key=}")

        self.store.delete(key)
        del self.model[key]

    @rule()
    def clear(self):
        assert not self.store.mode.readonly
        note("(clear)")
        self.store.clear()
        self.model.clear()

        assert len(self.model.keys()) == len(list(self.store.list())) == 0

    @rule()
    def empty(self) -> None:
        note("(empty)")

        # make sure they either both are or both aren't empty (same state)
        assert self.store.empty() == (not self.model)

    @rule(key=paths)
    def exists(self, key) -> None:
        note("(exists)")

        assert self.store.exists(key) == (key in self.model)

    @invariant()
    def check_paths_equal(self) -> None:
        note("Checking that paths are equal")
        paths = list(self.store.list())

        assert list(self.model.keys()) == paths

    @invariant()
    def check_vals_equal(self) -> None:
        note("Checking values equal")
        for key, _val in self.model.items():
            store_item = self.store.get(key, self.prototype).to_bytes()
            assert self.model[key].to_bytes() == store_item

    @invariant()
    def check_num_keys_equal(self) -> None:
        note("check num keys equal")

        assert len(self.model) == len(list(self.store.list()))

    @invariant()
    def check_keys(self) -> None:
        keys = list(self.store.list())

        if len(keys) == 0:
            assert self.store.empty() is True

        elif len(keys) != 0:
            assert self.store.empty() is False

            for key in keys:
                assert self.store.exists(key) is True
        note("checking keys / exists / empty")


StatefulStoreTest = ZarrStoreStateMachine.TestCase
