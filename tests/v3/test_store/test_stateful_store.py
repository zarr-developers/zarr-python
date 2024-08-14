# Stateful tests for arbitrary Zarr stores.

import asyncio

import hypothesis.strategies as st
from hypothesis import note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

from zarr.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.store import MemoryStore
from zarr.testing.strategies import key_ranges, paths

# zarr spec: https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html


class SyncStoreWrapper:
    def __init__(self, store):
        """Class to hold sync functions that map to async methods of MemoryStore
        MemoryStore methods are async, this class' methods are sync, so just need to call asyncio.run() in them
        then, methods in statemachine class are sync and call sync.
        Unfortunately, hypothesis' stateful testing infra does not support asyncio
        So we redefine sync versions of the Store API.
        https://github.com/HypothesisWorks/hypothesis/issues/3712#issuecomment-1668999041
        """
        self.store = store
        self.mode = store.mode

    def set(self, key, data_buffer):
        return asyncio.run(self.store.set(key, data_buffer))

    def list(self):
        async def wrapper(gen):
            return [i async for i in gen]

        gen = self.store.list()
        yield from asyncio.run(wrapper(gen))

    def get(self, key, prototype: BufferPrototype):
        # the input args on this get should match get() from memory store
        obs = asyncio.run(self.store.get(key, prototype=prototype))
        return obs

    def get_partial_values(self, key_ranges):
        obs_partial = asyncio.run(
            self.store.get_partial_values(
                prototype=default_buffer_prototype(), key_ranges=key_ranges
            )
        )
        return obs_partial

    def delete(self, path):
        return asyncio.run(self.store.delete(path))

    def empty(self):
        return asyncio.run(self.store.empty())

    def clear(self):
        return asyncio.run(self.store.clear())

    def exists(self, key):
        return asyncio.run(self.store.exists(key))

    def list_dir(self, prefix):
        raise NotImplementedError

    def list_prefix(self, prefix: str):
        raise NotImplementedError

    def set_partial_values(self, key_start_values):
        raise NotImplementedError

    def supports_listing(self):
        raise NotImplementedError

    def supports_partial_writes(self):
        raise NotImplementedError

    def supports_writes(self):
        raise NotImplementedError


class ZarrStoreStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.model: dict[str, bytes] = {}
        self.store = SyncStoreWrapper(MemoryStore(mode="w"))

    @rule(key=paths, data=st.binary(min_size=0, max_size=100))
    def set(self, key: str, data: bytes) -> None:
        note(f"(set) Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        self.store.set(key, data_buf)
        self.model[key] = data_buf

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(key=paths, data=st.data(), prototype=st.just(default_buffer_prototype()))
    def get(self, key, data, prototype) -> None:
        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        model_value = self.model[key]
        note(f"(get) model value {model_value.to_bytes()}")
        store_value = self.store.get(key, prototype)
        note(f"(get) store value: {store_value.to_bytes()}")
        # to bytes here necessary because data_buf set to model in set()
        assert self.model[key].to_bytes() == (store_value.to_bytes())

    # write test for invalid keys
    # eg @rule() def get_invalid() <- key strategy that can have valid and invalid keys
    # check that drawn key not in keys, then check that model/store returns None or raises error
    # look at using assume, assume key not in model keys

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get_partial_values(self, data) -> None:
        key_range = data.draw(key_ranges(keys=st.sampled_from(sorted(self.model.keys()))))
        note(f"(get partial) {key_range=}")
        obs_maybe = self.store.get_partial_values(key_range)
        observed = []

        for obs in obs_maybe:
            assert obs is not None
            observed.append(obs.to_bytes())

        model_vals_ls = []

        for key, byte_range in key_range:
            model_vals = self.model[key]
            start = byte_range[0] or 0
            step = byte_range[1]
            stop = start + step if step is not None else None
            model_vals_ls.append(model_vals[start:stop])

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

        # checking that clear did happen, since i don't think this is specifically
        # checked in invariants, though it should still work
        assert (
            len(self.model.keys()) == len(list(self.store.list())) == 0
        )  # <- make sure this is tested in an invariant

    @rule()
    def empty(self) -> None:
        note("(empty)")

        # make sure they either both are or both aren't empty (same state)
        assert self.store.empty() == (not self.model)

    @rule(key=paths)
    def exists(self, key) -> None:
        note("(exists)")

        assert self.store.exists(key) == key in self.model

    @invariant()
    def check_paths_equal(self) -> None:
        note("Checking that paths are equal")
        paths = list(self.store.list())

        assert list(self.model.keys()) == paths

    @invariant()
    def check_vals_equal(self) -> None:
        note("Checking values equal")
        for key, _val in self.model.items():
            store_item = self.store.get(key).to_bytes()
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
