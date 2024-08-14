# Stateful tests for arbitrary Zarr stores.

import asyncio

import hypothesis.strategies as st
from hypothesis import assume, note
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

from zarr.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.store import MemoryStore
from zarr.testing.strategies import key_ranges, paths


class SyncStoreWrapper:
    def __init__(self, store):
        """Class to hold sync functions that map to async methods of MemoryStore
        MemoryStore methods are async, this class' methods are sync, so just need to call asyncio.run() in them
        then, methods in statemachine class are sync and call sync.
        Unfortunately, hypothesis' stateful testing infra does not support asyncio
        So we redefine sync versions of the Store API.
        https://github.com/HypothesisWorks/hypothesis/issues/3712#issuecomment-1668999041

        Attributes
        ----------
        store: ZarrStore
            Currently only MemoryStore is implemented.
        mode: str
            Read/write mode. This should be 'w'.

        """
        self.store = store
        self.mode = store.mode

    def set(self, key, data_buffer):
        """
        Set value of 'key' to 'data_buffer'

        Parameters
        ----------
        key: str
        data_buffer: Buffer

        Returns
        -------
         asyncio.run() wrapping the set instance method. Can be passed to sync method
        """
        return asyncio.run(self.store.set(key, data_buffer))

    def list(self):
        """
        List keys in store

        Returns
        -------
         asyncio.run() wrapper. Can be passed to sync method
        """

        async def wrapper(gen):
            return [i async for i in gen]

        gen = self.store.list()
        yield from asyncio.run(wrapper(gen))

    def get(self, key, prototype: BufferPrototype):
        """
        Get value of 'key' from store

        Parameters
        ----------
        key: str
        prototype: BufferPrototype

        Returns
        -------
         asyncio.run() wrapper. Can be passed to sync method
        """
        obs = asyncio.run(self.store.get(key, prototype=prototype))
        return obs

    def get_partial_values(self, key_ranges, prototype: BufferPrototype):
        """

        Args:
            key_ranges (List): List of tuples w/ form: [(key, (range_start, range_step)),
                                    (key, (range_start, range_step)),...]
            prototype (BufferPrototype): Hardcoded to default_buffer_prototype()

        Returns
        -------
         asyncio.run() wrapper. Can be passed to sync method
        """
        obs_partial = asyncio.run(
            self.store.get_partial_values(prototype=prototype, key_ranges=key_ranges)
        )
        return obs_partial

    def delete(self, path):
        """delete key from store

        Parameters
        ----------
        path : str
            key to be deleted

        Returns
        -------
        asyncio.run() wrapper. Can be passed to sync method

        """
        return asyncio.run(self.store.delete(path))

    def empty(self):
        """empty: check if store empty

        Returns
        -------
        asyncio.run() wrapper. Can be passed to sync method

        """
        return asyncio.run(self.store.empty())

    def clear(self):
        """clear: delete all keys from store

        Returns
        -------
        asyncio.run() wrapper. Can be passed to sync method

        """
        return asyncio.run(self.store.clear())

    def exists(self, key):
        """exists: Check if key exists in store

        Parameters
        ----------
        key : str


        Returns
        -------
        asyncio.run() wrapper. Can be passed to sync method
        """
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
    """ZarrStoreStateMachine is a state machine for testing the Zarr store.

    This state machine defines a set of rules and invariants to test the behavior of a Zarr store.
    It provides methods for setting, getting, deleting, and clearing data in the store, as well as
    checking various properties of the store.

    Attributes
    ----------
    model : dict[str, bytes]
        A dictionary representing the model of the store, where keys are strings and values are bytes.
    store : SyncStoreWrapper
        An instance of the SyncStoreWrapper class, which wraps a Zarr store implementation.
    prototype : Buffer
        A prototype buffer used for storing data in the store.
    """

    def __init__(self):
        super().__init__()
        self.model: dict[str, bytes] = {}
        self.store = SyncStoreWrapper(MemoryStore(mode="w"))
        self.prototype = default_buffer_prototype()

    @rule(key=paths, data=st.binary(min_size=0, max_size=100))
    def set(self, key: str, data: bytes) -> None:
        """
        Set the given key with the provided data.

        Parameters
        ----------
        key : str
            The key to set.
        data : bytes
            The data to associate with the key.

        Returns
        -------
        None
        """
        note(f"(set) Setting {key!r} with {data}")
        assert not self.store.mode.readonly
        data_buf = Buffer.from_bytes(data)
        self.store.set(key, data_buf)
        self.model[key] = data_buf

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(key=paths, data=st.data())
    def get(self, key, data) -> None:
        """
        Retrieves a value from the store based on the given key.

        Parameters
        ----------
        key : str
            The key to retrieve the value for.
        data : hypothesis.strategies.DataObject
            The data generator used for testing.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the retrieved value from the store does not match the value in the model.
        """
        key = data.draw(
            st.sampled_from(sorted(self.model.keys()))
        )  # hypothesis wants to sample from sorted list
        model_value = self.model[key]
        note(f"(get) model value {model_value.to_bytes()}")
        store_value = self.store.get(key, self.prototype)
        note(f"(get) store value: {store_value.to_bytes()}")
        # to bytes here necessary because data_buf set to model in set()
        assert self.model[key].to_bytes() == (store_value.to_bytes())

    @rule(key=paths, data=st.data())
    def get_invalid_keys(self, key, data) -> None:
        """
        Test case to ensure that calling get() on an invalid key returns None.

        Parameters
        ----------
        key : str
            The key to be tested.
        data : hypothesis.strategies.DataObject
            The data generator used for testing.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the returned value of an invalid key is not None
        """
        note("(get_invalid)")
        assume(key not in self.model.keys())
        assert self.store.get(key, self.prototype) is None

    @precondition(lambda self: len(self.model.keys()) > 0)
    @rule(data=st.data())
    def get_partial_values(self, data) -> None:
        """
        Given a list of tuples with the form:
        [(key, (range_start, range_step)),
         (key, (range_start, range_step)),...]
        Retrieves partial bytes from values, using the byte range associated with each key.

        Parameters
        ----------
        data : hypothesis.strategies.DataObject
            The data generator used for testing.

        Returns
        -------
        None
        """
        key_range = data.draw(key_ranges(keys=st.sampled_from(sorted(self.model.keys()))))
        note(f"(get partial) {key_range=}")
        obs_maybe = self.store.get_partial_values(key_range, self.prototype)
        observed = []

        for obs in obs_maybe:
            assert obs is not None
            observed.append(obs.to_bytes())

        model_vals_ls = []

        for key, byte_range in key_range:
            # model_vals = self.model[key]
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
        """
        Deletes a key from the store and removes it from the model.

        Parameters
        ----------
        data : hypothesis.strategies.DataObject
            A data generator used to draw a key from the model.

        Returns
        -------
        None
        """
        key = data.draw(st.sampled_from(sorted(self.model.keys())))
        note(f"(delete) Deleting {key=}")

        self.store.delete(key)
        del self.model[key]

    @rule()
    def clear(self):
        """
        Clear the store and the model.

        This method clears the contents of the store and the model. It ensures that the store is not in readonly mode,
        and then proceeds to clear both the store and the model. After clearing, it asserts that the number of keys in
        the model and the number of items in the store's list are both zero.

        Returns:
        --------
        None

        Raises:
        -------
            AssertionError: If the store is in readonly mode.
        """
        assert not self.store.mode.readonly
        note("(clear)")
        self.store.clear()
        self.model.clear()

        assert len(self.model.keys()) == len(list(self.store.list())) == 0

    @rule()
    def empty(self) -> None:
        """
        Check if the store is empty.

        This method verifies whether the store is empty or not. It compares the state of the store with the state of the model and ensures that they are either both empty or both not empty.

        Returns:
        --------
        None

        Raises:
        -------
            AssertionError: If the state of the store does not match the state of the model.
        """
        note("(empty)")

        # make sure they either both are or both aren't empty (same state)
        assert self.store.empty() == (not self.model)

    @rule(key=paths)
    def exists(self, key) -> None:
        """
        Check if a given key exists in the store.

        Parameters:
        -----------
        key : str
            The key to check for existence in the store.

        Returns:
        --------
        None

        Raises:
        -------
        AssertionError
            If the key exists in the store but is not found in the model.
        """
        note("(exists)")

        assert self.store.exists(key) == (key in self.model)

    @invariant()
    def check_paths_equal(self) -> None:
        """
        Check if the keys in the store are equal to the keys in the model.

        This method compares the keys obtained from the store with the keys in the model.
        It raises an AssertionError if they are not equal.

        Returns:
        --------
        None
        """
        note("Checking that paths are equal")
        paths = list(self.store.list())

        assert list(self.model.keys()) == paths

    @invariant()
    def check_vals_equal(self) -> None:
        """
        Check if the values in the store are equal to the values in the model.

        This method iterates over the keys in the model and compares the corresponding
        values in the store with the values in the model. If any values are not equal,
        an assertion error is raised.

        Returns:
        --------
        None
        """
        note("Checking values equal")
        for key, _val in self.model.items():
            store_item = self.store.get(key, self.prototype).to_bytes()
            assert self.model[key].to_bytes() == store_item

    @invariant()
    def check_num_keys_equal(self) -> None:
        """
        Check if the number of keys in the model is equal to the number of keys in the store.

        This method compares the number of keys in the model with the number of keys obtained from the store.
        It raises an AssertionError if they are not equal.

        Returns:
        --------
        None
        """
        note("check num keys equal")

        assert len(self.model) == len(list(self.store.list()))

    @invariant()
    def check_keys(self) -> None:
        """
        Check the keys, existence, and emptiness of the store.

        Returns
        -------
        None
        """
        keys = list(self.store.list())

        if len(keys) == 0:
            assert self.store.empty() is True

        elif len(keys) != 0:
            assert self.store.empty() is False

            for key in keys:
                assert self.store.exists(key) is True
        note("checking keys / exists / empty")


StatefulStoreTest = ZarrStoreStateMachine.TestCase
