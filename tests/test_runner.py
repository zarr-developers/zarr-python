from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.core.array import Array, AsyncArray
from zarr.core.sync import Runner, SyncRunner
from zarr.errors import ZarrDeprecationWarning
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Coroutine


async def _coro() -> int:
    await asyncio.sleep(0)
    return 42


def test_sync_runner_runs_coroutine() -> None:
    runner = SyncRunner()
    assert runner.run(_coro()) == 42


def test_sync_runner_is_runner() -> None:
    assert isinstance(SyncRunner(), Runner)


def _make_array() -> Array[Any]:
    return zarr.create_array(store=MemoryStore(), shape=(8,), chunks=(4,), dtype="i4", fill_value=0)


def test_array_has_default_sync_runner() -> None:
    arr = _make_array()
    assert isinstance(arr._runner, SyncRunner)


def test_array_owns_state() -> None:
    arr = _make_array()
    assert arr.metadata is not None
    assert arr.store_path is not None
    assert arr.codec_pipeline is not None


def test_array_accepts_custom_runner() -> None:
    class RecordingRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
            self.calls += 1
            return SyncRunner().run(coro)

    runner = RecordingRunner()
    base = _make_array()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aa = base.async_array
    arr = Array(metadata=aa.metadata, store_path=aa.store_path, config=aa.config, runner=runner)
    assert arr._runner is runner


def test_async_array_property_deprecated() -> None:
    arr = _make_array()
    with pytest.warns(DeprecationWarning, match="async_array is deprecated"):
        aa = arr.async_array
    assert isinstance(aa, AsyncArray)


def test_from_async_array_roundtrip() -> None:
    arr = _make_array()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aa = arr.async_array
    arr2 = Array._from_async_array(aa)
    assert arr2.metadata == arr.metadata
    assert isinstance(arr2._runner, SyncRunner)


def test_getitem_sync_async_equivalence() -> None:
    arr = _make_array()
    arr[:] = np.arange(8, dtype="i4")
    sync_result = arr[2:6]
    async_via_runner = arr._runner.run(arr.getitem_async(slice(2, 6)))
    np.testing.assert_array_equal(sync_result, async_via_runner)


def test_setitem_async_roundtrip() -> None:
    arr = _make_array()
    arr._runner.run(arr.setitem_async(slice(0, 4), np.arange(4, dtype="i4")))
    np.testing.assert_array_equal(arr[0:4], np.arange(4, dtype="i4"))


def test_custom_runner_invoked_on_read() -> None:
    # The runner injected into Array is actually used by sync reads.
    class RecordingRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
            self.calls += 1
            return SyncRunner().run(coro)

    runner = RecordingRunner()
    base = _make_array()
    base[:] = np.arange(8, dtype="i4")
    arr = Array(
        metadata=base.metadata, store_path=base.store_path, config=base.config, runner=runner
    )
    _ = arr[2:6]
    assert runner.calls > 0


def test_resize_async() -> None:
    arr = _make_array()
    arr._runner.run(arr.resize_async((16,)))
    assert arr.shape == (16,)


def test_update_attributes_async() -> None:
    arr = _make_array()
    arr._runner.run(arr.update_attributes_async({"foo": "bar"}))
    assert arr.metadata.attributes["foo"] == "bar"


def test_legacy_constructor_rejects_extra_store_path() -> None:
    base = _make_array()
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", DeprecationWarning)
        aa = base.async_array
    with pytest.raises(TypeError, match="must not also be provided"):
        Array(aa, store_path=base.store_path)


def test_nchunks_initialized_async() -> None:
    arr = _make_array()
    arr[:] = np.arange(8, dtype="i4")
    n = arr._runner.run(arr.nchunks_initialized_async())
    assert n == arr.nchunks_initialized


def test_orthogonal_selection_async_roundtrip() -> None:
    arr = zarr.create_array(
        store=MemoryStore(), shape=(4, 4), chunks=(2, 2), dtype="i4", fill_value=0
    )
    arr[:] = np.arange(16, dtype="i4").reshape(4, 4)
    expected = arr.get_orthogonal_selection(([0, 2], slice(None)))  # type: ignore[arg-type]
    actual = arr._runner.run(arr.get_orthogonal_selection_async(([0, 2], slice(None))))  # type: ignore[arg-type]
    np.testing.assert_array_equal(actual, expected)


def test_coordinate_selection_async_roundtrip() -> None:
    arr = zarr.create_array(
        store=MemoryStore(), shape=(4, 4), chunks=(2, 2), dtype="i4", fill_value=0
    )
    arr[:] = np.arange(16, dtype="i4").reshape(4, 4)
    expected = arr.get_coordinate_selection(([0, 1], [0, 1]))
    actual = arr._runner.run(arr.get_coordinate_selection_async(([0, 1], [0, 1])))
    np.testing.assert_array_equal(actual, expected)


def test_block_selection_async_roundtrip() -> None:
    arr = zarr.create_array(
        store=MemoryStore(), shape=(4, 4), chunks=(2, 2), dtype="i4", fill_value=0
    )
    arr[:] = np.arange(16, dtype="i4").reshape(4, 4)
    expected = arr.get_block_selection((0, 0))
    actual = arr._runner.run(arr.get_block_selection_async((0, 0)))
    np.testing.assert_array_equal(actual, expected)


def test_set_orthogonal_selection_async() -> None:
    arr = zarr.create_array(
        store=MemoryStore(), shape=(4, 4), chunks=(2, 2), dtype="i4", fill_value=0
    )
    arr._runner.run(arr.set_orthogonal_selection_async(([0, 2], slice(None)), 7))  # type: ignore[arg-type]
    expected = arr.get_orthogonal_selection(([0, 2], slice(None)))  # type: ignore[arg-type]
    np.testing.assert_array_equal(expected, np.full((2, 4), 7, dtype="i4"))


def test_legacy_array_from_async_array_constructor() -> None:
    # Array(async_array) is the deprecated legacy construction form. It should
    # still work but emit a DeprecationWarning.
    base = _make_array()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aa = base.async_array  # a real AsyncArray
    with pytest.warns(DeprecationWarning, match="Array\\(async_array\\)"):
        arr = Array(aa)
    assert isinstance(arr, Array)
    assert arr.metadata == aa.metadata
    assert arr.store_path == aa.store_path
    assert isinstance(arr._runner, SyncRunner)


def test_legacy_array_constructor_passes_runner() -> None:
    base = _make_array()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aa = base.async_array
    runner = SyncRunner()
    with pytest.warns(DeprecationWarning, match="Array\\(async_array\\)"):
        arr = Array(aa, runner=runner)
    assert arr._runner is runner


def test_array_constructor_requires_store_path() -> None:
    # Constructing an Array from metadata without a store_path must error.
    arr = _make_array()
    md = arr.metadata
    with pytest.raises(TypeError, match="store_path is required"):
        Array(md)


def test_array_eq_non_array_is_false() -> None:
    # Array.__eq__ returns NotImplemented for non-Array operands; Python then
    # falls back to identity comparison, yielding False.
    arr = _make_array()
    assert (arr == 42) is False
    assert (arr == object()) is False
    assert (arr != object()) is True


def test_array_eq_other_array_true() -> None:
    # Two Arrays viewing the same state compare equal (exercises the True branch).
    arr = _make_array()
    other = Array(metadata=arr.metadata, store_path=arr.store_path, config=arr.config)
    assert arr == other


def test_compressor_v2_returns_without_error() -> None:
    arr2 = zarr.create_array(store={}, shape=(8,), chunks=(4,), dtype="i4", zarr_format=2)
    with pytest.warns(ZarrDeprecationWarning):
        # The value may be a codec or None depending on defaults; just confirm
        # the v2 branch returns rather than raising.
        _ = arr2.compressor


def test_compressor_v3_raises_typeerror() -> None:
    arr3 = zarr.create_array(store={}, shape=(8,), chunks=(4,), dtype="i4", zarr_format=3)
    with (
        pytest.warns(ZarrDeprecationWarning),
        pytest.raises(TypeError, match="not available for Zarr format 3"),
    ):
        _ = arr3.compressor


def test_filters_v2_non_none() -> None:
    # A v2 array created with explicit filters should report them via .filters.
    from numcodecs import Delta

    arr = zarr.create_array(
        store={},
        shape=(8,),
        chunks=(4,),
        dtype="i4",
        zarr_format=2,
        filters=[Delta(dtype="i4")],
    )
    filters = arr.filters
    assert filters == (Delta(dtype="i4"),)


def test_array_open_roundtrip() -> None:
    store = MemoryStore()
    created = zarr.create_array(store=store, shape=(8,), chunks=(4,), dtype="i4", fill_value=0)
    opened = Array.open(store)
    assert isinstance(opened, Array)
    assert opened.metadata == created.metadata
    assert isinstance(opened._runner, SyncRunner)


def test_array_create_roundtrip() -> None:
    # The Array._create classmethod returns a sync Array via _from_async_array.
    store = MemoryStore()
    arr = Array._create(store=store, shape=(8,), dtype="i4", chunk_shape=(4,), zarr_format=3)
    assert isinstance(arr, Array)
    assert arr.shape == (8,)
    assert isinstance(arr._runner, SyncRunner)
