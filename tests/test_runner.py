from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING, Any

import pytest

import zarr
from zarr.core.array import Array, AsyncArray
from zarr.core.sync import Runner, SyncRunner
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
