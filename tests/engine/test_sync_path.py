from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from zarr.abc.engine import ArrayEngine
from zarr.core.engine import DefaultArrayEngine
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.abc.engine import Region
    from zarr.core.buffer import BufferPrototype, NDArrayLike, NDBuffer
    from zarr.core.metadata import ArrayMetadata


def test_array_has_sync_engine() -> None:
    z = zarr.create_array(MemoryStore(), shape=(6,), chunks=(2,), dtype="uint8")
    assert isinstance(z.engine, ArrayEngine)
    assert isinstance(z.engine, DefaultArrayEngine)


def test_array_engine_is_cached() -> None:
    z = zarr.create_array(MemoryStore(), shape=(6,), chunks=(2,), dtype="uint8")
    assert z.engine is z.engine


class _NoLoopEngine:
    """A sync engine that asserts no event loop is running when called."""

    def __init__(self, inner: ArrayEngine) -> None:
        self._inner = inner
        self.calls = 0

    def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> NDArrayLike:
        self.calls += 1
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return self._inner.read_selection(selection, prototype=prototype)

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        self.calls += 1
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return self._inner.write_selection(selection, value, prototype=prototype)

    def with_metadata(self, metadata: ArrayMetadata) -> _NoLoopEngine:
        return _NoLoopEngine(self._inner.with_metadata(metadata))


def test_sync_data_path_runs_without_event_loop_in_caller_thread() -> None:
    z = zarr.create_array(MemoryStore(), shape=(6,), chunks=(2,), dtype="uint8")
    probe = _NoLoopEngine(z.engine)
    object.__setattr__(z, "_engine", probe)  # match the attribute name used in impl

    z[1:5] = np.arange(4, dtype="uint8")
    out = z[1:5]

    assert probe.calls == 2
    np.testing.assert_array_equal(np.asarray(out), np.arange(4, dtype="uint8"))


def test_resize_rebinds_cached_sync_engine() -> None:
    """After `resize`, reads/writes beyond the old bounds must go through an
    engine bound to the new metadata, not a stale cached one."""
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="uint8")
    z[:] = np.arange(4, dtype="uint8")
    assert np.asarray(z[:]).tolist() == [0, 1, 2, 3]

    # force engine resolution before the resize so the cache is populated
    assert isinstance(z.engine, DefaultArrayEngine)

    z.resize((8,))
    z[4:8] = np.arange(4, 8, dtype="uint8")
    out = z[:]

    np.testing.assert_array_equal(np.asarray(out), np.arange(8, dtype="uint8"))
