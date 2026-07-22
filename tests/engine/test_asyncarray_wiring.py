from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import zarr
from zarr.abc.engine import Region
from zarr.core.engine import DefaultAsyncArrayEngine
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype, NDArrayLike, NDBuffer
    from zarr.core.metadata import ArrayMetadata


class _SpyEngine:
    """Wraps a real engine, recording regions."""

    def __init__(self, inner: DefaultAsyncArrayEngine) -> None:
        self.inner = inner
        self.read_regions: list[Region] = []
        self.write_regions: list[Region] = []

    async def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> NDArrayLike:
        self.read_regions.append(selection)
        return await self.inner.read_selection(selection, prototype=prototype)

    async def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        self.write_regions.append(selection)
        return await self.inner.write_selection(selection, value, prototype=prototype)

    def with_metadata(self, metadata: ArrayMetadata) -> _SpyEngine:
        return _SpyEngine(self.inner.with_metadata(metadata))


async def test_asyncarray_routes_io_through_engine() -> None:
    # NOTE: the async variant of the spy test. Task 6 gives the sync `Array` its
    # own sync engine; until then the sync `z[...]` path shares this async engine,
    # but this task exercises the async `AsyncArray` methods directly.
    z = zarr.create_array(MemoryStore(), shape=(10,), chunks=(3,), dtype="int16")
    aa = z.async_array
    spy = _SpyEngine(
        DefaultAsyncArrayEngine(store_path=aa.store_path, metadata=aa.metadata, config=aa.config)
    )
    object.__setattr__(aa, "engine", spy)

    await aa.setitem(slice(2, 8), np.arange(6, dtype="int16"))
    data = await aa.getitem(slice(2, 8))

    assert spy.write_regions == [Region(start=(2,), end_exclusive=(8,))]
    assert spy.read_regions == [Region(start=(2,), end_exclusive=(8,))]
    np.testing.assert_array_equal(np.asarray(data), np.arange(6, dtype="int16"))


def test_asyncarray_default_engine_attribute() -> None:
    z = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="int8")
    assert isinstance(z.async_array.engine, DefaultAsyncArrayEngine)


def test_strided_read_preserves_fortran_order() -> None:
    z = zarr.create_array(
        MemoryStore(),
        shape=(8, 8),
        chunks=(4, 4),
        dtype="float64",
        config={"order": "F"},
    )
    z[:, :] = np.asfortranarray(np.arange(64, dtype="float64").reshape(8, 8))
    full = np.asarray(z[:, :])
    strided = np.asarray(z[::2, ::2])
    assert full.flags.f_contiguous
    assert strided.flags.f_contiguous
    np.testing.assert_array_equal(strided, np.arange(64.0).reshape(8, 8)[::2, ::2])
