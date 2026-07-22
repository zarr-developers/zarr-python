from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

import zarr
from zarr.abc.engine import Region
from zarr.core.engine import DefaultAsyncArrayEngine
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype, NDArrayLike, NDBuffer
    from zarr.core.indexing import BasicSelection
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
    # NOTE: the async variant of the spy test. `Array` (the sync facade) now
    # resolves and calls its own sync engine (see tests/engine/test_sync_path.py);
    # this test exercises the async `AsyncArray` methods directly, via its
    # separate async engine.
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


def test_basic_set_integer_axis_widens_value() -> None:
    # Regression: a basic write whose dropped integer axis is *not* the leading
    # axis (e.g. `arr[:, 0] = v`) took the full-box fast path, which broadcast the
    # dimension-dropped value straight into the ndim-preserving box -- (3,) could
    # not broadcast to box shape (3, 1). A numpy integer scalar keeps the routing
    # in the basic (not orthogonal) facade, matching the property-test example.
    expected = np.zeros((3, 3), dtype="int64")
    z = zarr.create_array(MemoryStore(), shape=(3, 3), chunks=(3, 3), dtype="int64")
    z[:, :] = expected
    value = np.array([1, 2, 3], dtype="int64")
    # a numpy integer scalar (not a Python int) keeps `__setitem__` routing on the
    # basic facade rather than the orthogonal one, reproducing the failing example.
    selection = cast("BasicSelection", (slice(None), np.int64(0)))
    z.set_basic_selection(selection, value)
    expected[:, 0] = value
    np.testing.assert_array_equal(np.asarray(z[:, :]), expected)


def test_empty_block_slice_reads_zero_length_box() -> None:
    # Regression: an empty block slice (`blocks[1:0]`) produced a
    # SliceDimIndexer with start > stop, which `_block_region` mapped to a
    # negative-length box (start=1, end_exclusive=0) and crashed with
    # "negative dimensions are not allowed".
    data = np.arange(2, dtype="int64")
    z = zarr.create_array(MemoryStore(), shape=(2,), chunks=(1,), dtype="int64")
    z[:] = data
    result = np.asarray(z.get_block_selection((slice(1, 0),)))
    np.testing.assert_array_equal(result, data[2:2])
    assert result.shape == (0,)
