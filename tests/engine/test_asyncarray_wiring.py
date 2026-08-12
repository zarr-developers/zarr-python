from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

import zarr
from zarr.core.engine import DefaultAsyncArrayEngine
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.abc.engine import SelectionRequest
    from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar, NDBuffer
    from zarr.core.indexing import BasicSelection, Fields
    from zarr.core.metadata import ArrayMetadata


class _SpyEngine:
    """Wraps a real engine, recording the requests it is handed."""

    def __init__(self, inner: DefaultAsyncArrayEngine) -> None:
        self.inner = inner
        self.reads: list[SelectionRequest] = []
        self.writes: list[SelectionRequest] = []

    async def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        self.reads.append(request)
        return await self.inner.read_selection(request, prototype=prototype, out=out, fields=fields)

    async def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        self.writes.append(request)
        return await self.inner.write_selection(request, value, prototype=prototype, fields=fields)

    def with_metadata(self, metadata: ArrayMetadata) -> _SpyEngine:
        return _SpyEngine(self.inner.with_metadata(metadata))


async def test_asyncarray_routes_io_through_engine() -> None:
    # NOTE: the async variant of the spy test. `Array` (the sync facade)
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

    for request in (*spy.writes, *spy.reads):
        # the selection reaches the engine as the user wrote it, tagged with
        # its dialect and the context needed to resolve it
        assert request.kind == "basic"
        assert request.selection == slice(2, 8)
        assert request.shape == (10,)
        assert request.chunk_grid.chunk_shape == (3,)
    assert len(spy.writes) == len(spy.reads) == 1
    np.testing.assert_array_equal(np.asarray(data), np.arange(6, dtype="int16"))


async def test_asyncarray_tags_each_selection_with_its_dialect() -> None:
    # Every accessor has to name the dialect its selection is written in --
    # the engine has no other way to tell an orthogonal tuple from a coordinate
    # one.
    z = zarr.create_array(MemoryStore(), shape=(8, 8), chunks=(4, 4), dtype="int16")
    aa = z.async_array
    spy = _SpyEngine(
        DefaultAsyncArrayEngine(store_path=aa.store_path, metadata=aa.metadata, config=aa.config)
    )
    object.__setattr__(aa, "engine", spy)

    await aa.getitem((slice(None), 1))
    await aa.get_orthogonal_selection((np.array([0, 3]), slice(None)))
    await aa.get_coordinate_selection((np.array([0, 3]), np.array([1, 2])))
    await aa.get_mask_selection(np.zeros((8, 8), dtype=bool))

    # no `block` here: `AsyncArray` exposes no block accessor, so that dialect
    # only ever reaches an engine through the sync `Array`.
    assert [r.kind for r in spy.reads] == ["basic", "orthogonal", "coordinate", "mask"]


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
    # axis (e.g. `arr[:, 0] = v`) has to widen the dimension-dropped value back
    # to the selection's rank. A numpy integer scalar (not a Python int) keeps
    # `__setitem__` routing on the basic facade rather than the orthogonal one,
    # reproducing the original property-test failure.
    expected = np.zeros((3, 3), dtype="int64")
    z = zarr.create_array(MemoryStore(), shape=(3, 3), chunks=(3, 3), dtype="int64")
    z[:, :] = expected
    value = np.array([1, 2, 3], dtype="int64")
    selection = cast("BasicSelection", (slice(None), np.int64(0)))
    z.set_basic_selection(selection, value)
    expected[:, 0] = value
    np.testing.assert_array_equal(np.asarray(z[:, :]), expected)


def test_empty_block_slice_reads_zero_length_box() -> None:
    # Regression: an empty block slice (`blocks[1:0]`) yields a SliceDimIndexer
    # with start > stop, which an engine that turns the block selection into a
    # box must not read as a negative-length one.
    data = np.arange(2, dtype="int64")
    z = zarr.create_array(MemoryStore(), shape=(2,), chunks=(1,), dtype="int64")
    z[:] = data
    result: Any = z.get_block_selection((slice(1, 0),))
    np.testing.assert_array_equal(np.asarray(result), data[2:2])
    assert result.shape == (0,)
