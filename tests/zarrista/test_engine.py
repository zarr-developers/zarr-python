from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("zarrista")

import zarr
from zarr.errors import UnsupportedEngineError
from zarr.storage import LocalStore
from zarr.zarrista import ZarristaAsyncEngine

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


def _make(tmp_path: Path) -> zarr.Array[Any]:
    z = zarr.create_array(LocalStore(tmp_path), shape=(10, 9), chunks=(3, 4), dtype="float32")
    z[:, :] = np.arange(90, dtype="float32").reshape(10, 9)
    return z


def test_zarrista_engine_read_write_combinations(tmp_path: Path) -> None:
    z = _make(tmp_path)
    ze = zarr.open_array(LocalStore(tmp_path), engine="zarrista")

    # contiguous read
    np.testing.assert_array_equal(np.asarray(ze[2:7, 1:5]), np.asarray(z[2:7, 1:5]))
    # strided read via bbox + post-index
    np.testing.assert_array_equal(np.asarray(ze[1:9:2, ::3]), np.asarray(z[1:9:2, ::3]))
    # full-chunk-aligned write
    ze[0:3, 0:4] = np.zeros((3, 4), dtype="float32")
    np.testing.assert_array_equal(np.asarray(z[0:3, 0:4]), np.zeros((3, 4), dtype="float32"))
    # partial-chunk RMW write
    ze[1:2, 1:2] = np.float32(99.0)
    assert float(np.asarray(z[1, 1])) == 99.0
    assert float(np.asarray(z[0, 0])) == 0.0  # neighbor in same chunk untouched


def test_zarrista_engine_edge_chunk_full_write(tmp_path: Path) -> None:
    # shape (10, 9) with chunks (3, 4): row-chunk index 3 (rows 9:10) is an
    # edge chunk whose *clipped* extent (1 row) is smaller than the nominal
    # chunk shape (3 rows). Writing the entirety of that chunk's valid
    # (clipped) region is not the same as writing the full nominal chunk, so
    # the engine must still take the RMW path (rather than treating it as a
    # "full chunk" write and handing zarrista a wrongly-shaped buffer).
    z = _make(tmp_path)
    ze = zarr.open_array(LocalStore(tmp_path), engine="zarrista")

    ze[9:10, 0:4] = -np.ones((1, 4), dtype="float32")
    np.testing.assert_array_equal(np.asarray(z[9:10, 0:4]), -np.ones((1, 4), dtype="float32"))
    # the preceding (non-edge) chunk must be untouched
    np.testing.assert_array_equal(
        np.asarray(z[6:9, 0:4]), np.arange(90, dtype="float32").reshape(10, 9)[6:9, 0:4]
    )


def test_zarrista_reads_are_writable(tmp_path: Path) -> None:
    # zarrista's `Tensor` wraps Rust-owned memory that `np.asarray` exposes
    # read-only. zarr-python reads have always returned writable arrays, and
    # nothing downstream of the engine copies any more, so the engine itself
    # must -- on a whole-array read as much as on a partial one.
    _make(tmp_path)
    ze = zarr.open_array(LocalStore(tmp_path), engine="zarrista")

    full = np.asarray(ze[:, :])
    assert full.flags.writeable
    partial = np.asarray(ze[2:7, 1:5])
    assert partial.flags.writeable


def test_zarrista_rejects_v2(tmp_path: Path) -> None:
    zarr.create_array(LocalStore(tmp_path), shape=(4,), chunks=(2,), dtype="int8", zarr_format=2)
    with pytest.raises(UnsupportedEngineError, match="v3"):
        zarr.open_array(LocalStore(tmp_path), engine="zarrista")


def test_zarrista_vlen_read_matches_default_engine(tmp_path: Path) -> None:
    # zarrista decodes a vlen dtype to a `VariableLengthTensor`, which the
    # engine exports through `to_numpy()`. Reads must therefore agree with the
    # default engine element for element, both whole and partial -- the shapes
    # are what a raw `np.asarray` on the tensor would get wrong.
    z = zarr.create_array(LocalStore(tmp_path), shape=(4,), chunks=(2,), dtype="str")
    z[:] = np.array(["a", "bb", "ccc", "dddd"], dtype=object)
    ze = zarr.open_array(LocalStore(tmp_path), engine="zarrista")

    np.testing.assert_array_equal(np.asarray(ze[:]), np.asarray(z[:]))
    np.testing.assert_array_equal(np.asarray(ze[1:3]), np.asarray(z[1:3]))


def test_zarrista_sync_rejects_read_missing_chunks_false(tmp_path: Path) -> None:
    # The zarrista engine cannot enforce read_missing_chunks=False (it fills
    # missing chunks instead of raising), so minting a sync engine with that
    # config must fail loudly rather than silently downgrade the semantics.
    from zarr.core.array_spec import ArrayConfig
    from zarr.zarrista._engine import ZarristaHierarchyEngine

    z = _make(tmp_path)
    config = ArrayConfig(order="C", write_empty_chunks=False, read_missing_chunks=False)
    hierarchy = ZarristaHierarchyEngine(LocalStore(tmp_path))
    with pytest.raises(UnsupportedEngineError, match="read_missing_chunks=False"):
        hierarchy.array_engine("", z.metadata, config)


async def test_zarrista_async_rejects_read_missing_chunks_false(tmp_path: Path) -> None:
    # Same fail-loud contract as the sync engine, exercised on the async
    # hierarchy engine's `array_engine` factory.
    from zarr.core.array_spec import ArrayConfig
    from zarr.zarrista._engine import ZarristaAsyncHierarchyEngine

    z = _make(tmp_path)
    config = ArrayConfig(order="C", write_empty_chunks=False, read_missing_chunks=False)
    hierarchy = ZarristaAsyncHierarchyEngine(LocalStore(tmp_path))
    with pytest.raises(UnsupportedEngineError, match="read_missing_chunks=False"):
        hierarchy.array_engine("", z.metadata, config)


async def _async_arrays(tmp_path: Path) -> tuple[Any, Any, npt.NDArray[np.float32]]:
    """A filled array over an obstore-backed store, opened on both engines.

    The async zarrista engine refuses a `LocalStore` (see `translate_store_async`),
    so an `ObjectStore` is the store to exercise it with.
    """
    obstore = pytest.importorskip("obstore")
    from zarr.api import asynchronous as async_api
    from zarr.storage import ObjectStore

    store = ObjectStore(obstore.store.LocalStore(str(tmp_path)))
    z = await async_api.create_array(store=store, shape=(10, 9), chunks=(3, 4), dtype="float32")
    data = np.arange(90, dtype="float32").reshape(10, 9)
    await z.setitem((slice(None), slice(None)), data)
    ze = await async_api.open_array(store=store, engine="zarrista")
    # guard against a silent fallback making the assertions below vacuous
    assert isinstance(ze.engine, ZarristaAsyncEngine)
    return z, ze, data


async def test_zarrista_async_engine_reads(tmp_path: Path) -> None:
    # exercises `ZarristaAsyncEngine` directly, rather than through the sync
    # `Array`/`ZarristaEngine` path the other tests in this module cover. The
    # engine resolves its `LazyArray` off the event loop and hands each zarrista
    # call back to it, so every dialect has to survive that round trip.
    _, ze, data = await _async_arrays(tmp_path)

    contiguous = await ze.getitem((slice(2, 7), slice(1, 5)))
    np.testing.assert_array_equal(np.asarray(contiguous), data[2:7, 1:5])

    strided = await ze.getitem((slice(1, 9, 2), slice(None, None, 3)))
    np.testing.assert_array_equal(np.asarray(strided), data[1:9:2, ::3])

    ortho = await ze.get_orthogonal_selection((np.array([7, 1, 4]), np.array([0, 8])))
    np.testing.assert_array_equal(np.asarray(ortho), data[np.ix_([7, 1, 4], [0, 8])])

    points = await ze.get_coordinate_selection((np.array([9, 0, 3]), np.array([8, 0, 2])))
    np.testing.assert_array_equal(np.asarray(points), data[[9, 0, 3], [8, 0, 2]])

    mask = np.zeros((10, 9), dtype=bool)
    mask[0, 0] = mask[4, 5] = mask[9, 8] = True
    np.testing.assert_array_equal(np.asarray(await ze.get_mask_selection(mask)), data[mask])


async def test_zarrista_async_engine_writes(tmp_path: Path) -> None:
    # `AsyncArray` exposes only basic writes, so this covers the box and
    # read-modify-write tiers of the engine's write path; the fancy write tiers
    # are covered through the sync engine in tests/engine/test_differential.py.
    z, ze, data = await _async_arrays(tmp_path)
    expected = data.copy()

    await ze.setitem((slice(0, 3), slice(0, 4)), np.zeros((3, 4), dtype="float32"))
    expected[0:3, 0:4] = 0.0
    # a partial-chunk write, which the engine has to read-modify-write
    await ze.setitem((slice(4, 6), slice(2, 9)), np.float32(99.0))
    expected[4:6, 2:9] = 99.0
    # strided, so the write scatters pointwise rather than laying down a box
    await ze.setitem((slice(1, 9, 3), slice(None, None, 4)), np.float32(-1.0))
    expected[1:9:3, ::4] = -1.0

    # read back through the *default* engine: the bytes have to be right on
    # disk, not merely round-trip through zarrista
    np.testing.assert_array_equal(np.asarray(await z.getitem((slice(None), slice(None)))), expected)
