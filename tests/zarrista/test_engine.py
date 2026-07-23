from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("zarrista")

import zarr
from zarr.errors import UnsupportedEngineError
from zarr.storage import LocalStore

if TYPE_CHECKING:
    from pathlib import Path


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
    # read-only. zarr-python reads have always returned writable arrays, so the
    # facade must copy such a result -- both on the full-box identity path and
    # on a partial (bounding-box) read.
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


def test_zarrista_vlen_read_not_implemented(tmp_path: Path) -> None:
    # zarrista decodes vlen dtypes (e.g. strings) to a `VariableArray`, which
    # only exposes the Arrow C Data interface -- `np.asarray` on it does not
    # raise, it silently produces a wrong-shape 0-d `object` array. The engine
    # must reject this itself rather than let a bogus read through.
    z = zarr.create_array(LocalStore(tmp_path), shape=(4,), chunks=(2,), dtype="str")
    z[:] = np.array(["a", "bb", "ccc", "dddd"], dtype=object)
    ze = zarr.open_array(LocalStore(tmp_path), engine="zarrista")
    with pytest.raises(NotImplementedError, match="VariableArray"):
        ze[:]


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


async def test_zarrista_async_engine_read_write_combinations(tmp_path: Path) -> None:
    # exercises `ZarristaAsyncEngine` directly (over an obstore-backed
    # `ObjectStore`), rather than through the sync `Array`/`ZarristaEngine`
    # path the other tests in this module cover.
    obstore = pytest.importorskip("obstore")
    from zarr.api import asynchronous as async_api
    from zarr.storage import ObjectStore

    store = ObjectStore(obstore.store.LocalStore(prefix=str(tmp_path)))
    z = await async_api.create_array(store=store, shape=(10, 9), chunks=(3, 4), dtype="float32")
    await z.setitem((slice(None), slice(None)), np.arange(90, dtype="float32").reshape(10, 9))
    ze = await async_api.open_array(store=store, engine="zarrista")

    # contiguous read
    out = await ze.getitem((slice(2, 7), slice(1, 5)))
    expected = await z.getitem((slice(2, 7), slice(1, 5)))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
    # full-chunk-aligned write
    await ze.setitem((slice(0, 3), slice(0, 4)), np.zeros((3, 4), dtype="float32"))
    written = await z.getitem((slice(0, 3), slice(0, 4)))
    np.testing.assert_array_equal(np.asarray(written), np.zeros((3, 4), dtype="float32"))
    # partial-chunk RMW write
    await ze.setitem((slice(1, 2), slice(1, 2)), np.float32(99.0))
    assert float(np.asarray(await z.getitem((1, 1)))) == 99.0
    assert float(np.asarray(await z.getitem((0, 0)))) == 0.0  # untouched neighbor
