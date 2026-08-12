from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.abc.engine import ArrayEngine, AsyncArrayEngine, SelectionRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.engine import DefaultArrayEngine, DefaultAsyncArrayEngine
from zarr.core.sync import sync
from zarr.errors import ChunkNotFoundError
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.abc.engine import SelectionKind

SHAPE = (10, 9)
CHUNKS = (3, 4)


def _make_array() -> zarr.Array[Any]:
    z = zarr.create_array(MemoryStore(), shape=SHAPE, chunks=CHUNKS, dtype="int32", fill_value=0)
    z[:, :] = np.arange(90, dtype="int32").reshape(SHAPE)
    return z


def _engine(z: zarr.Array[Any]) -> DefaultAsyncArrayEngine:
    return DefaultAsyncArrayEngine(
        store_path=z.async_array.store_path,
        metadata=z.async_array.metadata,
        config=z.async_array.config,
    )


def _request(z: zarr.Array[Any], kind: SelectionKind, selection: Any) -> SelectionRequest:
    return SelectionRequest(
        kind=kind,
        selection=selection,
        shape=z.shape,
        chunk_grid=ChunkGrid.from_metadata(z.async_array.metadata),
    )


def test_default_async_engine_read_write_roundtrip() -> None:
    z = _make_array()
    eng = _engine(z)
    assert isinstance(eng, AsyncArrayEngine)
    proto = default_buffer_prototype()
    request = _request(z, "basic", (slice(2, 7), slice(1, 5)))

    out = sync(eng.read_selection(request, prototype=proto))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(z[2:7, 1:5]))

    new: np.ndarray[Any, Any] = np.full((5, 4), -1, dtype="int32")
    sync(eng.write_selection(request, new, prototype=proto))
    np.testing.assert_array_equal(np.asarray(z[2:7, 1:5]), new)


@pytest.mark.parametrize(
    ("kind", "selection"),
    [
        ("basic", (slice(2, 7), slice(1, 5))),
        ("orthogonal", (np.array([7, 1, 4]), np.array([0, 8]))),
        ("coordinate", (np.array([9, 0, 3]), np.array([8, 0, 2]))),
        ("block", (1, 2)),
    ],
)
def test_default_engine_reads_every_dialect(kind: SelectionKind, selection: Any) -> None:
    """Each dialect reaches the engine as a raw selection and reads what the
    equivalent `Array` accessor reads."""
    z = _make_array()
    data = np.arange(90, dtype="int32").reshape(SHAPE)
    expected = {
        "basic": lambda: data[2:7, 1:5],
        "orthogonal": lambda: data[np.ix_([7, 1, 4], [0, 8])],
        "coordinate": lambda: data[[9, 0, 3], [8, 0, 2]],
        "block": lambda: data[3:6, 8:9],
    }[kind]()
    result = sync(
        _engine(z).read_selection(
            _request(z, kind, selection), prototype=default_buffer_prototype()
        )
    )
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_default_engine_rank_0_basic_read_is_a_scalar() -> None:
    # The protocol mirrors `_get_selection`, including its scalar return, so an
    # engine swap cannot change the type `get_basic_selection` hands back.
    z = _make_array()
    result = sync(
        _engine(z).read_selection(
            _request(z, "basic", (3, 4)), prototype=default_buffer_prototype()
        )
    )
    assert isinstance(result, np.int32)
    assert result == np.arange(90, dtype="int32").reshape(SHAPE)[3, 4]


def test_default_sync_engine_matches_async() -> None:
    z = _make_array()
    eng = DefaultArrayEngine(_engine(z))
    assert isinstance(eng, ArrayEngine)
    request = _request(z, "basic", (slice(None), slice(None)))
    np.testing.assert_array_equal(
        np.asarray(eng.read_selection(request, prototype=default_buffer_prototype())),
        np.asarray(z[:, :]),
    )


def test_with_metadata_rebinds() -> None:
    z = _make_array()
    eng = _engine(z)
    assert eng.with_metadata(z.async_array.metadata) is not eng


def test_read_missing_chunks_false_raises() -> None:
    z = zarr.create_array(
        MemoryStore(),
        shape=(6,),
        chunks=(2,),
        dtype="int16",
        config={"read_missing_chunks": False},
    )
    z[0:2] = np.arange(2, dtype="int16")  # chunks 1 and 2 never written
    with pytest.raises(ChunkNotFoundError):
        sync(
            _engine(z).read_selection(
                _request(z, "basic", (slice(0, 6),)), prototype=default_buffer_prototype()
            )
        )
