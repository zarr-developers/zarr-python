"""Tests for the top-level ``engine`` wiring (Phase A: config + read + create/open).

The engine is a per-array execution setting in ``ArrayConfig``, defaulted from the
``array.engine`` config key. ``"zarr"`` (the default) is the native path; the crud
backend names route data access and creation/open through ``zarr.crud``. The policy
is strict: operations the selected engine cannot express raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import zarr
from zarr.core.array_spec import ArrayConfig
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from pathlib import Path


def _zarrista_available() -> bool:
    try:
        import zarr.crud
        import zarr.zarrista

        zarr.crud.get_backend("zarrista")
    except (ImportError, KeyError):
        return False
    return True


# Engines that route through crud. "reference" is always available; "zarrista"
# needs the package and only ingests LocalStore.
ENGINES = [
    "reference",
    pytest.param(
        "zarrista",
        marks=pytest.mark.skipif(not _zarrista_available(), reason="zarrista not installed"),
    ),
]


@pytest.fixture
def local(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "store")


def _ramp(shape: tuple[int, ...] = (8, 8), dtype: str = "uint16") -> np.ndarray:
    return np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)


# --- config / ArrayConfig ----------------------------------------------------


def test_array_config_engine_defaults_to_zarr() -> None:
    """The engine defaults to 'zarr' (native) and lives in ArrayConfig."""
    assert ArrayConfig.from_dict({}).engine == "zarr"


def test_array_config_engine_from_global_config() -> None:
    """A global array.engine config value is picked up by ArrayConfig.from_dict."""
    with zarr.config.set({"array.engine": "reference"}):
        assert ArrayConfig.from_dict({}).engine == "reference"


def test_unknown_engine_raises(local: LocalStore) -> None:
    """Selecting an unregistered engine raises when an operation needs the backend."""
    with pytest.raises(KeyError):
        zarr.create_array(
            store=local, name="a", shape=(4, 4), chunks=(2, 2), dtype="uint8", engine="nope"
        )


# --- read path ---------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_getitem_via_engine_matches_native(local: LocalStore, engine: str) -> None:
    """Reading through an engine returns the same values as the native path."""
    data = _ramp()
    native = zarr.create_array(store=local, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    native[:] = data

    arr = zarr.open_array(store=local, path="a", engine=engine)
    np.testing.assert_array_equal(arr[:], data)
    np.testing.assert_array_equal(arr[2:6, 1:5], data[2:6, 1:5])


@pytest.mark.parametrize("engine", ENGINES)
def test_scalar_getitem_via_engine(local: LocalStore, engine: str) -> None:
    """A full-integer selection returns a scalar, matching native semantics."""
    data = _ramp()
    native = zarr.create_array(store=local, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    native[:] = data

    arr = zarr.open_array(store=local, path="a", engine=engine)
    result = arr[3, 4]
    assert np.ndim(result) == 0
    assert result == data[3, 4]


@pytest.mark.parametrize("engine", ENGINES)
def test_advanced_indexing_raises_under_engine(local: LocalStore, engine: str) -> None:
    """Orthogonal/coordinate indexing under a non-native engine raises (strict policy)."""
    native = zarr.create_array(store=local, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    native[:] = _ramp()
    arr = zarr.open_array(store=local, path="a", engine=engine)

    with pytest.raises(NotImplementedError):
        arr.oindex[[0, 2], :]  # type: ignore[index]
    with pytest.raises(NotImplementedError):
        arr.vindex[[0, 1], [2, 3]]


# --- creation / open ---------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_create_array_via_engine_roundtrips(local: LocalStore, engine: str) -> None:
    """An array created, written, and read entirely through an engine round-trips,
    and a native reader sees the same data."""
    data = _ramp()
    arr = zarr.create_array(
        store=local, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16", engine=engine
    )
    arr[:] = data  # write through the engine

    assert arr.metadata.shape == (8, 8)
    np.testing.assert_array_equal(arr[:], data)  # read through the engine
    np.testing.assert_array_equal(zarr.open_array(store=local, path="a")[:], data)  # native read


def test_open_array_carries_engine(local: LocalStore) -> None:
    """open_array(engine=...) sets the engine on the returned array's config."""
    zarr.create_array(store=local, name="a", shape=(4, 4), chunks=(2, 2), dtype="uint8")
    arr = zarr.open_array(store=local, path="a", engine="reference")
    assert arr._async_array.config.engine == "reference"


@pytest.mark.skipif(not _zarrista_available(), reason="zarrista not installed")
def test_engine_unsupported_store_raises_on_create() -> None:
    """Creating under zarrista on a MemoryStore (un-ingestable) raises UnsupportedStoreError."""
    from zarr.zarrista import UnsupportedStoreError

    with pytest.raises(UnsupportedStoreError):
        zarr.create_array(
            store=MemoryStore(),
            name="a",
            shape=(4, 4),
            chunks=(2, 2),
            dtype="uint8",
            engine="zarrista",
        )


# --- write path (Phase B) ----------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_setitem_via_engine_matches_native(local: LocalStore, engine: str) -> None:
    """Writing a chunk-aligned region through an engine matches a native write."""
    data = _ramp()
    zarr.create_array(store=local, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    arr = zarr.open_array(store=local, path="a", engine=engine)
    arr[:] = data

    np.testing.assert_array_equal(zarr.open_array(store=local, path="a")[:], data)


@pytest.mark.parametrize("engine", ENGINES)
def test_partial_chunk_write_via_engine(local: LocalStore, engine: str) -> None:
    """A region that partially covers boundary chunks read-modify-writes correctly."""
    base = _ramp()
    native = zarr.create_array(store=local, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    native[:] = base

    arr = zarr.open_array(store=local, path="a", engine=engine)
    patch = np.full((4, 5), 999, dtype="uint16")
    arr[2:6, 1:6] = patch  # crosses all four chunk boundaries partially

    expected = base.copy()
    expected[2:6, 1:6] = patch
    np.testing.assert_array_equal(zarr.open_array(store=local, path="a")[:], expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_advanced_write_raises_under_engine(local: LocalStore, engine: str) -> None:
    """Orthogonal/coordinate writes under a non-native engine raise (strict policy)."""
    zarr.create_array(store=local, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    arr = zarr.open_array(store=local, path="a", engine=engine)

    with pytest.raises(NotImplementedError):
        arr.oindex[[0, 2], :] = 5  # type: ignore[index]
    with pytest.raises(NotImplementedError):
        arr.vindex[[0, 1], [2, 3]] = 5


# --- re-entrancy guard -------------------------------------------------------


def test_global_reference_engine_does_not_recurse(local: LocalStore) -> None:
    """With array.engine='reference' set globally, a read completes (no infinite recursion).

    The reference backend's internal AsyncArray must be pinned to engine='zarr'.
    """
    data = _ramp((4, 4))
    with zarr.config.set({"array.engine": "reference"}):
        arr = zarr.create_array(store=local, name="a", shape=(4, 4), chunks=(2, 2), dtype="uint16")
        native = zarr.open_array(store=local, path="a", engine="zarr")
        native[:] = data
        np.testing.assert_array_equal(arr[:], data)
