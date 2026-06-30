"""Tests for the zarrista CRUD backend (`zarr.zarrista`).

The zarrista backend binds the Rust `zarrs` crate through the `zarrista`
package. It accelerates chunk I/O and, unlike the in-tree zarrs bindings, has no
generic Python-store callback bridge: it can only operate on stores it can hand
to zarrista natively. Every method therefore rejects stores it cannot ingest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr.crud import ReferenceBackend
from zarr.zarrista import UnsupportedStoreError, ZarristaBackend

from .conftest import array_metadata, ramp

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from zarr.abc.store import Store

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


async def _create_filled(
    backend: ZarristaBackend, store: Store, data: np.ndarray[Any, np.dtype[Any]]
) -> dict[str, Any]:
    """Create array 'a', write its single-region data chunk-by-chunk, return metadata."""
    meta = array_metadata(shape=data.shape, chunks=(4, 4), dtype=str(data.dtype))
    await backend.create_array(store, "a", meta, overwrite=False)
    for i in range(0, data.shape[0], 4):
        for j in range(0, data.shape[1], 4):
            chunk = np.ascontiguousarray(data[i : i + 4, j : j + 4])
            await backend.write_chunk(store, "a", meta, (i // 4, j // 4), chunk.tobytes())
    return meta


# --- The headline requirement: error on un-ingestable stores ----------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda be, s: be.read_metadata(s, "a"), id="read_metadata"),
        pytest.param(
            lambda be, s: be.create_array(s, "a", array_metadata(), overwrite=False),
            id="create_array",
        ),
        pytest.param(
            lambda be, s: be.create_group(
                s, "a", {"zarr_format": 3, "node_type": "group"}, overwrite=False
            ),
            id="create_group",
        ),
        pytest.param(
            lambda be, s: be.read_chunk(s, "a", array_metadata(), (0, 0)), id="read_chunk"
        ),
        pytest.param(
            lambda be, s: be.read_subset(s, "a", array_metadata(), (0, 0), (4, 4)),
            id="read_subset",
        ),
        pytest.param(
            lambda be, s: be.write_chunk(s, "a", array_metadata(), (0, 0), b"\x00" * 32),
            id="write_chunk",
        ),
        pytest.param(
            lambda be, s: be.delete_chunk(s, "a", array_metadata(), (0, 0)), id="delete_chunk"
        ),
        pytest.param(lambda be, s: be.delete_node(s, "a"), id="delete_node"),
        pytest.param(lambda be, s: be.list_children(s, ""), id="list_children"),
    ],
)
async def test_unsupported_store_raises(
    memory_store: Store, call: Callable[[ZarristaBackend, Store], Awaitable[object]]
) -> None:
    """Every backend method rejects a store zarrista cannot ingest (e.g. MemoryStore)."""
    backend = ZarristaBackend()
    with pytest.raises(UnsupportedStoreError):
        await call(backend, memory_store)


def test_unsupported_store_error_is_type_error() -> None:
    """UnsupportedStoreError is a TypeError so generic store-type handling catches it."""
    assert issubclass(UnsupportedStoreError, TypeError)


async def test_local_store_is_ingestable(local_store: Store) -> None:
    """A LocalStore is accepted: a create + read round-trips without raising."""
    backend = ZarristaBackend()
    meta = array_metadata()
    await backend.create_array(local_store, "a", meta, overwrite=False)
    assert (await backend.read_metadata(local_store, "a"))["node_type"] == "array"


# --- Correctness on an ingestable store -------------------------------------


async def test_chunk_roundtrip_matches_numpy(local_store: Store) -> None:
    """A chunk written through zarrista reads back equal to the source array."""
    backend = ZarristaBackend()
    data = ramp((8, 8))
    meta = await _create_filled(backend, local_store, data)

    raw = await backend.read_chunk(local_store, "a", meta, (1, 1))
    chunk = np.frombuffer(raw, dtype="uint16").reshape(4, 4)
    np.testing.assert_array_equal(chunk, data[4:8, 4:8])


async def test_subset_read_matches_numpy(local_store: Store) -> None:
    """A cross-chunk region read through zarrista matches the numpy slice."""
    backend = ZarristaBackend()
    data = ramp((8, 8))
    meta = await _create_filled(backend, local_store, data)

    raw = await backend.read_subset(local_store, "a", meta, (2, 1), (4, 5))
    region = np.frombuffer(raw, dtype="uint16").reshape(4, 5)
    np.testing.assert_array_equal(region, data[2:6, 1:6])


async def test_reads_agree_with_reference_backend(local_store: Store) -> None:
    """zarrista and the pure-Python reference backend return identical chunk bytes."""
    zarrista_be = ZarristaBackend()
    reference_be = ReferenceBackend()
    data = ramp((8, 8))
    meta = await _create_filled(zarrista_be, local_store, data)

    for coords in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert await zarrista_be.read_chunk(
            local_store, "a", meta, coords
        ) == await reference_be.read_chunk(local_store, "a", meta, coords)


async def test_delete_chunk_then_read_returns_fill_value(local_store: Store) -> None:
    """After delete_chunk the region reads back as the fill value (0)."""
    backend = ZarristaBackend()
    data = ramp((8, 8)) + 1  # avoid fill-value (0) so the chunk is actually stored
    meta = await _create_filled(backend, local_store, data)

    await backend.delete_chunk(local_store, "a", meta, (0, 0))
    raw = await backend.read_chunk(local_store, "a", meta, (0, 0))
    np.testing.assert_array_equal(np.frombuffer(raw, dtype="uint16").reshape(4, 4), 0)


# --- Node-level metadata operations -----------------------------------------


async def test_create_group_and_list_children(local_store: Store) -> None:
    """create_group + create_array then list_children reports the child array."""
    backend = ZarristaBackend()
    await backend.create_group(
        local_store, "", {"zarr_format": 3, "node_type": "group"}, overwrite=False
    )
    await backend.create_array(local_store, "child", array_metadata(), overwrite=False)

    children = await backend.list_children(local_store, "")
    assert [name for name, _ in children] == ["child"]


async def test_delete_node_removes_array(local_store: Store) -> None:
    """delete_node removes the array; a subsequent read_metadata raises NodeNotFoundError."""
    from zarr.errors import NodeNotFoundError

    backend = ZarristaBackend()
    await backend.create_array(local_store, "a", array_metadata(), overwrite=False)
    await backend.delete_node(local_store, "a")
    with pytest.raises(NodeNotFoundError):
        await backend.read_metadata(local_store, "a")


# --- Zarr V2 (the V3-compatible subset zarrs supports) ----------------------


async def test_v2_array_chunk_roundtrips(local_store: Store) -> None:
    """zarrs/zarrista support the V3-compatible subset of Zarr V2: a v2 array
    written through zarrista reads back correctly, and zarr-python agrees."""
    import zarr

    backend = ZarristaBackend()
    data = ramp((8, 8)) + 1  # avoid fill-value so chunks are stored
    meta = array_metadata(zarr_format=2, shape=(8, 8), chunks=(4, 4), dtype="uint16")
    await backend.create_array(local_store, "a", meta, overwrite=False)
    for i in (0, 4):
        for j in (0, 4):
            chunk = np.ascontiguousarray(data[i : i + 4, j : j + 4])
            await backend.write_chunk(local_store, "a", meta, (i // 4, j // 4), chunk.tobytes())

    raw = await backend.read_chunk(local_store, "a", meta, (1, 1))
    np.testing.assert_array_equal(np.frombuffer(raw, dtype="uint16").reshape(4, 4), data[4:8, 4:8])
    # The bytes zarrista wrote are a standard v2 array zarr-python can read.
    np.testing.assert_array_equal(zarr.open_array(store=local_store, path="a", mode="r")[:], data)


# --- Registration via the crud registry -------------------------------------


def test_importing_registers_backend() -> None:
    """Importing zarr.zarrista registers it under the 'zarrista' name."""
    import zarr.crud
    import zarr.zarrista

    assert isinstance(zarr.crud.get_backend("zarrista"), ZarristaBackend)


def test_registry_lazily_imports_backend() -> None:
    """get_backend('zarrista') self-imports the package, like the 'zarrs' backend.

    Run in a subprocess so the resolution starts from a process that has only
    imported `zarr.crud`, never `zarr.zarrista`.
    """
    import subprocess
    import sys

    code = (
        "import zarr.crud;"
        "be = zarr.crud.get_backend('zarrista');"
        "assert type(be).__name__ == 'ZarristaBackend', type(be)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
