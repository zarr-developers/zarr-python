from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from tests.crud.conftest import array_metadata, filled
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.crud import (
    NodeExistsError,
    create_new_array,
    create_new_group,
    create_overwrite_array,
    create_overwrite_group,
    delete_chunk,
    delete_node,
    list_children,
    read_chunk,
    read_encoded_chunk,
    read_metadata,
    read_region,
    write_chunk,
)
from zarr.errors import NodeNotFoundError

if TYPE_CHECKING:
    from zarr.abc.store import Store

GROUP_META: dict[str, Any] = {"zarr_format": 3, "node_type": "group", "attributes": {"answer": 42}}
GROUP_META_V2: dict[str, Any] = {"zarr_format": 2, "attributes": {"answer": 42}}


# --- node lifecycle ---


async def test_create_new_group(backend: str, store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo", backend=backend)
    assert dict(zarr.open_group(store=store, path="foo", mode="r").attrs) == {"answer": 42}


async def test_create_new_group_existing_raises(backend: str, store: Store) -> None:
    await create_new_group(GROUP_META, store, "foo", backend=backend)
    with pytest.raises(NodeExistsError):
        await create_new_group(GROUP_META, store, "foo", backend=backend)


async def test_create_overwrite_group_replaces_array(backend: str, store: Store) -> None:
    arr = zarr.create_array(store=store, name="foo", shape=(4,), chunks=(2,), dtype="uint8")
    arr[:] = 1
    await create_overwrite_group(GROUP_META, store, "foo", backend=backend)
    assert dict(zarr.open_group(store=store, path="foo", mode="r").attrs) == {"answer": 42}
    assert not await store.exists("foo/c/0")


async def test_create_new_array(backend: str, store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr", backend=backend)
    a = zarr.open_array(store=store, path="arr", mode="r")
    assert a.shape == (8, 8)
    assert a.dtype == np.dtype("uint16")


async def test_create_new_array_v2(backend: str, store: Store) -> None:
    await create_new_array(array_metadata(zarr_format=2), store, "arr", backend=backend)
    assert zarr.open_array(store=store, path="arr", mode="r").metadata.zarr_format == 2


async def test_create_overwrite_array(backend: str, store: Store) -> None:
    zarr.create_group(store=store, path="arr")
    await create_overwrite_array(array_metadata(), store, "arr", backend=backend)
    assert zarr.open_array(store=store, path="arr", mode="r").shape == (8, 8)


async def test_read_metadata(backend: str, store: Store) -> None:
    await create_new_array(array_metadata(), store, "arr", backend=backend)
    observed = await read_metadata(store, "arr", backend=backend)
    raw = await store.get("arr/zarr.json", prototype=default_buffer_prototype())
    assert raw is not None
    assert observed == json.loads(raw.to_bytes())


async def test_read_metadata_missing(backend: str, store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await read_metadata(store, "nope", backend=backend)


async def test_delete_node(backend: str, store: Store) -> None:
    arr = zarr.create_array(store=store, name="doomed", shape=(4,), chunks=(2,), dtype="uint8")
    arr[:] = 1
    await delete_node(store, "doomed", backend=backend)
    assert not await store.exists("doomed/zarr.json")
    assert not await store.exists("doomed/c/0")


async def test_delete_node_missing(backend: str, store: Store) -> None:
    with pytest.raises(NodeNotFoundError):
        await delete_node(store, "nope", backend=backend)


async def test_list_children(backend: str, store: Store) -> None:
    root = zarr.create_group(store=store)
    root.create_group("sub_group", attributes={"kind": "group"})
    root.create_array("sub_array", shape=(4,), chunks=(2,), dtype="uint8")
    by_path = dict(await list_children(store, "", backend=backend))
    assert set(by_path) == {"sub_group", "sub_array"}
    assert by_path["sub_group"]["node_type"] == "group"
    assert by_path["sub_array"]["node_type"] == "array"
    assert not any(p.startswith("/") for p in by_path)


async def test_create_read_delete_v2_group(backend: str, store: Store) -> None:
    await create_new_group(GROUP_META_V2, store, "g2", backend=backend)
    meta = await read_metadata(store, "g2", backend=backend)
    assert meta["zarr_format"] == 2
    with pytest.raises(NodeExistsError):
        await create_new_group(GROUP_META_V2, store, "g2", backend=backend)
    await delete_node(store, "g2", backend=backend)
    with pytest.raises(NodeNotFoundError):
        await read_metadata(store, "g2", backend=backend)


async def test_read_metadata_v2_array(backend: str, store: Store) -> None:
    await create_new_array(array_metadata(zarr_format=2), store, "arr", backend=backend)
    meta = await read_metadata(store, "arr", backend=backend)
    assert meta["zarr_format"] == 2


# --- chunk I/O ---


@pytest.mark.parametrize("dtype", ["uint8", "int32", "float64", "<u2", ">u2"])
async def test_read_chunk_differential(backend: str, store: Store, dtype: str) -> None:
    data, meta = filled(store, dtype=dtype)
    observed = await read_chunk(meta, store, "a", (1, 0), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 0:4])


@pytest.mark.parametrize(
    "compressors", [None, (GzipCodec(),), (ZstdCodec(),), (BloscCodec(cname="lz4"),)]
)
async def test_read_chunk_codecs(backend: str, store: Store, compressors: Any) -> None:
    data, meta = filled(store, compressors=compressors)
    observed = await read_chunk(meta, store, "a", (0, 1), backend=backend)
    np.testing.assert_array_equal(observed, data[0:4, 4:8])


async def test_read_chunk_v2(backend: str, store: Store) -> None:
    data, meta = filled(store, dtype="<u2", zarr_format=2)
    observed = await read_chunk(meta, store, "a", (1, 1), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_read_chunk_v2_fortran_order(backend: str, store: Store) -> None:
    data, meta = filled(store, dtype="uint16", zarr_format=2, order="F")
    observed = await read_chunk(meta, store, "a", (1, 1), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_read_chunk_sharding(backend: str, store: Store) -> None:
    data, meta = filled(store, chunks=(2, 2), shards=(4, 4))
    observed = await read_chunk(meta, store, "a", (1, 1), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_read_chunk_missing_is_fill(backend: str, store: Store) -> None:
    arr = zarr.create_array(
        store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16", fill_value=7
    )
    meta = dict(arr.metadata.to_dict())
    observed = await read_chunk(meta, store, "a", (0, 0), backend=backend)
    np.testing.assert_array_equal(observed, np.full((4, 4), 7, dtype="uint16"))


async def test_read_chunk_metadata_view(backend: str, store: Store) -> None:
    data, meta = filled(store, dtype="uint16", compressors=None)
    view = copy.deepcopy(meta)
    view["data_type"] = "uint8"
    view["shape"] = [8, 16]
    view["chunk_grid"]["configuration"]["chunk_shape"] = [4, 8]
    observed = await read_chunk(view, store, "a", (1, 0), backend=backend)
    np.testing.assert_array_equal(observed, data[4:8, 0:4].view("uint8"))


async def test_read_chunk_readonly(backend: str, store: Store) -> None:
    _, meta = filled(store)
    observed = await read_chunk(meta, store, "a", (0, 0), backend=backend)
    assert not observed.flags.writeable


async def test_write_chunk_differential(backend: str, store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a", backend=backend)
    value = np.arange(16, dtype="uint16").reshape(4, 4)
    await write_chunk(meta, store, "a", (0, 1), value, backend=backend)
    np.testing.assert_array_equal(zarr.open_array(store=store, path="a", mode="r")[0:4, 4:8], value)


async def test_write_chunk_shape_mismatch(backend: str, store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a", backend=backend)
    with pytest.raises(ValueError, match="chunk shape"):
        await write_chunk(
            meta, store, "a", (0, 0), np.zeros((2, 2), dtype="uint16"), backend=backend
        )


async def test_delete_chunk(backend: str, store: Store) -> None:
    _data, meta = filled(store)
    assert await store.exists("a/c/0/0")
    await delete_chunk(meta, store, "a", (0, 0), backend=backend)
    assert not await store.exists("a/c/0/0")


async def test_write_all_fill_chunk_is_dropped(backend: str, store: Store) -> None:
    arr = zarr.create_array(
        store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16", fill_value=0
    )
    meta = dict(arr.metadata.to_dict())
    await write_chunk(meta, store, "a", (0, 0), np.zeros((4, 4), dtype="uint16"), backend=backend)
    assert not await store.exists("a/c/0/0")
    np.testing.assert_array_equal(
        await read_chunk(meta, store, "a", (0, 0), backend=backend),
        np.zeros((4, 4), dtype="uint16"),
    )


async def test_overwrite_chunk_with_fill_removes_it(backend: str, store: Store) -> None:
    _data, meta = filled(store)  # chunk (0,0) exists with nonzero data, fill_value default 0
    assert await store.exists("a/c/0/0")
    await write_chunk(meta, store, "a", (0, 0), np.zeros((4, 4), dtype="uint16"), backend=backend)
    assert not await store.exists("a/c/0/0")


async def test_read_encoded_chunk_matches_store(backend: str, store: Store) -> None:
    _, meta = filled(store)
    raw = await read_encoded_chunk(meta, store, "a", (0, 0), backend=backend)
    expected = await store.get("a/c/0/0", prototype=default_buffer_prototype())
    assert expected is not None
    assert raw == expected.to_bytes()


async def test_read_encoded_chunk_missing_is_none(backend: str, store: Store) -> None:
    arr = zarr.create_array(store=store, name="e", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    meta = dict(arr.metadata.to_dict())
    assert await read_encoded_chunk(meta, store, "e", (0, 0), backend=backend) is None


# --- region I/O ---

SELECTIONS: list[Any] = [
    (slice(None), slice(None)),
    (slice(2, 7), slice(1, 5)),
    (slice(None), 3),
    (5, slice(None)),
    (3, 4),
    (slice(1, 8, 2), slice(None)),
    (slice(None), slice(6, 1, -2)),
    (slice(-3, None), slice(None, -1)),
    ...,
    (..., slice(2, 4)),
    (slice(0, 0), slice(None)),
    (slice(2, 6),),
]


@pytest.mark.parametrize("sel", SELECTIONS)
async def test_read_region_differential(backend: str, store: Store, sel: Any) -> None:
    data, meta = filled(store)
    observed = await read_region(meta, store, "a", sel, backend=backend)
    np.testing.assert_array_equal(observed, data[sel])


async def test_read_region_sharding(backend: str, store: Store) -> None:
    data, meta = filled(store, chunks=(2, 2), shards=(4, 4))
    observed = await read_region(meta, store, "a", (slice(1, 7), slice(3, 8)), backend=backend)
    np.testing.assert_array_equal(observed, data[1:7, 3:8])


async def test_read_region_too_many_indices(backend: str, store: Store) -> None:
    _, meta = filled(store)
    with pytest.raises(IndexError, match="too many indices"):
        await read_region(meta, store, "a", (0, 0, 0), backend=backend)


async def test_read_region_fancy_rejected(backend: str, store: Store) -> None:
    _, meta = filled(store)
    with pytest.raises(TypeError, match="only integers, slices"):
        await read_region(meta, store, "a", ([0, 1], slice(None)), backend=backend)  # type: ignore[arg-type]


async def test_read_region_out_of_bounds(backend: str, store: Store) -> None:
    _, meta = filled(store)
    with pytest.raises(IndexError, match="out of bounds"):
        await read_region(meta, store, "a", (8, slice(None)), backend=backend)
