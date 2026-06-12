from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip(
    "_zarrs_bindings", reason="zarrs-bindings is not installed", exc_type=ImportError
)

import zarr
from tests.zarrs.conftest import array_metadata
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec
from zarr.core.buffer.core import default_buffer_prototype
from zarr.zarrs import (
    create_new_array,
    decode_chunk,
    encode_chunk,
    erase_chunk,
    read_encoded_chunk,
)

if TYPE_CHECKING:
    from zarr.abc.store import Store


def _filled(store: Store, **kwargs: Any) -> tuple[np.ndarray[Any, np.dtype[Any]], dict[str, Any]]:
    """Create an 8x8 array named 'a' via zarr-python, fill it with a ramp, and
    return (data, metadata_document)."""
    params: dict[str, Any] = {"shape": (8, 8), "chunks": (4, 4), "dtype": "uint16"} | kwargs
    arr = zarr.create_array(store=store, name="a", **params)
    data = np.arange(64, dtype=params["dtype"]).reshape(8, 8)
    arr[:, :] = data
    doc = dict(arr.metadata.to_dict())
    if params.get("zarr_format") == 2:
        # v2 attributes live in .zattrs, not in the .zarray document
        doc.pop("attributes", None)
    return data, doc


@pytest.mark.parametrize("dtype", ["uint8", "int32", "float64"])
async def test_decode_chunk_differential(store: Store, dtype: str) -> None:
    data, meta = _filled(store, dtype=dtype)
    observed = await decode_chunk(meta, store, "a", (1, 0))
    np.testing.assert_array_equal(observed, data[4:8, 0:4])


@pytest.mark.parametrize(
    "compressors", [None, (GzipCodec(),), (ZstdCodec(),), (BloscCodec(cname="lz4"),)]
)
async def test_decode_chunk_codecs(store: Store, compressors: Any) -> None:
    data, meta = _filled(store, compressors=compressors)
    observed = await decode_chunk(meta, store, "a", (0, 1))
    np.testing.assert_array_equal(observed, data[0:4, 4:8])


async def test_decode_chunk_v2(store: Store) -> None:
    data, meta = _filled(store, zarr_format=2)
    observed = await decode_chunk(meta, store, "a", (1, 1))
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_decode_chunk_v2_big_endian(store: Store) -> None:
    data, meta = _filled(store, dtype=">u2", zarr_format=2)
    observed = await decode_chunk(meta, store, "a", (1, 1))
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_encode_chunk_v2_big_endian(store: Store) -> None:
    meta = array_metadata(dtype=">u2", zarr_format=2)
    await create_new_array(meta, store, "a")
    value = np.arange(16, dtype="uint16").reshape(4, 4)
    await encode_chunk(meta, store, "a", (0, 1), value)
    arr = zarr.open_array(store=store, path="a", mode="r")
    np.testing.assert_array_equal(arr[0:4, 4:8], value)


async def test_decode_chunk_readonly(store: Store) -> None:
    _, meta = _filled(store)
    observed = await decode_chunk(meta, store, "a", (0, 0))
    assert not observed.flags.writeable


async def test_decode_chunk_sharding(store: Store) -> None:
    # with sharding, the metadata chunk grid is the shard grid
    data, meta = _filled(store, chunks=(2, 2), shards=(4, 4))
    observed = await decode_chunk(meta, store, "a", (1, 1))
    np.testing.assert_array_equal(observed, data[4:8, 4:8])


async def test_decode_chunk_missing_returns_fill_value(store: Store) -> None:
    arr = zarr.create_array(
        store=store, name="a", shape=(8, 8), chunks=(4, 4), dtype="uint16", fill_value=7
    )
    meta = dict(arr.metadata.to_dict())
    observed = await decode_chunk(meta, store, "a", (0, 0))
    np.testing.assert_array_equal(observed, np.full((4, 4), 7, dtype="uint16"))


async def test_decode_chunk_selection_not_implemented(store: Store) -> None:
    _, meta = _filled(store)
    with pytest.raises(NotImplementedError):
        await decode_chunk(meta, store, "a", (0, 0), selection=(slice(0, 2), slice(0, 2)))


async def test_decode_chunk_metadata_view(store: Store) -> None:
    # the read-only-view case: decode with a metadata document the store never saw
    data, meta = _filled(store, dtype="uint16", compressors=None)
    view = copy.deepcopy(meta)
    view["data_type"] = "uint8"
    view["shape"] = [8, 16]
    view["chunk_grid"]["configuration"]["chunk_shape"] = [4, 8]
    observed = await decode_chunk(view, store, "a", (1, 0))
    np.testing.assert_array_equal(observed, data[4:8, 0:4].view("uint8"))


async def test_encode_chunk_differential(store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a")
    value = np.arange(16, dtype="uint16").reshape(4, 4)
    await encode_chunk(meta, store, "a", (0, 1), value)
    arr = zarr.open_array(store=store, path="a", mode="r")
    np.testing.assert_array_equal(arr[0:4, 4:8], value)


async def test_encode_chunk_shape_mismatch(store: Store) -> None:
    meta = array_metadata()
    await create_new_array(meta, store, "a")
    with pytest.raises(ValueError, match="chunk shape"):
        await encode_chunk(meta, store, "a", (0, 0), np.zeros((2, 2), dtype="uint16"))


async def test_read_encoded_chunk_matches_store(store: Store) -> None:
    _, meta = _filled(store)
    raw = await read_encoded_chunk(meta, store, "a", (0, 0))
    expected = await store.get("a/c/0/0", prototype=default_buffer_prototype())
    assert expected is not None
    assert raw == expected.to_bytes()


async def test_read_encoded_chunk_missing_returns_none(store: Store) -> None:
    arr = zarr.create_array(store=store, name="empty", shape=(8, 8), chunks=(4, 4), dtype="uint16")
    meta = dict(arr.metadata.to_dict())
    assert await read_encoded_chunk(meta, store, "empty", (0, 0)) is None


async def test_erase_chunk(store: Store) -> None:
    _, meta = _filled(store)
    assert await store.exists("a/c/0/0")
    await erase_chunk(meta, store, "a", (0, 0))
    assert not await store.exists("a/c/0/0")
    arr = zarr.open_array(store=store, path="a", mode="r")
    np.testing.assert_array_equal(arr[0:4, 0:4], np.zeros((4, 4), dtype="uint16"))
