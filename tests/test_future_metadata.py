"""Tests for the ``_future_metadata`` attribute on array and group classes.

``_future_metadata`` exposes the node's metadata as a ``zarr_metadata``
document model. It is the planned future type of the public ``metadata``
attribute; these tests pin down its type, its consistency with the stored
document, and its cache invalidation across metadata changes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from zarr_metadata.model import (
    ArrayMetadataModelV2,
    ArrayMetadataModelV3,
    ConsolidatedMetadataModelV3,
    GroupMetadataModelV2,
    GroupMetadataModelV3,
)

import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_future_metadata_type(zarr_format: ZarrFormat) -> None:
    array = zarr.create_array(
        MemoryStore(),
        shape=(10, 10),
        chunks=(5, 5),
        dtype="int32",
        zarr_format=zarr_format,
        attributes={"a": 1},
    )
    model = array._future_metadata
    if zarr_format == 2:
        assert isinstance(model, ArrayMetadataModelV2)
        assert model.chunks == (5, 5)
        assert model.dtype == "<i4"
    else:
        assert isinstance(model, ArrayMetadataModelV3)
        assert model.data_type.name == "int32"
        assert model.chunk_grid.configuration == {"chunk_shape": (5, 5)}
    assert model.shape == (10, 10)
    assert model.attributes == {"a": 1}
    assert array._future_metadata is array.async_array._future_metadata


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_future_metadata_matches_stored_document(zarr_format: ZarrFormat) -> None:
    """Parsing the document the array wrote to the store yields the same
    model as ``_future_metadata``.

    The model's own document form is not byte-identical to what zarr-python
    writes (the model normalizes bare-name metadata fields and omits empty
    optional keys), so the comparison happens in model space, where both
    spellings converge.
    """
    store = MemoryStore()
    array = zarr.create_array(
        store, shape=(6,), chunks=(2,), dtype="float64", zarr_format=zarr_format
    )
    async def read(key: str) -> bytes:
        buffer = await array.store_path.store.get(key, prototype=default_buffer_prototype())
        assert buffer is not None
        return buffer.to_bytes()

    model = array._future_metadata
    if zarr_format == 2:
        # zarr-python always writes a `.zattrs` file, so a freshly created
        # array's model has attributes {} (present), not UNSET (absent).
        key_value = {
            ".zarray": zarr.core.sync.sync(read(".zarray")),
            ".zattrs": zarr.core.sync.sync(read(".zattrs")),
        }
        assert ArrayMetadataModelV2.from_key_value(key_value) == model
    else:
        stored_document = json.loads(zarr.core.sync.sync(read("zarr.json")))
        assert ArrayMetadataModelV3.from_json(stored_document) == model


def test_array_future_metadata_cached() -> None:
    array = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="uint8")
    assert array._future_metadata is array._future_metadata


def test_array_future_metadata_updates_on_resize() -> None:
    array = zarr.create_array(MemoryStore(), shape=(4,), chunks=(2,), dtype="uint8")
    before = array._future_metadata
    assert before.shape == (4,)
    array.resize((8,))
    after = array._future_metadata
    assert after.shape == (8,)
    assert before.shape == (4,)  # models are immutable; the old one is unchanged


def test_array_future_metadata_updates_on_update_attributes() -> None:
    array = zarr.create_array(
        MemoryStore(), shape=(4,), chunks=(2,), dtype="uint8", attributes={"a": 1}
    )
    assert array._future_metadata.attributes == {"a": 1}
    array.update_attributes({"b": 2})
    assert array._future_metadata.attributes == {"a": 1, "b": 2}


def test_array_future_metadata_updates_via_attrs_interface() -> None:
    array = zarr.create_array(
        MemoryStore(), shape=(4,), chunks=(2,), dtype="uint8", attributes={"a": 1}
    )
    assert array._future_metadata.attributes == {"a": 1}
    array.attrs["b"] = 2
    assert array._future_metadata.attributes == {"a": 1, "b": 2}
    array.attrs.put({"c": 3})
    assert array._future_metadata.attributes == {"c": 3}


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_picklable_with_populated_cache(zarr_format: ZarrFormat) -> None:
    """The cached model holds PEP 661 sentinels, which refuse to pickle; the
    cache must be excluded from pickled state and re-derived on demand."""
    import pickle

    array = zarr.create_array(
        MemoryStore(), shape=(4,), chunks=(2,), dtype="uint8", zarr_format=zarr_format
    )
    model = array._future_metadata  # populate the cache
    restored = pickle.loads(pickle.dumps(array))
    assert restored._future_metadata == model


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_group_picklable_with_populated_cache(zarr_format: ZarrFormat) -> None:
    import pickle

    group = zarr.create_group(MemoryStore(), zarr_format=zarr_format)
    model = group._future_metadata  # populate the cache
    restored = pickle.loads(pickle.dumps(group))
    assert restored._future_metadata == model


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_group_future_metadata_type(zarr_format: ZarrFormat) -> None:
    group = zarr.create_group(MemoryStore(), zarr_format=zarr_format, attributes={"g": True})
    model = group._future_metadata
    if zarr_format == 2:
        assert isinstance(model, GroupMetadataModelV2)
    else:
        assert isinstance(model, GroupMetadataModelV3)
    assert model.attributes == {"g": True}
    assert group._future_metadata is group._async_group._future_metadata


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_group_future_metadata_updates_on_update_attributes(zarr_format: ZarrFormat) -> None:
    group = zarr.create_group(MemoryStore(), zarr_format=zarr_format, attributes={"a": 1})
    assert group._future_metadata.attributes == {"a": 1}
    group.update_attributes({"b": 2})
    assert group._future_metadata.attributes == {"a": 1, "b": 2}


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
def test_group_future_metadata_consolidated() -> None:
    store = MemoryStore()
    group = zarr.create_group(store, zarr_format=3)
    group.create_array("child", shape=(4,), chunks=(2,), dtype="uint8")
    zarr.consolidate_metadata(store)
    reopened = zarr.open_group(store, mode="r")
    model = reopened._future_metadata
    assert isinstance(model, GroupMetadataModelV3)
    assert isinstance(model.consolidated_metadata, ConsolidatedMetadataModelV3)
    assert set(model.consolidated_metadata.metadata) == {"child"}
    child_model = model.consolidated_metadata.metadata["child"]
    assert isinstance(child_model, ArrayMetadataModelV3)
    assert child_model.shape == (4,)
