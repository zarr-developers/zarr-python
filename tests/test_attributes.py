import json
from typing import Any

import numpy as np
import pytest

import zarr.core
import zarr.core.attributes
import zarr.storage
from tests.conftest import deep_nan_equal
from zarr.core.common import ZarrFormat


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "data", [{"inf": np.inf, "-inf": -np.inf, "nan": np.nan}, {"a": 3, "c": 4}]
)
def test_put(data: dict[str, Any], zarr_format: ZarrFormat) -> None:
    store = zarr.storage.MemoryStore()
    attrs = zarr.core.attributes.Attributes(zarr.Group.from_store(store, zarr_format=zarr_format))
    attrs.put(data)
    expected = json.loads(json.dumps(data, allow_nan=True))
    assert deep_nan_equal(dict(attrs), expected)


def test_asdict() -> None:
    store = zarr.storage.MemoryStore()
    attrs = zarr.core.attributes.Attributes(
        zarr.Group.from_store(store, attributes={"a": 1, "b": 2})
    )
    result = attrs.asdict()
    assert result == {"a": 1, "b": 2}


def test_update_attributes_preserves_existing() -> None:
    """
    Test that `update_attributes` only updates the specified attributes
    and preserves existing ones.
    """
    store = zarr.storage.MemoryStore()
    z = zarr.create(10, store=store, overwrite=True)
    z.attrs["a"] = []
    z.attrs["b"] = 3
    assert dict(z.attrs) == {"a": [], "b": 3}

    z.update_attributes({"a": [3, 4], "c": 4})
    assert dict(z.attrs) == {"a": [3, 4], "b": 3, "c": 4}


def test_update_empty_attributes() -> None:
    """
    Ensure updating when initial attributes are empty works.
    """
    store = zarr.storage.MemoryStore()
    z = zarr.create(10, store=store, overwrite=True)
    assert dict(z.attrs) == {}
    z.update_attributes({"a": [3, 4], "c": 4})
    assert dict(z.attrs) == {"a": [3, 4], "c": 4}


def test_update_no_changes() -> None:
    """
    Ensure updating when no new or modified attributes does not alter existing ones.
    """
    store = zarr.storage.MemoryStore()
    z = zarr.create(10, store=store, overwrite=True)
    z.attrs["a"] = []
    z.attrs["b"] = 3

    z.update_attributes({})
    assert dict(z.attrs) == {"a": [], "b": 3}


@pytest.mark.parametrize("group", [True, False])
def test_del_works(group: bool) -> None:
    store = zarr.storage.MemoryStore()
    z: zarr.Group | zarr.Array
    if group:
        z = zarr.create_group(store)
    else:
        z = zarr.create_array(store=store, shape=10, dtype=int)
    assert dict(z.attrs) == {}
    z.update_attributes({"a": [3, 4], "c": 4})
    del z.attrs["a"]
    assert dict(z.attrs) == {"c": 4}

    z2: zarr.Group | zarr.Array
    if group:
        z2 = zarr.open_group(store)
    else:
        z2 = zarr.open_array(store)
    assert dict(z2.attrs) == {"c": 4}
