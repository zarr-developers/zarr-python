import zarr.core
import zarr.core.attributes
import zarr.storage


def test_put() -> None:
    store = zarr.storage.MemoryStore()
    attrs = zarr.core.attributes.Attributes(
        zarr.Group.from_store(store, attributes={"a": 1, "b": 2})
    )
    attrs.put({"a": 3, "c": 4})
    expected = {"a": 3, "c": 4}
    assert dict(attrs) == expected


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


def test_del_works() -> None:
    store = zarr.storage.MemoryStore()
    z = zarr.create(10, store=store, overwrite=True)
    assert dict(z.attrs) == {}
    z.update_attributes({"a": [3, 4], "c": 4})
    del z.attrs["a"]
    assert "c" in z.attrs
