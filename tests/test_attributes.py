import zarr.core
import zarr.core.attributes
import zarr.storage


def test_put() -> None:
    store = zarr.storage.MemoryStore({}, mode="w")
    attrs = zarr.core.attributes.Attributes(
        zarr.Group.from_store(store, attributes={"a": 1, "b": 2})
    )
    attrs.put({"a": 3, "c": 4})
    expected = {"a": 3, "c": 4}
    assert dict(attrs) == expected


def test_asdict() -> None:
    store = zarr.storage.MemoryStore({}, mode="w")
    attrs = zarr.core.attributes.Attributes(
        zarr.Group.from_store(store, attributes={"a": 1, "b": 2})
    )
    result = attrs.asdict()
    assert result == {"a": 1, "b": 2}
