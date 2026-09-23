from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import zarr
from zarr.core.dtype import DataTypeRegistry, data_type_registry
from zarr.dtype import UInt8
from zarr.registry import Context
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr.core.common import ZarrFormat


class _Byte(UInt8):
    """A data type that is not in the default registry: "test.byte" in Zarr V3, "<u1" in V2."""

    _zarr_v3_name = "test.byte"  # type: ignore[assignment]
    _zarr_v2_names = ("<u1",)  # type: ignore[assignment]


def _context_with_byte() -> Context:
    registry = DataTypeRegistry()
    for key, cls in data_type_registry.contents.items():
        registry.register(key, cls)
    registry.register(_Byte._zarr_v3_name, _Byte)
    return Context(data_types=registry)


def _set_json(store: MemoryStore, key: str, loc: tuple[str, ...], value: str) -> None:
    """Set the member at `loc` of the JSON document at `key`."""
    doc = json.loads(store._store_dict[key].to_bytes())
    parent = doc
    for name in loc[:-1]:
        parent = parent[name]
    parent[loc[-1]] = value
    store._store_dict[key] = zarr.core.buffer.cpu.Buffer.from_bytes(json.dumps(doc).encode())


def _byte_store(zarr_format: ZarrFormat, *, consolidated: bool) -> MemoryStore:
    """
    A group with an array "a" and a group "g" holding an array "b", each with the data type of
    _Byte, written as a uint8 array whose data type is then renamed in the metadata, including in
    the consolidated metadata if `consolidated`.
    """
    store = MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=zarr_format)
    for path in ("a", "g/b"):
        root.create_array(path, shape=(3,), dtype="uint8", fill_value=0)[:] = [0, 128, 255]
    if consolidated:
        zarr.consolidate_metadata(store)
    for path in ("a", "g/b"):
        if zarr_format == 3:
            _set_json(store, f"{path}/zarr.json", ("data_type",), "test.byte")
            if consolidated:
                loc = ("consolidated_metadata", "metadata", path, "data_type")
                _set_json(store, "zarr.json", loc, "test.byte")
        else:
            _set_json(store, f"{path}/.zarray", ("dtype",), "<u1")
            if consolidated:
                _set_json(store, ".zmetadata", ("metadata", f"{path}/.zarray", "dtype"), "<u1")
    return store


def _nested_group_getitem(store: MemoryStore, zarr_format: ZarrFormat, context: Context) -> Any:
    root = zarr.open_group(
        store, zarr_format=zarr_format, mode="r", use_consolidated=False, context=context
    )
    group = root["g"]
    assert isinstance(group, zarr.Group)
    return group["b"]


# each route returns the array it opens
_ROUTES: dict[str, Callable[[MemoryStore, ZarrFormat, Context], Any]] = {
    "open_array": lambda store, zarr_format, context: zarr.open_array(
        store, path="a", zarr_format=zarr_format, mode="r", context=context
    ),
    "open": lambda store, zarr_format, context: zarr.open(
        store, path="a", zarr_format=zarr_format, mode="r", context=context
    ),
    "group getitem": lambda store, zarr_format, context: zarr.open_group(
        store, zarr_format=zarr_format, mode="r", use_consolidated=False, context=context
    )["a"],
    "nested group getitem": _nested_group_getitem,
    "group members": lambda store, zarr_format, context: dict(
        zarr.open_group(
            store, zarr_format=zarr_format, mode="r", use_consolidated=False, context=context
        ).members(max_depth=None)
    )["g/b"],
    "consolidated getitem": lambda store, zarr_format, context: zarr.open_consolidated(
        store, zarr_format=zarr_format, mode="r", context=context
    )["g/b"],
}


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part:UserWarning")
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_open_with_context(route: str, zarr_format: ZarrFormat) -> None:
    """
    An array whose data type is in the context, and not in the default registry, is read with that
    data type through every route that opens it: directly, or as a member of a group opened with
    the context, with or without consolidated metadata.
    """
    store = _byte_store(zarr_format, consolidated=route.startswith("consolidated"))
    array = _ROUTES[route](store, zarr_format, _context_with_byte())
    assert type(array.metadata.dtype if zarr_format == 2 else array.metadata.data_type) is _Byte
    np.testing.assert_array_equal(array[:], np.array([0, 128, 255], dtype=np.uint8))


def test_created_group_inherits_context() -> None:
    """A group created in a group opened with a context reads its members with that context."""
    context = _context_with_byte()
    store = MemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3, context=context)
    sub = root.create_group("sub")
    sub.create_array("x", shape=(1,), dtype="uint8")
    doc: dict[str, Any] = json.loads(store._store_dict["sub/x/zarr.json"].to_bytes())
    doc["data_type"] = "test.byte"
    store._store_dict["sub/x/zarr.json"] = zarr.core.buffer.cpu.Buffer.from_bytes(
        json.dumps(doc).encode()
    )
    assert type(sub["x"].metadata.data_type) is _Byte  # type: ignore[union-attr]


def test_open_without_context() -> None:
    """Without a context, the data type is resolved from the default registry, which lacks it."""
    store = _byte_store(3, consolidated=False)
    with pytest.raises(
        ValueError, match="No Zarr data type found that matches 'test.byte' at /data_type$"
    ):
        zarr.open_array(store, path="a", mode="r")
