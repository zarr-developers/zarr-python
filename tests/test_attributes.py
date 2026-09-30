import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

import zarr.core
import zarr.core.attributes
import zarr.storage
from tests.conftest import deep_nan_equal
from zarr.core.common import ZarrFormat
from zarr.errors import ZarrFutureWarning

if TYPE_CHECKING:
    from zarr.types import AnyArray


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
    z: zarr.Group | AnyArray
    if group:
        z = zarr.create_group(store)
    else:
        z = zarr.create_array(store=store, shape=10, dtype=int)
    assert dict(z.attrs) == {}
    z.update_attributes({"a": [3, 4], "c": 4})
    del z.attrs["a"]
    assert dict(z.attrs) == {"c": 4}

    z2: zarr.Group | AnyArray
    if group:
        z2 = zarr.open_group(store)
    else:
        z2 = zarr.open_array(store)
    assert dict(z2.attrs) == {"c": 4}


NodeKind = Literal["array", "group"]
WriteMethod = Literal["create", "update"]

NODES: list[tuple[NodeKind, ZarrFormat]] = [
    ("array", 2),
    ("array", 3),
    ("group", 2),
    ("group", 3),
]


def _write_attributes(
    store: zarr.storage.MemoryStore,
    kind: NodeKind,
    zarr_format: ZarrFormat,
    how: WriteMethod,
    attributes: dict[str, Any],
) -> None:
    """Write `attributes` to a new array or group, at creation or by updating it."""
    initial = attributes if how == "create" else None
    node: zarr.Group | AnyArray
    if kind == "array":
        node = zarr.create_array(
            store, shape=(1,), dtype="i1", zarr_format=zarr_format, attributes=initial
        )
    else:
        node = zarr.create_group(store, zarr_format=zarr_format, attributes=initial)
    if how == "update":
        node.update_attributes(attributes)


def _stored_attributes(store: zarr.storage.MemoryStore) -> dict[str, Any]:
    return dict(zarr.open(store, mode="r").attrs)


def _same_json(a: object, b: object) -> bool:
    """Compare as canonical JSON text, so NaN compares equal to NaN."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize(("kind", "zarr_format"), NODES)
@pytest.mark.parametrize("how", ["create", "update"])
@pytest.mark.parametrize(
    ("attributes", "settings", "expected"),
    [
        # valid JSON is written unchanged under every policy
        (
            {"a": [1, 2.5, None, True], "b": {"c": "d"}},
            {},
            {"a": [1, 2.5, None, True], "b": {"c": "d"}},
        ),
        (
            {"a": [1, 2.5, None, True], "b": {"c": "d"}},
            {"attributes.non_string_keys": "raise", "attributes.non_finite_floats": "raise"},
            {"a": [1, 2.5, None, True], "b": {"c": "d"}},
        ),
        # "allow" keeps the current behavior: non-string keys are written as strings
        (
            {"a": {1: "x", None: "y"}},
            {"attributes.non_string_keys": "allow"},
            {"a": {"1": "x", "null": "y"}},
        ),
        # non-finite floats are allowed by default
        ({"a": [np.nan, np.inf], "b": -np.inf}, {}, {"a": [np.nan, np.inf], "b": -np.inf}),
    ],
)
def test_attributes_json_policies_write(
    kind: NodeKind,
    zarr_format: ZarrFormat,
    how: WriteMethod,
    attributes: dict[str, Any],
    settings: dict[str, str],
    expected: dict[str, Any],
) -> None:
    """
    Attributes are written without a warning when they are valid JSON, or when
    the policy for their kind of invalid value is "allow". The pytest config
    turns any unexpected warning into a failure.
    """
    store = zarr.storage.MemoryStore()
    with zarr.config.set(settings):
        _write_attributes(store, kind, zarr_format, how, attributes)
    assert _same_json(_stored_attributes(store), expected)


@pytest.mark.parametrize(("kind", "zarr_format"), NODES)
@pytest.mark.parametrize("how", ["create", "update"])
def test_non_string_attribute_keys_warn_by_default(
    kind: NodeKind, zarr_format: ZarrFormat, how: WriteMethod
) -> None:
    """
    By default a non-string attribute key emits a ZarrFutureWarning that names
    the key, and is still written as a string, as before.
    """
    store = zarr.storage.MemoryStore()
    with pytest.warns(ZarrFutureWarning, match=r"attributes\['a'\]\[1\].*future version"):
        _write_attributes(store, kind, zarr_format, how, {"a": {1: "x"}})
    assert _stored_attributes(store) == {"a": {"1": "x"}}


@pytest.mark.parametrize(("kind", "zarr_format"), NODES)
@pytest.mark.parametrize("how", ["create", "update"])
def test_non_string_attribute_keys_raise(
    kind: NodeKind, zarr_format: ZarrFormat, how: WriteMethod
) -> None:
    """
    With `attributes.non_string_keys` set to "raise", writing a non-string
    attribute key raises a TypeError that names the key.
    """
    store = zarr.storage.MemoryStore()
    with (
        zarr.config.set({"attributes.non_string_keys": "raise"}),
        pytest.raises(TypeError, match=r"attributes\['a'\]\[1\]"),
    ):
        _write_attributes(store, kind, zarr_format, how, {"a": {1: "x"}})


@pytest.mark.parametrize(("kind", "zarr_format"), NODES)
@pytest.mark.parametrize("how", ["create", "update"])
def test_non_finite_attribute_values_warn(
    kind: NodeKind, zarr_format: ZarrFormat, how: WriteMethod
) -> None:
    """
    With `attributes.non_finite_floats` set to "warn", a NaN or infinite
    attribute value emits a ZarrFutureWarning that names it, and is still written.
    """
    store = zarr.storage.MemoryStore()
    with (
        zarr.config.set({"attributes.non_finite_floats": "warn"}),
        pytest.warns(ZarrFutureWarning, match=r"attributes\['a'\]\[0\], attributes\['b'\]"),
    ):
        _write_attributes(store, kind, zarr_format, how, {"a": [np.nan], "b": np.inf})
    assert _same_json(_stored_attributes(store), {"a": [np.nan], "b": np.inf})


@pytest.mark.parametrize(("kind", "zarr_format"), NODES)
@pytest.mark.parametrize("how", ["create", "update"])
def test_non_finite_attribute_values_raise(
    kind: NodeKind, zarr_format: ZarrFormat, how: WriteMethod
) -> None:
    """
    With `attributes.non_finite_floats` set to "raise", writing a NaN or
    infinite attribute value raises a ValueError that names it.
    """
    store = zarr.storage.MemoryStore()
    with (
        zarr.config.set({"attributes.non_finite_floats": "raise"}),
        pytest.raises(ValueError, match=r"attributes\['a'\]"),
    ):
        _write_attributes(store, kind, zarr_format, how, {"a": np.nan})


@pytest.mark.parametrize("option", ["attributes.non_string_keys", "attributes.non_finite_floats"])
def test_invalid_attributes_json_policy(option: str) -> None:
    """An unrecognized policy value raises a ValueError naming the config option."""
    attributes: dict[str, Any] = (
        {"a": {1: "x"}} if option == "attributes.non_string_keys" else {"a": np.nan}
    )
    with (
        zarr.config.set({option: "ignore"}),
        pytest.raises(ValueError, match=f"Invalid value for config option '{option}'"),
    ):
        zarr.create_array(zarr.storage.MemoryStore(), shape=(1,), dtype="i1", attributes=attributes)
