from __future__ import annotations

from typing import Any, Literal, TypedDict

import pytest
from typing_extensions import ReadOnly

from src.zarr.core.type_check import check_type
from zarr.core.common import NamedConfig
from zarr.core.dtype.common import DTypeConfig_V2, DTypeSpec_V2, DTypeSpec_V3, StructuredName_V2
from zarr.core.dtype.npy.common import DateTimeUnit
from zarr.core.dtype.npy.structured import StructuredJSON_V2


# --- Sample TypedDicts for testing ---
class Address(TypedDict):
    street: str
    zipcode: int


class User(TypedDict):
    id: int
    name: str
    address: Address
    tags: list[str]


class PartialUser(TypedDict, total=False):
    id: int
    name: str


def test_int_valid() -> None:
    """
    Test that an integer matches the int type.
    """
    result = check_type(42, int)
    assert result.success


def test_int_invalid() -> None:
    """
    Test that a string does not match the int type.
    """
    result = check_type("oops", int)
    assert not result.success
    assert "expected int but got str" in result.errors[0]


def test_float_valid() -> None:
    """
    Test that a float matches the float type.
    """
    result = check_type(3.14, float)
    assert result.success


def test_float_invalid() -> None:
    """
    Test that a string does not match the float type.
    """
    result = check_type("oops", float)
    assert not result.success
    assert "expected float but got str" in result.errors[0]


def test_tuple_valid() -> None:
    """
    Test that a tuple of (int, str, None) matches the corresponding Tuple type.
    """
    result = check_type((1, "x", None), tuple[int, str, None])
    assert result.success


def test_tuple_invalid() -> None:
    """
    Test that a tuple with an incorrect element type fails type checking.
    """
    result = check_type((1, "x", 5), tuple[int, str, None])
    assert not result.success
    assert "expected None but got int" in result.errors[0]


def test_list_valid() -> None:
    """
    Test that a list of int | None matches list[int | None].
    """
    result = check_type([1, None, 3], list[int | None])
    assert result.success


def test_list_invalid() -> None:
    """
    Test that a list with an invalid element type fails type checking.
    """
    result = check_type([1, "oops", 3], list[int])
    assert not result.success
    assert "expected int but got str" in result.errors[0]


def test_dict_valid() -> None:
    """
    Test that a dict with string keys and int values matches dict[str, int].
    """
    result = check_type({"a": 1, "b": 2}, dict[str, int])
    assert result.success


def test_dict_invalid() -> None:
    """
    Test that a dict with a value of incorrect type fails type checking.
    """
    result = check_type({"a": 1, "b": "oops"}, dict[str, int])
    assert not result.success
    assert "expected int but got str" in result.errors[0]


def test_dict_any_valid() -> None:
    """
    Test that a dict with keys of type Any passes type checking.
    """
    result = check_type({1: "x", "y": 2}, dict[Any, Any])
    assert result.success


def test_typeddict_valid() -> None:
    """
    Test that a nested TypedDict with correct types passes type checking.
    """
    user: User = {
        "id": 1,
        "name": "Alice",
        "address": {"street": "Main St", "zipcode": 12345},
        "tags": ["x", "y"],
    }
    result = check_type(user, User)
    assert result.success


def test_typeddict_invalid() -> None:
    """
    Test that a nested TypedDict with an incorrect field type fails type checking.
    """
    bad_user = {
        "id": 1,
        "name": "Alice",
        "address": {"street": "Main St", "zipcode": "oops"},
        "tags": ["x", "y"],
    }
    result = check_type(bad_user, User)
    assert not result.success
    assert "expected int but got str" in "".join(result.errors)


def test_typeddict_fail_missing_required() -> None:
    """
    Test that a nested TypedDict missing a required key raises type check failure.
    """
    bad_user = {
        "id": 1,
        "name": "Alice",
        "address": {"street": "Main St"},  # missing zipcode
        "tags": ["x"],
    }
    result = check_type(bad_user, User)
    assert not result.success
    assert "missing required key 'zipcode'" in "".join(result.errors)


def test_typeddict_partial_total_false_pass() -> None:
    """
    Test that a TypedDict with total=False allows missing optional keys.
    """
    result = check_type({}, PartialUser)
    assert result.success


def test_typeddict_partial_total_false_fail() -> None:
    """
    Test that a TypedDict with total=False but an incorrect type fails type checking.
    """
    bad = {"id": "wrong-type"}
    result = check_type(bad, PartialUser)
    assert not result.success
    assert "expected int but got str" in "".join(result.errors)


def test_literal_valid() -> None:
    """
    Test that Literal values are correctly validated.
    """
    result = check_type(2, Literal[2, 3])
    assert result.success


def test_literal_invalid() -> None:
    """
    Test that values not in a Literal fail type checking.
    """
    result = check_type(1, Literal[2, 3])
    assert not result.success
    joined_errors = " ".join(result.errors)
    assert "expected literal" in joined_errors
    assert "but got 1" in joined_errors


@pytest.mark.parametrize("data", (10, {"nam": "foo", "configuration": {"foo": "bar"}}))
def test_typeddict_dtype_spec_invalid(data: DTypeSpec_V3) -> None:
    """
    Test that a TypedDict with dtype_spec fails type checking.
    """
    result = check_type(data, DTypeSpec_V3)
    assert not result.success


@pytest.mark.parametrize("data", ("foo", {"name": "foo", "configuration": {"foo": "bar"}}))
def test_typeddict_dtype_spec_valid(data: DTypeSpec_V3) -> None:
    """
    Test that a TypedDict with dtype_spec passes type checking.
    """
    x: DTypeSpec_V3 = "foo"
    result = check_type(x, DTypeSpec_V3)
    assert result.success


class InheritedTD(DTypeConfig_V2[str, None]): ...


@pytest.mark.parametrize("typ", [DTypeSpec_V2, DTypeConfig_V2[str, None], InheritedTD])
def test_typeddict_dtype_spec_v2_valid(typ: type) -> None:
    """
    Test that a TypedDict with dtype_spec passes type checking.
    """
    result = check_type({"name": "gzip", "object_codec_id": None}, typ)
    assert result.success


@pytest.mark.parametrize("typ", [DTypeConfig_V2[StructuredName_V2, None], StructuredJSON_V2])
def test_typeddict_recursive(typ: type) -> None:
    result = check_type(
        {"name": [["field1", ">i4"], ["field2", ">f8"]], "object_codec_id": None}, typ
    )
    assert result.success


def test_datetime_valid():
    class TimeConfig(TypedDict):
        unit: ReadOnly[DateTimeUnit]
        scale_factor: ReadOnly[int]

    DateTime64JSON_V3 = NamedConfig[Literal["numpy.datetime64"], TimeConfig]
    data = {"name": "numpy.datetime64", "configuration": {"unit": "ns", "scale_factor": 10}}
    result = check_type(data, DateTime64JSON_V3)
    assert result.success
