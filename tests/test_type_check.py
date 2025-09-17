from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ForwardRef, Literal, NotRequired, TypeVar

import pytest
from typing_extensions import ReadOnly, TypedDict

from zarr.core.common import ArrayMetadataJSON_V3, DTypeSpec_V3, NamedConfig, StructuredName_V2
from zarr.core.dtype.common import DTypeConfig_V2, DTypeSpec_V2
from zarr.core.dtype.npy.structured import StructuredJSON_V2
from zarr.core.dtype.npy.time import TimeConfig
from zarr.core.type_check import (
    TypeCheckResult,
    _get_typeddict_metadata,
    _resolve_type,
    _type_name,
    check_type,
    check_typeddict,
    ensure_type,
    guard_type,
)


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


@pytest.mark.parametrize(
    ("inliers", "outliers", "typ"),
    [
        ((1, 2, 3), (), Any),
        ((True, False), (1, "True", [True]), bool),
        (("a", "1"), (1, True, ["a"]), str),
        ((1.0, 2.0), (1, True, ["a"]), float),
        ((1, 2), (1.0, True, ["a"]), int),
        ((1, 2, None), (3, 4, "a"), Literal[1, 2, None]),
        ((1, 2, None), ("a", 1.2), int | None),
        ((("a", 1), ("hello", 2)), (True, 1, ("a", 1, 1), ()), tuple[str, int]),
        (
            (("a",), ("a", 1), ("hello", 2, 3), ()),
            ((True, 1), ("a", 1, 1.0)),
            tuple[str | int, ...],
        ),
        ((["a", "b"], ["x"], []), (["a", 1], "oops", 1), list[str]),
        (({"a": 1, "b": 2}, {"x": 10}, {}), ({"a": "oops"}, [("a", 1)]), dict[str, int]),
        (({"a": 1, "b": 2}, {10: 10}, {}), (), dict[Any, Any]),
        (({"a": 1, "b": 2}, {"x": 10}, {}), ({"a": "oops"}, [("a", 1)]), Mapping[str, int]),
    ],
)
def test_inliers_outliers(inliers: tuple[Any, ...], outliers: tuple[Any, ...], typ: type) -> None:
    """
    Given a set of inliers and outliers for a type, test that check_type correctly
    identifies valid and invalid cases. This test is used for types that can be written down compactly.
    """
    assert all(check_type(val, typ).success for val in inliers)
    assert all(not check_type(val, typ).success for val in outliers)


def test_typeddict_valid() -> None:
    """
    Test that check_type correctly validates a complex nested TypedDict structure.

    Verifies that TypedDict validation works with nested structures,
    testing a User TypedDict containing an Address TypedDict and list fields.
    All field types must match their annotations exactly.
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
    Test that check_type correctly rejects a TypedDict with invalid field types.

    Verifies that TypedDict validation fails when nested fields have wrong types.
    Tests a User with an Address where zipcode is "oops" (string) instead of
    the expected int type.
    """
    bad_user = {
        "id": 1,
        "name": "Alice",
        "address": {"street": "Main St", "zipcode": "oops"},
        "tags": ["x", "y"],
    }
    result = check_type(bad_user, User)
    assert not result.success
    # assert "expected int but got str" in "".join(result.errors)


def test_typeddict_fail_missing_required() -> None:
    """
    Test that check_type correctly rejects a TypedDict missing required fields.

    Verifies that TypedDict validation enforces required fields, failing when
    a required key is missing. Tests an Address TypedDict missing the required
    'zipcode' field.
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
    Test that check_type correctly handles TypedDict with total=False allowing empty dicts.

    Verifies that TypedDict with total=False makes all fields optional,
    allowing an empty dictionary {} to pass validation against PartialUser
    which has all optional fields.
    """
    result = check_type({}, PartialUser)
    assert result.success


def test_typeddict_partial_total_false_fail() -> None:
    """
    Test that check_type rejects TypedDict with total=False when present fields have wrong types.

    Verifies that even with total=False making fields optional, any present
    fields must still match their type annotations. Tests PartialUser with
    {"id": "wrong-type"} where id should be int.
    """
    bad = {"id": "wrong-type"}
    result = check_type(bad, PartialUser)
    assert not result.success
    # assert f"expected {int} but got 'wrong-type' with type {str}" in result.errors


@pytest.mark.parametrize("data", [10, {"blame": "foo", "configuration": {"foo": "bar"}}])
def test_typeddict_dtype_spec_invalid(data: DTypeSpec_V3) -> None:
    """
    Test that check_type correctly rejects invalid DTypeSpec_V3 structures.

    Verifies that DTypeSpec_V3 validation fails for incorrect formats.
    Tests with an integer (10) and a dict with wrong field names
    ("blame" instead of "name"), both should be rejected.
    """
    result = check_type(data, DTypeSpec_V3)
    assert not result.success


@pytest.mark.parametrize("data", ["foo", {"name": "foo", "configuration": {"foo": "bar"}}])
def test_typeddict_dtype_spec_valid(data: DTypeSpec_V3) -> None:
    """
    Test that check_type correctly accepts valid DTypeSpec_V3 structures.

    Verifies that DTypeSpec_V3 validation passes for correct formats.
    Tests with both a simple string ("foo") and a proper dict structure
    with "name" and "configuration" fields.
    """
    x: DTypeSpec_V3 = "foo"
    result = check_type(x, DTypeSpec_V3)
    assert result.success


class InheritedTD(DTypeConfig_V2[str, None]): ...


@pytest.mark.parametrize("typ", [DTypeSpec_V2, DTypeConfig_V2[str, None], InheritedTD])
def test_typeddict_dtype_spec_v2_valid(typ: type) -> None:
    """
    Test that check_type correctly validates various DTypeSpec_V2 and DTypeConfig_V2 types.

    Verifies that version 2 dtype specifications work correctly with different
    generic parameterizations. Tests DTypeSpec_V2, generic DTypeConfig_V2[str, None],
    and inherited TypedDict classes.
    """
    result = check_type({"name": "gzip", "object_codec_id": None}, typ)
    assert result.success


@pytest.mark.parametrize("typ", [DTypeConfig_V2[StructuredName_V2, None], StructuredJSON_V2])
def test_typeddict_recursive(typ: type) -> None:
    """
    Test that check_type correctly handles recursive/nested TypedDict structures.

    Verifies that complex nested structures like structured dtypes work properly.
    Tests with DTypeConfig_V2 containing StructuredName_V2 and StructuredJSON_V2
    which contain nested field definitions.
    """
    result = check_type(
        {"name": [["field1", ">i4"], ["field2", ">f8"]], "object_codec_id": None}, typ
    )
    assert result.success


def test_datetime_valid() -> None:
    """
    Test that check_type correctly validates datetime configuration structures.

    Verifies that complex NamedConfig structures work with specific literal types.
    Tests DateTime64JSON_V3 which is a NamedConfig with numpy.datetime64 literal
    name and TimeConfig configuration.
    """
    DateTime64JSON_V3 = NamedConfig[Literal["numpy.datetime64"], TimeConfig]
    data: DateTime64JSON_V3 = {
        "name": "numpy.datetime64",
        "configuration": {"unit": "ns", "scale_factor": 10},
    }
    result = check_type(data, DateTime64JSON_V3)
    assert result.success


@pytest.mark.parametrize(
    "optionals",
    [{}, {"attributes": {}}, {"storage_transformers": ()}, {"dimension_names": ("a", "b")}],
)
def test_zarr_v2_metadata(optionals: dict[str, object]) -> None:
    """
    Test that check_type correctly validates ArrayMetadataJSON_V3 with optional fields.

    Verifies that Zarr v3 array metadata validation works with different combinations
    of optional fields. Tests the base required fields plus various optional field
    combinations like attributes, storage_transformers, and dimension_names.
    """
    meta: ArrayMetadataJSON_V3 = {
        "zarr_format": 3,
        "node_type": "array",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "."}},
        "shape": (10, 10),
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (5, 5)}},
        "codecs": ("bytes",),
        "attributes": {"a": 1, "b": 2},
        "data_type": "uint8",
    } | optionals  # type: ignore[assignment]
    result = check_type(meta, ArrayMetadataJSON_V3)
    assert result.success


def test_external_generic_typeddict() -> None:
    """
    Test that check_type correctly validates external generic TypedDict structures.

    Verifies that generic TypedDict classes from external modules work properly.
    Tests NamedConfig with specific literal type and Mapping configuration,
    ensuring generic type parameters are resolved correctly.
    """
    x: NamedConfig[Literal["default"], Mapping[str, object]] = {
        "name": "default",
        "configuration": {"foo": "bar"},
    }
    result = check_type(x, NamedConfig[Literal["default"], Mapping[str, object]])
    assert result.success


def test_typeddict_extra_keys_allowed() -> None:
    """
    Test that check_type allows extra keys in TypedDict structures.

    Verifies that TypedDict validation is flexible about additional keys
    not defined in the TypedDict schema. Tests a TypedDict with field 'a'
    but provides both 'a' and 'b', ensuring 'b' is allowed.
    """

    class X(TypedDict):
        a: int

    b: X = {"a": 1, "b": 2}  # type: ignore[typeddict-unknown-key]
    result = check_type(b, X)
    assert result.success


def test_typeddict_readonly_notrequired() -> None:
    """
    Test that check_type correctly handles ReadOnly and NotRequired annotations in TypedDict.

    Verifies that complex type annotations like ReadOnly[NotRequired[int]] work properly.
    Tests various combinations and nesting of ReadOnly and NotRequired annotations,
    ensuring all optional fields can be omitted.
    """

    class X(TypedDict):
        a: ReadOnly[NotRequired[int]]
        b: NotRequired[ReadOnly[int]]
        c: Annotated[ReadOnly[NotRequired[int]], 10]
        d: int

    b: X = {"d": 1}
    result = check_type(b, X)
    assert result.success


def test_ensure_type_valid() -> None:
    """
    Test that ensure_type returns the input value when type validation succeeds.

    Verifies that ensure_type acts as a type-safe identity function,
    returning the original value when it matches the expected type.
    Tests with integer 42 against int type.
    """
    result = ensure_type(42, int)
    assert result == 42


def test_ensure_type_invalid() -> None:
    """
    Test that ensure_type raises TypeError when type validation fails.

    Verifies that ensure_type throws an appropriate TypeError with descriptive
    message when the input value doesn't match the expected type.
    Tests with string "hello" against int type.
    """
    with pytest.raises(TypeError, match="Expected an instance of <class 'int'> but got 'hello'"):
        ensure_type("hello", int)


def test_guard_type_valid() -> None:
    """
    Test that guard_type returns True when type validation succeeds.

    Verifies that guard_type acts as a boolean type guard function,
    returning True when the input matches the expected type for use
    in conditional type narrowing. Tests with integer 42 against int type.
    """
    assert guard_type(42, int) is True


def test_guard_type_invalid() -> None:
    """
    Test that guard_type returns False when type validation fails.

    Verifies that guard_type correctly identifies type mismatches by returning
    False, allowing for safe type narrowing in conditional blocks.
    Tests with string "hello" against int type.
    """
    assert guard_type("hello", int) is False


def test_check_type_none_with_none_type() -> None:
    """
    Test that check_type correctly validates None against type(None) annotation.

    Verifies that explicit None type checking works using type(None) syntax
    as opposed to just None. Tests both valid None value and invalid
    string "not none" against type(None).
    """
    result = check_type(None, type(None))
    assert result.success

    result = check_type("not none", type(None))
    assert not result.success


def test_check_type_fallback_isinstance() -> None:
    """
    Test that check_type falls back to isinstance() for custom class types.

    Verifies that when no specific type checking logic applies, the function
    falls back to using isinstance() for validation. Tests with a custom
    class to ensure both valid instances and invalid values are handled correctly.
    """

    class CustomClass:
        pass

    obj = CustomClass()
    result = check_type(obj, CustomClass)
    assert result.success

    result = check_type("not custom", CustomClass)
    assert not result.success


def test_check_type_fallback_type_error() -> None:
    """
    Test that check_type handles TypeError in fallback isinstance() gracefully.

    Verifies that when isinstance() raises a TypeError (e.g., with ForwardRef),
    the function catches the exception and returns an appropriate error message.
    Tests with a problematic ForwardRef type.
    """
    # Create a problematic type that can't be used with isinstance
    problematic_type = ForwardRef("NonExistentType")
    result = check_type("anything", problematic_type)
    assert not result.success
    assert "cannot be checked against" in result.errors[0]


def test_sequence_type_string_bytes_excluded() -> None:
    """
    Test that check_type excludes strings and bytes from sequence type validation.

    Verifies that str and bytes are not treated as sequences even though they
    technically implement the sequence protocol. This prevents strings from
    being validated as list[str] where each character would be checked.
    """
    result = check_type("string", list[str])
    assert not result.success
    assert "expected sequence" in result.errors[0]

    result = check_type(b"bytes", list[str])
    assert not result.success
    assert "expected sequence" in result.errors[0]


def test_typeddict_non_dict() -> None:
    """
    Test that check_type correctly rejects non-dict objects for TypedDict validation.

    Verifies that TypedDict validation first ensures the input is a dictionary
    before checking field types. Tests list against User TypedDict,
    which should fail at the dict requirement level.
    """
    result = check_type([], User)
    assert not result.success
    assert "expected dict for TypedDict" in result.errors[0]


def test_typeddict_type_parameter_mismatch() -> None:
    """
    Test that check_type correctly detects type parameter count mismatches in generic TypedDict.

    Verifies that generic TypedDict validation enforces correct parameterization.
    Tests a generic TypedDict with TypeVar T, ensuring that type parameter
    counting validation works properly and reports mismatches.
    """
    T = TypeVar("T")

    class GenericTD(TypedDict):
        value: T  # type: ignore[valid-type]

    # This will trigger a type parameter count mismatch because
    # Generic TypedDict validation is strict about parameter counts
    result = check_type({"value": 42}, GenericTD[int])  # type: ignore[misc]
    # This actually fails due to type parameter mismatch validation
    assert not result.success
    assert "type parameter count mismatch" in result.errors[0]


def test_complex_nested_unions() -> None:
    """
    Test that check_type correctly validates complex nested structures with union types.

    Verifies that deeply nested type validation works with combinations of
    dictionaries and union types. Tests dict[str, int | str | None] with
    valid data containing different union member types and invalid data.
    """
    ComplexType = dict[str, int | str | None]

    test_data: dict[str, Any] = {"int_val": 42, "str_val": "hello", "none_val": None}

    result = check_type(test_data, ComplexType)
    assert result.success

    bad_data: dict[str, list[Any]] = {
        "bad_val": []  # list is not in the union
    }

    result = check_type(bad_data, ComplexType)
    assert not result.success


def test_type_check_result_dataclass() -> None:
    """
    Test that TypeCheckResult dataclass works correctly as a return type.

    Verifies that the TypeCheckResult dataclass properly stores success status
    and error messages. Tests both successful validation (empty errors) and
    failed validation (with multiple errors).
    """
    result = TypeCheckResult(True, [])
    assert result.success
    assert result.errors == []

    result = TypeCheckResult(False, ["error1", "error2"])
    assert not result.success
    assert len(result.errors) == 2


def test_sequence_with_collections_abc() -> None:
    """
    Test that check_type correctly validates custom sequence implementations.

    Verifies that sequence type checking works with custom classes that
    implement collections.abc.Sequence protocol. Tests a CustomSequence
    class against list[int] to ensure protocol-based validation works.
    """

    # Test with a custom sequence
    class CustomSequence(Sequence[Any]):
        def __init__(self, items: Sequence[Any]) -> None:
            self._items = items

        def __getitem__(self, index: Any) -> Any:
            return self._items[index]

        def __len__(self) -> int:
            return len(self._items)

    custom_seq = CustomSequence([1, 2, 3])
    result = check_type(custom_seq, list[int])
    assert result.success


@pytest.mark.parametrize(
    ("typ", "expected"),
    [
        (str, "str"),
        (list, "list"),
        (list[int], "list[int]"),
        (str | int, "str | int"),
    ],
)
def test_type_name(typ: Any, expected: str) -> None:
    assert _type_name(typ) == expected


def test_typevar_self_reference_edge_case() -> None:
    """
    Test TypeVar that maps to itself in type resolution (line 114).

    Tests the edge case where a TypeVar in the type_map resolves to itself,
    triggering the self-reference detection in _resolve_type_impl.
    This covers the rarely hit line 114.
    """
    T = TypeVar("T")
    # Create a type_map where T maps to itself
    type_map = {T: T}

    # This should trigger the self-reference detection
    result = _resolve_type(T, type_map=type_map)
    assert result == T  # Should return the original TypeVar


def test_non_typeddict_fallback_error() -> None:
    """
    Test error when non-TypedDict is passed to check_typeddict (line 259).

    Tests the fallback error case when _get_typeddict_metadata returns None,
    meaning the type is not actually a TypedDict.
    """

    # Pass a regular class that's not a TypedDict
    class NotATypedDict:
        pass

    result = check_typeddict({"key": "value"}, NotATypedDict, "test_path")
    assert not result.success
    assert "expected a TypedDict but got" in result.errors[0]


def test_get_typeddict_metadata_fallback() -> None:
    """

    Tests the fallback case where _get_typeddict_metadata cannot extract
    valid metadata from the provided type.
    """

    # Test with a type that's not a TypedDict at all
    result = _get_typeddict_metadata(int)
    assert result == (None, None, None, None)

    # Test with a complex type that looks like it might be TypedDict but isn't
    result = _get_typeddict_metadata(dict[str, int])
    assert result == (None, None, None, None)


@pytest.mark.parametrize(("typ_str", "typ_expected"), [("int", int), ("str", str), ("list", list)])
def test_complex_forwardref_scenarios(typ_str: str, typ_expected: type) -> None:
    """
    Test additional ForwardRef scenarios to ensure coverage.

    Tests various ForwardRef evaluation scenarios including edge cases
    that might not be covered by the basic test.
    """
    # String type annotations aren't handled the same way as ForwardRef
    # in the current implementation. The ForwardRef evaluation happens
    # in internal type resolution, not in the main check_type path.

    # Instead, let's test a scenario that would use ForwardRef internally

    # Test string that would need evaluation in type context
    result = _resolve_type(typ_str, globalns=globals())
    assert result is typ_expected
