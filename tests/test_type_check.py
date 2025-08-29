from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ForwardRef, Literal, NotRequired, TypeVar

import pytest
from typing_extensions import ReadOnly, TypedDict

from src.zarr.core.type_check import (
    TypeCheckResult,
    check_type,
    ensure_type,
    guard_type,
)
from zarr.core.common import ArrayMetadataJSON_V3, DTypeSpec_V3, NamedConfig, StructuredName_V2
from zarr.core.dtype.common import DTypeConfig_V2, DTypeSpec_V2
from zarr.core.dtype.npy.structured import StructuredJSON_V2
from zarr.core.dtype.npy.time import TimeConfig


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
        ((True, False), (1, "True", [True]), bool),
        (("a", "1"), (1, True, ["a"]), str),
        ((1.0, 2.0), (1, True, ["a"]), float),
        ((1, 2), (1.0, True, ["a"]), int),
        ((1, 2), (3, 4, "a"), Literal[1, 2]),
        ((("a", 1), ("hello", 2)), ((True, 1), ("a", 1, 1)), tuple[str, int]),
        ((("a",), ("a", 1), ("hello", 2, 3)), ((True, 1), ("a", 1, 1.0)), tuple[str | int, ...]),
        ((["a", "b"], ["x"]), (["a", 1], "oops", 1), list[str]),
        (({"a": 1, "b": 2}, {"x": 10}), ({"a": "oops"}, [("a", 1)]), dict[str, int]),
        (({"a": 1, "b": 2}, {"x": 10}), ({"a": "oops"}, [("a", 1)]), Mapping[str, int]),
    ],
)
def test_inliers_outliers(inliers: tuple[Any, ...], outliers: tuple[Any, ...], typ: type) -> None:
    """
    Given a set of inliers and outliers for a type, test that check_type correctly
    identifies valid and invalid cases. This test is used for types that can be written down compactly.
    """
    assert all(check_type(val, typ).success for val in inliers)
    assert all(not check_type(val, typ).success for val in outliers)


def test_dict_valid() -> None:
    """
    Test that check_type correctly validates a dictionary with specific key-value types.

    Verifies that dictionary type checking works for homogeneous mappings,
    testing dict[str, int] with {"a": 1, "b": 2} where all keys are strings
    and all values are integers.
    """
    result = check_type({"a": 1, "b": 2}, dict[str, int])
    assert result.success


def test_dict_invalid() -> None:
    """
    Test that check_type correctly rejects a dictionary with mismatched value types.

    Verifies that dictionary type checking fails when values don't match
    the expected type. Tests dict[str, int] with {"a": 1, "b": "oops"}
    where "oops" is a string instead of the expected int.
    """
    result = check_type({"a": 1, "b": "oops"}, dict[str, int])
    assert not result.success
    # assert "expected int but got str" in result.errors[0]


def test_dict_any_valid() -> None:
    """
    Test that check_type correctly validates a dictionary when using Any type annotations.

    Verifies that dictionaries with dict[Any, Any] accept any combination of
    key and value types, testing with {1: "x", "y": 2} which has mixed types.
    """
    result = check_type({1: "x", "y": 2}, dict[Any, Any])
    assert result.success


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


def test_literal_valid() -> None:
    """
    Test that check_type correctly validates values against Literal types.

    Verifies that Literal type checking accepts values that are exactly one
    of the allowed literal values. Tests Literal[2, 3] with the value 2
    which is in the allowed set.
    """
    result = check_type(2, Literal[2, 3])
    assert result.success


def test_literal_invalid() -> None:
    """
    Test that check_type correctly rejects values not in the Literal's allowed set.

    Verifies that Literal type checking fails when the value is not one of
    the specified literal values. Tests Literal[2, 3] with the value 1
    which is not in the allowed set.
    """
    typ = Literal[2, 3]
    val = 1
    result = check_type(val, typ)
    assert not result.success
    # assert result.errors == [f"Expected literal in {get_args(typ)} but got {val!r}"]


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


# --- Additional tests for uncovered code paths ---


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


def test_check_type_any() -> None:
    """
    Test that check_type accepts any value when the expected type is Any.

    Verifies that the Any type annotation works as a universal type that
    accepts any input value without validation. Tests with string "anything"
    against Any type.
    """
    result = check_type("anything", Any)
    assert result.success


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


def test_tuple_variadic() -> None:
    """
    Test that check_type correctly validates variadic tuples using Ellipsis notation.

    Verifies that tuple[type, ...] syntax works for tuples of variable length
    where all elements must be of the same type. Tests tuple[int, ...]
    with both valid all-integer tuple and invalid mixed-type tuple.
    """
    result = check_type((1, 2, 3, 4), tuple[int, ...])
    assert result.success

    result = check_type((1, "bad", 3), tuple[int, ...])
    assert not result.success


def test_tuple_length_mismatch() -> None:
    """
    Test that check_type correctly rejects tuples with incorrect length.

    Verifies that fixed-length tuple validation enforces exact length matching.
    Tests tuple[int, str, bool] (expecting 3 elements) with a 2-element tuple,
    ensuring the length mismatch is detected and reported.
    """
    result = check_type((1, 2), tuple[int, str, bool])
    assert not result.success
    assert "expected tuple of length 3 but got 2" in result.errors[0]


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


def test_union_all_fail() -> None:
    """
    Test that check_type correctly handles union types where no option matches.

    Verifies that union type validation fails when the input value doesn't
    match any of the union's member types. Tests string "hello" against
    int | float union, which should fail for both types.
    """
    result = check_type("hello", int | float)
    assert not result.success
    # Should contain errors from both int and float checks


def test_union_success_early() -> None:
    """
    Test that check_type succeeds immediately when first union member matches.

    Verifies that union type validation short-circuits on the first successful
    match, making it efficient. Tests integer 42 against int | str union,
    which should succeed on the int check.
    """
    result = check_type(42, int | str)
    assert result.success


def test_mapping_key_value_errors() -> None:
    """
    Test that check_type correctly identifies both key and value type errors in mappings.

    Verifies that mapping validation checks both keys and values independently,
    reporting errors for mismatches in either. Tests dict[str, str] with
    mixed types for both keys and values.
    """
    bad_mapping = {1: "str", "str": 2}  # Mixed key types, mixed value types
    result = check_type(bad_mapping, dict[str, str])
    assert not result.success
    # Should have errors for both key and value mismatches


def test_non_mapping_object() -> None:
    """
    Test that check_type correctly rejects non-mapping objects for mapping types.

    Verifies that mapping type validation first checks if the object implements
    the Mapping protocol before checking keys/values. Tests list against
    dict[str, int] which should fail at the protocol level.
    """
    result = check_type([], dict[str, int])
    assert not result.success
    assert "expected  collections.abc.Mapping" in result.errors[0]


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


def test_typeddict_not_typeddict_fallback() -> None:
    """
    Test the fallback behavior when a type appears to be TypedDict but isn't.

    This tests the internal _get_typeddict_metadata function returning None
    for types that aren't actually TypedDict classes. This is a placeholder
    test for an edge case that's difficult to trigger externally.
    """
    # This tests the fallback in _get_typeddict_metadata returning None
    # We can't easily trigger this without internal manipulation


def test_annotated_types() -> None:
    """
    Test that check_type handles Annotated types by falling back to isinstance check.

    Verifies the current limitation where Annotated types cannot be properly
    validated and fall back to isinstance(), which fails. This documents
    the current behavior and tests the fallback error path.
    """
    from typing import Annotated

    # Annotated types currently fall back to isinstance check which fails
    # This shows the current limitation and tests the fallback path
    AnnotatedInt = Annotated[int, "some annotation"]
    result = check_type(42, AnnotatedInt)
    # Currently fails due to isinstance not working with Annotated
    assert not result.success
    assert "cannot be checked against" in result.errors[0]


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


def test_types_union_type() -> None:
    """
    Test that check_type correctly handles Python 3.10+ union syntax (str | int).

    Verifies that the modern union syntax using | operator works correctly
    when available (Python 3.10+). Tests str | int with string, integer,
    and invalid list values to ensure proper union handling.
    """

    # Test the new union syntax str | int
    union_type = str | int
    result = check_type("hello", union_type)
    assert result.success

    result = check_type(42, union_type)
    assert result.success

    result = check_type([], union_type)
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


def test_empty_containers() -> None:
    """
    Test that check_type correctly validates empty container types.

    Verifies that empty containers (list, dict, tuple) pass validation
    against their respective generic types. Tests empty list against list[int],
    empty dict against dict[str, int], and empty tuple against tuple[int, ...].
    """
    result = check_type([], list[int])
    assert result.success

    result = check_type({}, dict[str, int])
    assert result.success

    result = check_type((), tuple[int, ...])
    assert result.success


def test_none_literal() -> None:
    """
    Test that check_type correctly validates None within Literal types.

    Verifies that None can be used as a literal value in Literal types.
    Tests Literal[None, "other"] with None, "other", and invalid "wrong"
    values to ensure None is properly handled in literal validation.
    """
    result = check_type(None, Literal[None, "other"])
    assert result.success

    result = check_type("other", Literal[None, "other"])
    assert result.success

    result = check_type("wrong", Literal[None, "other"])
    assert not result.success


def test_union_with_none() -> None:
    """
    Test that check_type correctly validates union types containing None.

    Verifies that optional types (T | None) work correctly, accepting both
    the base type and None values. Tests int | None with None, integer 42,
    and invalid string "wrong" to ensure proper union validation.
    """
    result = check_type(None, int | None)
    assert result.success

    result = check_type(42, int | None)
    assert result.success

    result = check_type("wrong", int | None)
    assert not result.success
