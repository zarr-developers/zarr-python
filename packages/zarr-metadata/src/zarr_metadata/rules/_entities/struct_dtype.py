"""Composition rules for the `struct` data type.

`StructField`'s own docstring promises field names are unique within a
struct and non-empty. Neither is expressible in a TypedDict, so both are
composition judgments and live here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._engine import as_string_mapping
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_NAME_PATTERN
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE_NAME

if TYPE_CHECKING:
    from zarr_metadata.rules._spec import ArraySpec

_ARRAY_V3 = "zarr_v3_array"

_FIXED_SIZE_NAMES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "numpy.datetime64",
        "numpy.timedelta64",
    }
)
_VARIABLE_SIZE_NAMES = frozenset({"bytes", "string"})


def _field_names(configuration: Mapping[str, object]) -> tuple[tuple[int, str], ...]:
    """`(index, name)` for each field with a string name, else nothing.

    Anything the shape validator would reject is skipped: it owns that
    complaint, and judging names inside a malformed field list is noise.
    """
    fields = configuration.get("fields")
    if not isinstance(fields, tuple):
        return ()
    named: list[tuple[int, str]] = []
    for index, field in enumerate(cast("tuple[object, ...]", fields)):
        if not isinstance(field, Mapping):
            continue
        name = cast("Mapping[object, object]", field).get("name")
        if isinstance(name, str):
            named.append((index, name))
    return tuple(named)


def _known_fixed_size(data_type: object) -> bool | None:
    """Whether a known data type is fixed-size; None means unknown."""
    if isinstance(data_type, str):
        name = data_type
        envelope = None
    else:
        envelope = as_string_mapping(data_type)
        raw_name = envelope.get("name") if envelope is not None else None
        name = raw_name if isinstance(raw_name, str) else None
    if name in _FIXED_SIZE_NAMES or (
        isinstance(name, str) and RAW_BYTES_NAME_PATTERN.fullmatch(name)
    ):
        return True
    if name in _VARIABLE_SIZE_NAMES:
        return False
    if name != STRUCT_DATA_TYPE_NAME or envelope is None:
        return None
    nested_configuration = as_string_mapping(envelope.get("configuration"))
    fields = nested_configuration.get("fields") if nested_configuration is not None else None
    if not isinstance(fields, tuple):
        return None
    results: list[bool] = []
    for field in cast("tuple[object, ...]", fields):
        field_mapping = as_string_mapping(field)
        if field_mapping is None or "data_type" not in field_mapping:
            return None
        result = _known_fixed_size(field_mapping["data_type"])
        if result is None:
            return None
        results.append(result)
    return all(results)


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME)
def fields_are_non_empty(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    fields = cast("tuple[object, ...]", configuration["fields"])
    if len(fields) != 0:
        return ()
    return (ValidationProblem(("fields",), "expected at least one struct field", "invalid_value"),)


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME)
def field_data_types_are_fixed_size(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    fields = cast("tuple[object, ...]", configuration["fields"])
    problems: list[ValidationProblem] = []
    for index, field in enumerate(fields):
        field_mapping = as_string_mapping(field)
        if field_mapping is None or "data_type" not in field_mapping:
            continue
        if _known_fixed_size(field_mapping["data_type"]) is False:
            problems.append(
                ValidationProblem(
                    ("fields", index, "data_type"),
                    "struct fields must use fixed-size data types",
                    "invalid_value",
                )
            )
    return tuple(problems)


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME)
def field_names_are_non_empty(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """A struct field must be addressable, so its name cannot be empty."""
    return tuple(
        ValidationProblem(
            ("fields", index, "name"), "expected a non-empty field name", "invalid_value"
        )
        for index, name in _field_names(configuration)
        if name == ""
    )


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME)
def field_names_are_unique(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArraySpec
) -> tuple[ValidationProblem, ...]:
    """Duplicate field names make a fill value's per-field mapping ambiguous."""
    seen: dict[str, int] = {}
    problems: list[ValidationProblem] = []
    for index, name in _field_names(configuration):
        first = seen.get(name)
        if first is None:
            seen[name] = index
            continue
        problems.append(
            ValidationProblem(
                ("fields", index, "name"),
                f"duplicate field name {name!r}, already used by field {first}",
                "invalid_value",
            )
        )
    return tuple(problems)
