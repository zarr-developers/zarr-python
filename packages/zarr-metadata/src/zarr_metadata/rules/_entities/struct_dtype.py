"""Composition rules for the `struct` data type.

`StructField`'s own docstring promises field names are unique within a
struct and non-empty. Neither is expressible in a TypedDict, so both are
composition judgments and live here.

The fixed-size field rule is the `bytes` codec's question asked from the
other side — the spec writes it as "Variable-length data types (e.g.
`"string"`) MUST NOT be used as field types, as they do not have a fixed
encoded size" — so it defers to the shared classifier in
`zarr_metadata.rules._storage_class` rather than keeping a second table
of data-type sizes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._engine import as_string_mapping
from zarr_metadata.rules._registry import entity_rule, run_entity_rules
from zarr_metadata.rules._storage_class import storage_class
from zarr_metadata.v3._extension_points import DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE_NAME

if TYPE_CHECKING:
    from zarr_metadata.rules._spec import ArrayParts

_ARRAY_V3 = "zarr_v3_array"


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME, reads=frozenset({"fields"}))
def field_data_types_obey_their_rules(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArrayParts | None
) -> tuple[ValidationProblem, ...]:
    """Apply every known data type's rules inside struct fields, recursively."""
    problems: list[ValidationProblem] = []
    for index, field in enumerate(cast("tuple[object, ...]", configuration["fields"])):
        field_mapping = as_string_mapping(field)
        if field_mapping is not None:
            problems.extend(
                run_entity_rules(
                    DATA_TYPE,
                    field_mapping.get("data_type"),
                    document,
                    ("fields", index, "data_type"),
                )
            )
    return tuple(problems)


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


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME, reads=frozenset({"fields"}))
def fields_are_non_empty(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArrayParts | None
) -> tuple[ValidationProblem, ...]:
    fields = cast("tuple[object, ...]", configuration["fields"])
    if len(fields) != 0:
        return ()
    return (ValidationProblem(("fields",), "expected at least one struct field", "invalid_value"),)


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME, reads=frozenset({"fields"}))
def field_data_types_are_fixed_size(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArrayParts | None
) -> tuple[ValidationProblem, ...]:
    fields = cast("tuple[object, ...]", configuration["fields"])
    problems: list[ValidationProblem] = []
    for index, field in enumerate(fields):
        field_mapping = as_string_mapping(field)
        if field_mapping is None or "data_type" not in field_mapping:
            continue
        if storage_class(field_mapping["data_type"]) == "variable_length":
            problems.append(
                ValidationProblem(
                    ("fields", index, "data_type"),
                    "struct fields must use fixed-size data types",
                    "invalid_value",
                )
            )
    return tuple(problems)


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME, reads=frozenset({"fields"}))
def field_names_are_non_empty(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArrayParts | None
) -> tuple[ValidationProblem, ...]:
    """A struct field must be addressable, so its name cannot be empty."""
    return tuple(
        ValidationProblem(
            ("fields", index, "name"), "expected a non-empty field name", "invalid_value"
        )
        for index, name in _field_names(configuration)
        if name == ""
    )


@entity_rule(_ARRAY_V3, DATA_TYPE, STRUCT_DATA_TYPE_NAME, reads=frozenset({"fields"}))
def field_names_are_unique(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArrayParts | None
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
