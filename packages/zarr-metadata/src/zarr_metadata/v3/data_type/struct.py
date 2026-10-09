"""
Zarr `struct` data type (heterogeneous record, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md
"""

from collections.abc import Iterator, Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    DataTypeField,
    Nested,
    StorageClass,
    fill_value_problems,
    spelled_canonically,
    storage_of,
)

STRUCT_DATA_TYPE_NAME: Final = "struct"
"""The `name` field value of the `struct` data type."""

StructDataTypeName = Literal["struct"]
"""Literal type of the `name` field of the `struct` data type."""


class StructField(TypedDict, closed=True):
    """
    A single field entry inside a structured dtype.

    Attributes
    ----------
    name
        The field name (must be unique within a struct and non-empty).
    data_type
        The field's data type. Recursive: may be a bare-string primitive
        or a named-config envelope including another `struct`.
    """

    name: ReadOnly[str]
    data_type: ReadOnly[DataTypeField]


class StructConfiguration(TypedDict, closed=True):
    """Configuration for the `struct` data type."""

    fields: ReadOnly[tuple[StructField, ...]]


class Struct(TypedDict, closed=True):
    """`struct` data type metadata."""

    name: StructDataTypeName
    configuration: StructConfiguration
    must_understand: NotRequired[bool]


StructFillValue = Mapping[str, JSONValue]
"""Permitted JSON shape of the `fill_value` field for `struct`.

A JSON object mapping each field name to that field's fill value. Field
fill values are themselves shaped per the field's `data_type`, recursively.
"""


def _rules(configuration: StructConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
    """Fields exist, their names are non-empty and distinct, and their types of fixed size.

    A fill value addresses fields by name. "Variable-length data types
    (e.g. "string") MUST NOT be used as field types, as they do not have a
    fixed encoded size"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md?plain=1#L42-L49):
    judged of each field type the scope read, and one whose storage is
    unknown is left be.
    """
    fields = configuration["fields"]
    if len(fields) == 0:
        yield ValidationProblem(("fields",), "expected at least one struct field", "invalid_value")
    seen: dict[str, int] = {}
    for index, member in enumerate(fields):
        name = member["name"]
        if name == "":
            yield ValidationProblem(
                ("fields", index, "name"), "expected a non-empty field name", "invalid_value"
            )
        first = seen.setdefault(name, index)
        if first != index:
            yield ValidationProblem(
                ("fields", index, "name"),
                f"duplicate field name {name!r}, already used by field {first}",
                "invalid_value",
            )
        field_type = nested.get(("fields", index, "data_type"))
        if field_type is not None and storage_of(field_type) == "variable_length":
            yield ValidationProblem(
                ("fields", index, "data_type"),
                f"expected a data type of fixed size, got {shown(field_type.name)}, whose values "
                "vary in size",
                "invalid_value",
            )


def _fill_value_rules(
    configuration: StructConfiguration, nested: Nested, value: StructFillValue
) -> Iterator[ValidationProblem]:
    """A fill value for every field, each one judged by that field's own type, and for nothing else.

    Every field needs one, valid for its type
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md?plain=1#L221-L224).
    A field type the scope did not read, or whose reading the struct's does
    not hold, leaves its fill value unjudged.
    """
    names = [member["name"] for member in configuration["fields"]]
    declared = set(names)
    for index, name in enumerate(names):
        if name not in value:
            yield ValidationProblem(
                (name,), f"expected a fill value for struct field {name!r}", "missing_key"
            )
            continue
        field_type = nested.get(("fields", index, "data_type"))
        if field_type is not None:
            yield from fill_value_problems(field_type, value[name], (name,))
    for key in value:
        if key not in declared:
            yield ValidationProblem((key,), f"no struct field is named {key!r}", "unknown_key")


def _fill_value_canonical(
    configuration: StructConfiguration, nested: Nested, value: StructFillValue
) -> dict[str, JSONValue]:
    """Each field's fill value in the canonical spelling of that field's own type, in the order the fields are declared.

    A field type the scope did not read, or whose reading the struct's does
    not hold, spells its fill value as written.
    """
    spelled: dict[str, JSONValue] = {}
    for index, member in enumerate(configuration["fields"]):
        name = member["name"]
        field_type = nested.get(("fields", index, "data_type"))
        held = value[name]
        spelled[name] = held if field_type is None else spelled_canonically(field_type, held)
    return spelled


def _storage(configuration: StructConfiguration, nested: Nested) -> StorageClass | None:
    """Its fields' values, packed together: numbers of several bytes if any field holds them, single bytes if every field is made of them.

    A nested struct "is encoded as the packed concatenation of its own
    sub-fields, recursively"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md?plain=1#L142-L144),
    and the `bytes` codec "MUST be configured with an explicit endian"
    for a struct that "contains multi-byte numeric fields", while one
    "composed entirely of single-byte fields" may go without
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md?plain=1#L211-L217).
    A field whose storage is unknown leaves the struct's unknown, unless
    another field's settles it. A struct with a field whose values vary in
    size is refused by its rules, so is never asked.
    """
    found = {
        None if field_type is None else storage_of(field_type)
        for field_type in (
            nested.get(("fields", index, "data_type"))
            for index in range(len(configuration["fields"]))
        )
    }
    if "multi_byte" in found:
        return "multi_byte"
    return "single_byte" if found == {"single_byte"} else None


STRUCT_DATA_TYPE: Final = DataTypeDefinition(
    name=STRUCT_DATA_TYPE_NAME,
    configuration=StructConfiguration,
    rules=_rules,
    fill_value=StructFillValue,
    fill_value_rules=_fill_value_rules,
    fill_value_canonical=_fill_value_canonical,
    storage=_storage,
)
"""The `struct` data type: a record of named fields, each field's type a nested field."""


__all__ = [
    "STRUCT_DATA_TYPE",
    "STRUCT_DATA_TYPE_NAME",
    "Struct",
    "StructConfiguration",
    "StructDataTypeName",
    "StructField",
    "StructFillValue",
]
