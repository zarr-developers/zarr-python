"""
Zarr `struct` data type (heterogeneous record, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md
"""

from collections.abc import Iterator, Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import DataTypeDefinition, DataTypeField

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


def _rules(configuration: StructConfiguration) -> Iterator[ValidationProblem]:
    """Fields exist, and their names are non-empty and distinct.

    A fill value addresses fields by name. Whether each field's type is
    fixed-size is a question about that type, asked where types are read
    together.
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


STRUCT_DATA_TYPE: Final = DataTypeDefinition(
    name=STRUCT_DATA_TYPE_NAME, configuration=StructConfiguration, rules=_rules
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
