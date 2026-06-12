"""
Zarr `struct` data type (heterogeneous record, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/main/data-types/struct/README.md
"""

from collections.abc import Mapping
from typing import Final, Literal

from typing_extensions import ReadOnly, TypedDict

STRUCT_DATA_TYPE_NAME: Final = "struct"
"""The `name` field value of the `struct` data type."""

StructDataTypeName = Literal["struct"]
"""Literal type of the `name` field of the `struct` data type."""


class StructField(TypedDict):
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
    data_type: ReadOnly[object]


class StructConfiguration(TypedDict):
    """Configuration for the `struct` data type."""

    fields: ReadOnly[tuple[StructField, ...]]


class Struct(TypedDict):
    """`struct` data type metadata."""

    name: StructDataTypeName
    configuration: StructConfiguration


StructFillValue = Mapping[str, object]
"""Permitted JSON shape of the `fill_value` field for `struct`.

A JSON object mapping each field name to that field's fill value. Field
fill values are themselves shaped per the field's `data_type`, recursively.
"""

__all__ = [
    "STRUCT_DATA_TYPE_NAME",
    "Struct",
    "StructConfiguration",
    "StructDataTypeName",
    "StructField",
    "StructFillValue",
]
