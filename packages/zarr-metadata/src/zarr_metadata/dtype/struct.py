"""
Structured (heterogeneous) Zarr v3 data type.

See https://github.com/zarr-developers/zarr-extensions/blob/main/data-types/struct/README.md
"""

from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

from zarr_metadata.dtype import DType

STRUCT_DTYPE_NAME: Final = "struct"
"""The `name` field value of the `struct` data type."""

StructDTypeName = Literal["struct"]
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
    data_type: ReadOnly[DType]


class StructConfig(TypedDict):
    """Configuration for the `struct` data type."""

    fields: ReadOnly[tuple[StructField, ...]]


class Struct(TypedDict):
    """`struct` data type metadata."""

    name: StructDTypeName
    configuration: StructConfig


__all__ = [
    "STRUCT_DTYPE_NAME",
    "Struct",
    "StructConfig",
    "StructDTypeName",
    "StructField",
]
