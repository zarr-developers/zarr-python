"""
String-shaped Zarr v3 data types.

This module covers:

- `string` (variable-length utf-8, bare string, zarr-extensions)
- `fixed_length_utf32` (fixed-length utf-32, envelope, zarr-extensions)

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
"""

from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

STRING_DTYPE_NAME: Final = "string"
"""The `data_type` value for the variable-length `string` type."""

StringDTypeName = Literal["string"]
"""Literal type of the `data_type` field for `string`."""

FIXED_LENGTH_UTF32_DTYPE_NAME: Final = "fixed_length_utf32"
"""The `name` field value of the `fixed_length_utf32` data type."""

FixedLengthUtf32DTypeName = Literal["fixed_length_utf32"]
"""Literal type of the `name` field of the `fixed_length_utf32` data type."""


class LengthBytesConfig(TypedDict):
    """
    Configuration for fixed-length string-shaped data types.

    Used by `fixed_length_utf32`. `length_bytes` is the allocated byte count
    per element; for utf-32 this must be a multiple of 4.
    """

    length_bytes: ReadOnly[int]


class FixedLengthUtf32(TypedDict):
    """`fixed_length_utf32` data type metadata."""

    name: FixedLengthUtf32DTypeName
    configuration: LengthBytesConfig


__all__ = [
    "FIXED_LENGTH_UTF32_DTYPE_NAME",
    "STRING_DTYPE_NAME",
    "FixedLengthUtf32",
    "FixedLengthUtf32DTypeName",
    "LengthBytesConfig",
    "StringDTypeName",
]
