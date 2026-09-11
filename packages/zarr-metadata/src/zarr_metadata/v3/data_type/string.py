"""
Zarr `string` data type (variable-length utf-8, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/string
"""

from typing import Final, Literal

STRING_DATA_TYPE_NAME: Final = "string"
"""The `data_type` value for the `string` type."""

StringDataTypeName = Literal["string"]
"""Literal type of the `data_type` field for `string`."""

StringFillValue = str
"""Permitted JSON shape of the `fill_value` field for `string`: a JSON unicode string."""


__all__ = [
    "STRING_DATA_TYPE_NAME",
    "StringDataTypeName",
    "StringFillValue",
]
