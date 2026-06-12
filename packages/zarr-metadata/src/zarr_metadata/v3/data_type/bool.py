"""
Zarr v3 `bool` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

BOOL_DATA_TYPE_NAME: Final = "bool"
"""The `data_type` value for the `bool` type."""

BoolDataTypeName = Literal["bool"]
"""Literal type of the `data_type` field for `bool`."""

BoolFillValue = bool
"""Permitted JSON shape of the `fill_value` field for `bool`: a JSON boolean."""


__all__ = [
    "BOOL_DATA_TYPE_NAME",
    "BoolDataTypeName",
    "BoolFillValue",
]
