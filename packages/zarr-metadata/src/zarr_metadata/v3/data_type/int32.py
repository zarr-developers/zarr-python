"""
Zarr v3 `int32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

INT32_DTYPE_NAME: Final = "int32"
"""The `data_type` value for the `int32` type."""

Int32DTypeName = Literal["int32"]
"""Literal type of the `data_type` field for `int32`."""

Int32FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int32`: a JSON integer in [-2**31, 2**31 - 1]."""


__all__ = [
    "INT32_DTYPE_NAME",
    "Int32DTypeName",
    "Int32FillValue",
]
