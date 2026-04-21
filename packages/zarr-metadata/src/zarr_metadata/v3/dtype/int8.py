"""
Zarr v3 `int8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

INT8_DTYPE_NAME: Final = "int8"
"""The `data_type` value for the `int8` type."""

Int8DTypeName = Literal["int8"]
"""Literal type of the `data_type` field for `int8`."""

Int8FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int8`: a JSON integer in [-128, 127]."""


__all__ = [
    "INT8_DTYPE_NAME",
    "Int8DTypeName",
    "Int8FillValue",
]
