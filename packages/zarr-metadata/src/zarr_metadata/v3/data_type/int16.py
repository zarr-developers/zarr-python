"""
Zarr v3 `int16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

INT16_DATA_TYPE_NAME: Final = "int16"
"""The `data_type` value for the `int16` type."""

Int16DataTypeName = Literal["int16"]
"""Literal type of the `data_type` field for `int16`."""

Int16FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int16`: a JSON integer in [-32768, 32767]."""


__all__ = [
    "INT16_DATA_TYPE_NAME",
    "Int16DataTypeName",
    "Int16FillValue",
]
