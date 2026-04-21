"""
Zarr v3 `uint8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

UINT8_DTYPE_NAME: Final = "uint8"
"""The `data_type` value for the `uint8` type."""

Uint8DTypeName = Literal["uint8"]
"""Literal type of the `data_type` field for `uint8`."""

Uint8FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint8`: a JSON integer in [0, 255]."""


__all__ = [
    "UINT8_DTYPE_NAME",
    "Uint8DTypeName",
    "Uint8FillValue",
]
