"""
Zarr v3 `uint32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

UINT32_DTYPE_NAME: Final = "uint32"
"""The `data_type` value for the `uint32` type."""

Uint32DTypeName = Literal["uint32"]
"""Literal type of the `data_type` field for `uint32`."""

Uint32FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint32`: a JSON integer in [0, 2**32 - 1]."""


__all__ = [
    "UINT32_DTYPE_NAME",
    "Uint32DTypeName",
    "Uint32FillValue",
]
