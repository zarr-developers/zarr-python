"""
Zarr v3 `uint64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

UINT64_DATA_TYPE_NAME: Final = "uint64"
"""The `data_type` value for the `uint64` type."""

Uint64DataTypeName = Literal["uint64"]
"""Literal type of the `data_type` field for `uint64`."""

Uint64FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint64`: a JSON integer in [0, 2**64 - 1]."""


__all__ = [
    "UINT64_DATA_TYPE_NAME",
    "Uint64DataTypeName",
    "Uint64FillValue",
]
