"""
Zarr v3 `int64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

INT64_DTYPE_NAME: Final = "int64"
"""The `data_type` value for the `int64` type."""

Int64DTypeName = Literal["int64"]
"""Literal type of the `data_type` field for `int64`."""

Int64FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int64`: a JSON integer in [-2**63, 2**63 - 1]."""


__all__ = [
    "INT64_DTYPE_NAME",
    "Int64DTypeName",
    "Int64FillValue",
]
