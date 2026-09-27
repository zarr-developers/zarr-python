"""
Zarr v3 `uint64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration
from zarr_metadata.v3.data_type._integer import integer_fill_value_rules

UINT64_DATA_TYPE_NAME: Final = "uint64"
"""The `data_type` value for the `uint64` type."""

Uint64DataTypeName = Literal["uint64"]
"""Literal type of the `data_type` field for `uint64`."""

Uint64FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint64`: a JSON integer in [0, 2**64 - 1]."""


UINT64_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint64FillValue,
    fill_value_rules=integer_fill_value_rules(0, 2**64 - 1),
)
"""The `uint64` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "UINT64_DATA_TYPE",
    "UINT64_DATA_TYPE_NAME",
    "Uint64DataTypeName",
    "Uint64FillValue",
]
