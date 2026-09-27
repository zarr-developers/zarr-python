"""
Zarr v3 `uint8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration
from zarr_metadata.v3.data_type._integer import integer_fill_value_rules

UINT8_DATA_TYPE_NAME: Final = "uint8"
"""The `data_type` value for the `uint8` type."""

Uint8DataTypeName = Literal["uint8"]
"""Literal type of the `data_type` field for `uint8`."""

Uint8FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint8`: a JSON integer in [0, 255]."""


UINT8_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT8_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint8FillValue,
    fill_value_rules=integer_fill_value_rules(0, 2**8 - 1),
)
"""The `uint8` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "UINT8_DATA_TYPE",
    "UINT8_DATA_TYPE_NAME",
    "Uint8DataTypeName",
    "Uint8FillValue",
]
