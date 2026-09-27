"""
Zarr v3 `uint16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration
from zarr_metadata.v3.data_type._integer import integer_fill_value_rules

UINT16_DATA_TYPE_NAME: Final = "uint16"
"""The `data_type` value for the `uint16` type."""

Uint16DataTypeName = Literal["uint16"]
"""Literal type of the `data_type` field for `uint16`."""

Uint16FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint16`: a JSON integer in [0, 65535]."""


UINT16_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT16_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint16FillValue,
    fill_value_rules=integer_fill_value_rules(0, 2**16 - 1),
)
"""The `uint16` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "UINT16_DATA_TYPE",
    "UINT16_DATA_TYPE_NAME",
    "Uint16DataTypeName",
    "Uint16FillValue",
]
