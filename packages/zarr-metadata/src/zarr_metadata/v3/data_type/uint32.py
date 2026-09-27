"""
Zarr v3 `uint32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, multi_byte
from zarr_metadata.v3.data_type._integer import integer_fill_value_rules

UINT32_DATA_TYPE_NAME: Final = "uint32"
"""The `data_type` value for the `uint32` type."""

Uint32DataTypeName = Literal["uint32"]
"""Literal type of the `data_type` field for `uint32`."""

Uint32FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint32`: a JSON integer in [0, 2**32 - 1]."""


UINT32_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT32_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint32FillValue,
    fill_value_rules=integer_fill_value_rules(0, 2**32 - 1),
    storage=multi_byte,
)
"""The `uint32` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "UINT32_DATA_TYPE",
    "UINT32_DATA_TYPE_NAME",
    "Uint32DataTypeName",
    "Uint32FillValue",
]
