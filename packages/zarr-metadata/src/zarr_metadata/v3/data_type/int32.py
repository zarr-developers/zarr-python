"""
Zarr v3 `int32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, multi_byte
from zarr_metadata.v3.data_type._integer import integer_fill_value_rules

INT32_DATA_TYPE_NAME: Final = "int32"
"""The `data_type` value for the `int32` type."""

Int32DataTypeName = Literal["int32"]
"""Literal type of the `data_type` field for `int32`."""

Int32FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int32`: a JSON integer in [-2**31, 2**31 - 1]."""


INT32_DATA_TYPE: Final = DataTypeDefinition(
    name=INT32_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int32FillValue,
    fill_value_rules=integer_fill_value_rules(-(2**31), 2**31 - 1),
    storage=multi_byte,
)
"""The `int32` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "INT32_DATA_TYPE",
    "INT32_DATA_TYPE_NAME",
    "Int32DataTypeName",
    "Int32FillValue",
]
