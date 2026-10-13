"""
Zarr v3 `int16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, multi_byte
from zarr_metadata.v3.data_type._integer import integer_fill_value_rules

INT16_DATA_TYPE_NAME: Final = "int16"
"""The `data_type` value for the `int16` type."""

Int16DataTypeName = Literal["int16"]
"""Literal type of the `data_type` field for `int16`."""

Int16FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int16`: a JSON integer in [-32768, 32767]."""


INT16_DATA_TYPE: Final = DataTypeDefinition(
    name=INT16_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int16FillValue,
    fill_value_rules=integer_fill_value_rules(-(2**15), 2**15 - 1),
    storage=multi_byte,
)
"""The `int16` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "INT16_DATA_TYPE",
    "INT16_DATA_TYPE_NAME",
    "Int16DataTypeName",
    "Int16FillValue",
]
