"""
Zarr v3 `int8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    single_byte,
)
from zarr_metadata.v3._types.data_type import INT8_DATA_TYPE_NAME, Int8DataTypeName, Int8FillValue

INT8_DATA_TYPE: Final = DataTypeDefinition(
    name=INT8_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int8FillValue,
    fill_value_default=ConstantFillValue(0),
    storage=single_byte,
)
"""The `int8` data type. It is written as a bare name and has no configuration. Its fill value is an integer in its range."""


__all__ = [
    "INT8_DATA_TYPE",
    "INT8_DATA_TYPE_NAME",
    "Int8DataTypeName",
    "Int8FillValue",
]
