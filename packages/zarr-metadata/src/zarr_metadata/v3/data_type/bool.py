"""
Zarr v3 `bool` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    single_byte,
)
from zarr_metadata.v3._types.data_type import BOOL_DATA_TYPE_NAME, BoolDataTypeName, BoolFillValue

BOOL_DATA_TYPE: Final = DataTypeDefinition(
    name=BOOL_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=BoolFillValue,
    fill_value_default=ConstantFillValue(False),
    storage=single_byte,
    number_class="non_numeric",
)
"""The `bool` data type. It is written as a bare name and has no configuration."""


__all__ = [
    "BOOL_DATA_TYPE",
    "BOOL_DATA_TYPE_NAME",
    "BoolDataTypeName",
    "BoolFillValue",
]
