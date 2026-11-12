"""
Zarr v3 `uint8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    single_byte,
)
from zarr_metadata.v3._types.data_type import (
    UINT8_DATA_TYPE_NAME,
    Uint8DataTypeName,
    Uint8FillValue,
)

UINT8_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT8_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint8FillValue,
    fill_value_default=ConstantFillValue(0),
    storage=single_byte,
    number_class="integral",
)
"""The `uint8` data type. It is written as a bare name and has no configuration. Its fill value is an integer in its range."""


__all__ = [
    "UINT8_DATA_TYPE",
    "UINT8_DATA_TYPE_NAME",
    "Uint8DataTypeName",
    "Uint8FillValue",
]
