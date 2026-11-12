"""
Zarr v3 `int16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    INT16_DATA_TYPE_NAME,
    Int16DataTypeName,
    Int16FillValue,
)

INT16_DATA_TYPE: Final = DataTypeDefinition(
    name=INT16_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int16FillValue,
    storage=multi_byte,
)
"""The `int16` data type. It is written as a bare name and has no configuration. Its fill value is an integer in its range."""


__all__ = [
    "INT16_DATA_TYPE",
    "INT16_DATA_TYPE_NAME",
    "Int16DataTypeName",
    "Int16FillValue",
]
