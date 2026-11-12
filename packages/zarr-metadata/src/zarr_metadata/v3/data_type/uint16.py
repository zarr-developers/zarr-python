"""
Zarr v3 `uint16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    UINT16_DATA_TYPE_NAME,
    Uint16DataTypeName,
    Uint16FillValue,
)

UINT16_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT16_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint16FillValue,
    storage=multi_byte,
)
"""The `uint16` data type. It is written as a bare name and has no configuration. Its fill value is an integer in its range."""


__all__ = [
    "UINT16_DATA_TYPE",
    "UINT16_DATA_TYPE_NAME",
    "Uint16DataTypeName",
    "Uint16FillValue",
]
