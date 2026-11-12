"""
Zarr v3 `uint32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    UINT32_DATA_TYPE_NAME,
    Uint32DataTypeName,
    Uint32FillValue,
)

UINT32_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT32_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint32FillValue,
    storage=multi_byte,
)
"""The `uint32` data type. It is written as a bare name and has no configuration. Its fill value is an integer in its range."""


__all__ = [
    "UINT32_DATA_TYPE",
    "UINT32_DATA_TYPE_NAME",
    "Uint32DataTypeName",
    "Uint32FillValue",
]
