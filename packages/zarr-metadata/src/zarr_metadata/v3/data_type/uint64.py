"""
Zarr v3 `uint64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.definition._definition import ConstantFillValue
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    multi_byte,
)
from zarr_metadata.v3._types.data_type import (
    UINT64_DATA_TYPE_NAME,
    Uint64DataTypeName,
    Uint64FillValue,
)

UINT64_DATA_TYPE: Final = DataTypeDefinition(
    name=UINT64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Uint64FillValue,
    fill_value_default=ConstantFillValue(0),
    storage=multi_byte,
    number_class="integral",
)
"""The `uint64` data type. It is written as a bare name and has no configuration. Its fill value is an integer in its range."""


__all__ = [
    "UINT64_DATA_TYPE",
    "UINT64_DATA_TYPE_NAME",
    "Uint64DataTypeName",
    "Uint64FillValue",
]
