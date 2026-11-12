"""
Zarr v3 `int64` data type.

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
    INT64_DATA_TYPE_NAME,
    Int64DataTypeName,
    Int64FillValue,
)

INT64_DATA_TYPE: Final = DataTypeDefinition(
    name=INT64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int64FillValue,
    fill_value_default=ConstantFillValue(0),
    storage=multi_byte,
)
"""The `int64` data type. It is written as a bare name and has no configuration. Its fill value is an integer in its range."""


__all__ = [
    "INT64_DATA_TYPE",
    "INT64_DATA_TYPE_NAME",
    "Int64DataTypeName",
    "Int64FillValue",
]
