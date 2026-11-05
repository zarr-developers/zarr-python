"""
Zarr v3 `bool` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.definition._definition import EmptyConfiguration
from zarr_metadata.v3._definition import (
    DataTypeDefinition,
    single_byte,
)

BOOL_DATA_TYPE_NAME: Final = "bool"
"""The `data_type` value for the `bool` type."""

BoolDataTypeName = Literal["bool"]
"""Literal type of the `data_type` field for `bool`."""

BoolFillValue = bool
"""Permitted JSON shape of the `fill_value` field for `bool`: a JSON boolean."""


BOOL_DATA_TYPE: Final = DataTypeDefinition(
    name=BOOL_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=BoolFillValue,
    storage=single_byte,
)
"""The `bool` data type. It is written as a bare name and has no configuration."""


__all__ = [
    "BOOL_DATA_TYPE",
    "BOOL_DATA_TYPE_NAME",
    "BoolDataTypeName",
    "BoolFillValue",
]
