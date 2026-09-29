"""
Zarr v3 `int64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, multi_byte
from zarr_metadata.v3.data_type._integer import integer_fill_value_rules

INT64_DATA_TYPE_NAME: Final = "int64"
"""The `data_type` value for the `int64` type."""

Int64DataTypeName = Literal["int64"]
"""Literal type of the `data_type` field for `int64`."""

Int64FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int64`: a JSON integer in [-2**63, 2**63 - 1]."""


INT64_DATA_TYPE: Final = DataTypeDefinition(
    name=INT64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int64FillValue,
    fill_value_rules=integer_fill_value_rules(-(2**63), 2**63 - 1),
    storage=multi_byte,
)
"""The `int64` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "INT64_DATA_TYPE",
    "INT64_DATA_TYPE_NAME",
    "Int64DataTypeName",
    "Int64FillValue",
]
