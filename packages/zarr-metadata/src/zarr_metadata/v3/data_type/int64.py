"""
Zarr v3 `int64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Annotated, Final, Literal

from annotated_types import Interval

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, multi_byte

INT64_DATA_TYPE_NAME: Final = "int64"
"""The `data_type` value for the `int64` type."""

Int64DataTypeName = Literal["int64"]
"""Literal type of the `data_type` field for `int64`."""

Int64FillValue = Annotated[int, Interval(ge=-(2**63), le=2**63 - 1)]
"""Permitted JSON shape of the `fill_value` field for `int64`: a JSON integer in [-2**63, 2**63 - 1]."""


INT64_DATA_TYPE: Final = DataTypeDefinition(
    name=INT64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int64FillValue,
    storage=multi_byte,
)
"""The `int64` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "INT64_DATA_TYPE",
    "INT64_DATA_TYPE_NAME",
    "Int64DataTypeName",
    "Int64FillValue",
]
