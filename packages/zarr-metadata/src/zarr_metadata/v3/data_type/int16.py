"""
Zarr v3 `int16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Annotated, Final, Literal

from annotated_types import Interval

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, multi_byte

INT16_DATA_TYPE_NAME: Final = "int16"
"""The `data_type` value for the `int16` type."""

Int16DataTypeName = Literal["int16"]
"""Literal type of the `data_type` field for `int16`."""

Int16FillValue = Annotated[int, Interval(ge=-(2**15), le=2**15 - 1)]
"""Permitted JSON shape of the `fill_value` field for `int16`: a JSON integer in [-32768, 32767]."""


INT16_DATA_TYPE: Final = DataTypeDefinition(
    name=INT16_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int16FillValue,
    storage=multi_byte,
)
"""The `int16` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "INT16_DATA_TYPE",
    "INT16_DATA_TYPE_NAME",
    "Int16DataTypeName",
    "Int16FillValue",
]
