"""
Zarr v3 `int8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Annotated, Final, Literal

from annotated_types import Interval

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, single_byte

INT8_DATA_TYPE_NAME: Final = "int8"
"""The `data_type` value for the `int8` type."""

Int8DataTypeName = Literal["int8"]
"""Literal type of the `data_type` field for `int8`."""

Int8FillValue = Annotated[int, Interval(ge=-(2**7), le=2**7 - 1)]
"""Permitted JSON shape of the `fill_value` field for `int8`: a JSON integer in [-128, 127]."""


INT8_DATA_TYPE: Final = DataTypeDefinition(
    name=INT8_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Int8FillValue,
    storage=single_byte,
)
"""The `int8` data type: a bare name, with nothing to configure; its fill value an integer in its range."""


__all__ = [
    "INT8_DATA_TYPE",
    "INT8_DATA_TYPE_NAME",
    "Int8DataTypeName",
    "Int8FillValue",
]
