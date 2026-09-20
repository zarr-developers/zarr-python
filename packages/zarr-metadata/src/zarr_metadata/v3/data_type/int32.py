"""
Zarr v3 `int32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import MetadataEntity

INT32_DATA_TYPE_NAME: Final = "int32"
"""The `data_type` value for the `int32` type."""

Int32DataTypeName = Literal["int32"]
"""Literal type of the `data_type` field for `int32`."""

Int32FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int32`: a JSON integer in [-2**31, 2**31 - 1]."""


__all__ = [
    "INT32_DATA_TYPE_NAME",
    "Int32DataType",
    "Int32DataTypeName",
    "Int32FillValue",
]


@dataclass(frozen=True)
class Int32DataType(MetadataEntity):
    """The `int32` data type. The name says everything."""

    identifier: ClassVar[str] = INT32_DATA_TYPE_NAME
