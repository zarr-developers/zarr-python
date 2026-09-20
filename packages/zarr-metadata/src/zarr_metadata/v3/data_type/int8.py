"""
Zarr v3 `int8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import MetadataEntity

INT8_DATA_TYPE_NAME: Final = "int8"
"""The `data_type` value for the `int8` type."""

Int8DataTypeName = Literal["int8"]
"""Literal type of the `data_type` field for `int8`."""

Int8FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int8`: a JSON integer in [-128, 127]."""


__all__ = [
    "INT8_DATA_TYPE_NAME",
    "Int8DataType",
    "Int8DataTypeName",
    "Int8FillValue",
]


@dataclass(frozen=True)
class Int8DataType(MetadataEntity):
    """The `int8` data type. The name says everything."""

    identifier: ClassVar[str] = INT8_DATA_TYPE_NAME
