"""
Zarr v3 `int8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import StorageClass
from zarr_metadata.v3.data_type._families import IntegerDataType

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
class Int8DataType(IntegerDataType):
    """The `int8` data type. The name says everything."""

    scalar_storage: ClassVar[StorageClass] = "single_byte"
    bounds: ClassVar[tuple[int, int]] = (-128, 127)
    identifier: ClassVar[str] = INT8_DATA_TYPE_NAME

    def to_json(self) -> Int8DataTypeName:
        return "int8"
