"""
Zarr v3 `int16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import StorageClass
from zarr_metadata.v3.data_type._families import IntegerDataType

INT16_DATA_TYPE_NAME: Final = "int16"
"""The `data_type` value for the `int16` type."""

Int16DataTypeName = Literal["int16"]
"""Literal type of the `data_type` field for `int16`."""

Int16FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int16`: a JSON integer in [-32768, 32767]."""


__all__ = [
    "INT16_DATA_TYPE_NAME",
    "Int16DataType",
    "Int16DataTypeName",
    "Int16FillValue",
]


@dataclass(frozen=True)
class Int16DataType(IntegerDataType[Int16DataTypeName]):
    """The `int16` data type. The name says everything."""

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    bounds: ClassVar[tuple[int, int]] = (-32768, 32767)
    identifier: ClassVar[str] = INT16_DATA_TYPE_NAME

    def to_json(self) -> Int16DataTypeName:
        return "int16"
