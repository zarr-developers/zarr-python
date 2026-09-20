"""
Zarr v3 `uint16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import DataTypeEntity, StorageClass

UINT16_DATA_TYPE_NAME: Final = "uint16"
"""The `data_type` value for the `uint16` type."""

Uint16DataTypeName = Literal["uint16"]
"""Literal type of the `data_type` field for `uint16`."""

Uint16FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint16`: a JSON integer in [0, 65535]."""


__all__ = [
    "UINT16_DATA_TYPE_NAME",
    "Uint16DataType",
    "Uint16DataTypeName",
    "Uint16FillValue",
]


@dataclass(frozen=True)
class Uint16DataType(DataTypeEntity):
    """The `uint16` data type. The name says everything."""

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    identifier: ClassVar[str] = UINT16_DATA_TYPE_NAME
