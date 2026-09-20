"""
Zarr v3 `uint8` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import MetadataEntity

UINT8_DATA_TYPE_NAME: Final = "uint8"
"""The `data_type` value for the `uint8` type."""

Uint8DataTypeName = Literal["uint8"]
"""Literal type of the `data_type` field for `uint8`."""

Uint8FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint8`: a JSON integer in [0, 255]."""


__all__ = [
    "UINT8_DATA_TYPE_NAME",
    "Uint8DataType",
    "Uint8DataTypeName",
    "Uint8FillValue",
]


@dataclass(frozen=True)
class Uint8DataType(MetadataEntity):
    """The `uint8` data type. The name says everything."""

    identifier: ClassVar[str] = UINT8_DATA_TYPE_NAME
