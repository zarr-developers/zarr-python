"""
Zarr v3 `uint32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import MetadataEntity

UINT32_DATA_TYPE_NAME: Final = "uint32"
"""The `data_type` value for the `uint32` type."""

Uint32DataTypeName = Literal["uint32"]
"""Literal type of the `data_type` field for `uint32`."""

Uint32FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint32`: a JSON integer in [0, 2**32 - 1]."""


__all__ = [
    "UINT32_DATA_TYPE_NAME",
    "Uint32DataType",
    "Uint32DataTypeName",
    "Uint32FillValue",
]


@dataclass(frozen=True)
class Uint32DataType(MetadataEntity):
    """The `uint32` data type. The name says everything."""

    identifier: ClassVar[str] = UINT32_DATA_TYPE_NAME
