"""
Zarr v3 `bool` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import MetadataEntity

BOOL_DATA_TYPE_NAME: Final = "bool"
"""The `data_type` value for the `bool` type."""

BoolDataTypeName = Literal["bool"]
"""Literal type of the `data_type` field for `bool`."""

BoolFillValue = bool
"""Permitted JSON shape of the `fill_value` field for `bool`: a JSON boolean."""


__all__ = [
    "BOOL_DATA_TYPE_NAME",
    "BoolDataType",
    "BoolDataTypeName",
    "BoolFillValue",
]


@dataclass(frozen=True)
class BoolDataType(MetadataEntity):
    """The `bool` data type. The name says everything."""

    identifier: ClassVar[str] = BOOL_DATA_TYPE_NAME
