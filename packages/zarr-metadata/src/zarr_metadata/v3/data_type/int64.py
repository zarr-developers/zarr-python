"""
Zarr v3 `int64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import DataTypeEntity, StorageClass

INT64_DATA_TYPE_NAME: Final = "int64"
"""The `data_type` value for the `int64` type."""

Int64DataTypeName = Literal["int64"]
"""Literal type of the `data_type` field for `int64`."""

Int64FillValue = int
"""Permitted JSON shape of the `fill_value` field for `int64`: a JSON integer in [-2**63, 2**63 - 1]."""


__all__ = [
    "INT64_DATA_TYPE_NAME",
    "Int64DataType",
    "Int64DataTypeName",
    "Int64FillValue",
]


@dataclass(frozen=True)
class Int64DataType(DataTypeEntity):
    """The `int64` data type. The name says everything."""

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    identifier: ClassVar[str] = INT64_DATA_TYPE_NAME
