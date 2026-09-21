"""
Zarr v3 `uint64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import StorageClass
from zarr_metadata.v3.data_type._families import IntegerDataType

UINT64_DATA_TYPE_NAME: Final = "uint64"
"""The `data_type` value for the `uint64` type."""

Uint64DataTypeName = Literal["uint64"]
"""Literal type of the `data_type` field for `uint64`."""

Uint64FillValue = int
"""Permitted JSON shape of the `fill_value` field for `uint64`: a JSON integer in [0, 2**64 - 1]."""


__all__ = [
    "UINT64_DATA_TYPE_NAME",
    "Uint64DataType",
    "Uint64DataTypeName",
    "Uint64FillValue",
]


@dataclass(frozen=True)
class Uint64DataType(IntegerDataType[Uint64DataTypeName]):
    """The `uint64` data type. The name says everything."""

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    bounds: ClassVar[tuple[int, int]] = (0, 18446744073709551615)
    identifier: ClassVar[str] = UINT64_DATA_TYPE_NAME

    def to_json(self) -> Uint64DataTypeName:
        return "uint64"
