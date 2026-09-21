"""
Zarr v3 `complex64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal

from zarr_metadata.v3._entity import StorageClass
from zarr_metadata.v3.data_type._families import ComplexDataType, FloatDataType
from zarr_metadata.v3.data_type.float32 import Float32DataType, Float32FillValue

COMPLEX64_DATA_TYPE_NAME: Final = "complex64"
"""The `data_type` value for the `complex64` type."""

Complex64DataTypeName = Literal["complex64"]
"""Literal type of the `data_type` field for `complex64`."""

Complex64Component = Float32FillValue
"""One real or imaginary component of a `complex64` fill value.

Same shape as a `float32` fill value: a JSON number, a named sentinel,
or a `HexFloat32` string.
"""

Complex64FillValue = tuple[Complex64Component, Complex64Component]
"""Permitted JSON shape of the `fill_value` field for `complex64`.

A two-element JSON array `[real, imag]` where each component is a
`Complex64Component`.
"""


__all__ = [
    "COMPLEX64_DATA_TYPE_NAME",
    "Complex64Component",
    "Complex64DataType",
    "Complex64DataTypeName",
    "Complex64FillValue",
]


@dataclass(frozen=True)
class Complex64DataType(ComplexDataType[Complex64DataTypeName]):
    """The `complex64` data type. The name says everything."""

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    component: ClassVar[type[FloatDataType]] = Float32DataType
    identifier: ClassVar[str] = COMPLEX64_DATA_TYPE_NAME
