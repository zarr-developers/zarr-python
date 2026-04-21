"""
Zarr v3 `complex64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3.data_type.float32 import Float32FillValue

COMPLEX64_DTYPE_NAME: Final = "complex64"
"""The `data_type` value for the `complex64` type."""

Complex64DTypeName = Literal["complex64"]
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
    "COMPLEX64_DTYPE_NAME",
    "Complex64Component",
    "Complex64DTypeName",
    "Complex64FillValue",
]
