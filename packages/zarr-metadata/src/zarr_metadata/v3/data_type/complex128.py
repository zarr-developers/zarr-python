"""
Zarr v3 `complex128` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3.data_type.float64 import Float64FillValue

COMPLEX128_DTYPE_NAME: Final = "complex128"
"""The `data_type` value for the `complex128` type."""

Complex128DTypeName = Literal["complex128"]
"""Literal type of the `data_type` field for `complex128`."""

Complex128Component = Float64FillValue
"""One real or imaginary component of a `complex128` fill value.

Same shape as a `float64` fill value: a JSON number, a named sentinel,
or a `HexFloat64` string.
"""

Complex128FillValue = tuple[Complex128Component, Complex128Component]
"""Permitted JSON shape of the `fill_value` field for `complex128`.

A two-element JSON array `[real, imag]` where each component is a
`Complex128Component`.
"""


__all__ = [
    "COMPLEX128_DTYPE_NAME",
    "Complex128Component",
    "Complex128DTypeName",
    "Complex128FillValue",
]
