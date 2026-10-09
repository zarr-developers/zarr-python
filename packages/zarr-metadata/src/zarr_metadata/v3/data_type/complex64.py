"""
Zarr v3 `complex64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

from typing import Final, Literal

from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration, multi_byte
from zarr_metadata.v3.data_type._float import (
    complex_fill_value_canonical,
    complex_fill_value_rules,
)
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE, Float32FillValue

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


COMPLEX64_DATA_TYPE: Final = DataTypeDefinition(
    name=COMPLEX64_DATA_TYPE_NAME,
    configuration=EmptyConfiguration,
    fill_value=Complex64FillValue,
    fill_value_rules=complex_fill_value_rules(FLOAT32_DATA_TYPE.fill_value_rules),
    fill_value_canonical=complex_fill_value_canonical(FLOAT32_DATA_TYPE.fill_value_canonical),
    storage=multi_byte,
)
"""The `complex64` data type: a bare name, with nothing to configure; its fill value a pair of `float32` components."""


__all__ = [
    "COMPLEX64_DATA_TYPE",
    "COMPLEX64_DATA_TYPE_NAME",
    "Complex64Component",
    "Complex64DataTypeName",
    "Complex64FillValue",
]
