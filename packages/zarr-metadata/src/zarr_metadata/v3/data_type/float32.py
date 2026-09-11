"""
Zarr v3 `float32` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

import re
from typing import Final, Literal, NewType

FLOAT32_DATA_TYPE_NAME: Final = "float32"
"""The `data_type` value for the `float32` type."""

Float32DataTypeName = Literal["float32"]
"""Literal type of the `data_type` field for `float32`."""

Float32SpecialFillValue = Literal["NaN", "Infinity", "-Infinity"]
"""Named non-finite fill values permitted by the spec for IEEE 754 floats."""

HexFloat32 = NewType("HexFloat32", str)
"""A 10-character hex string (`0x` + 8 hex digits) encoding the
unsigned-integer representation of a float32."""

_HEX_FLOAT32_RE: Final = re.compile(r"^0x[0-9a-fA-F]{8}$")


def hex_float32(value: str) -> HexFloat32:
    """Validate `value` as a HexFloat32 and brand it.

    Raises ValueError if `value` is not exactly `0x` followed by 8 hex
    digits.
    """
    if not _HEX_FLOAT32_RE.fullmatch(value):
        raise ValueError(f"Expected '0x' followed by 8 hex digits, got {value!r}")
    return HexFloat32(value)


Float32FillValue = float | int | Float32SpecialFillValue | HexFloat32
"""Permitted JSON shape of the `fill_value` field for `float32`.

Either a JSON number, one of the named non-finite sentinels (`"NaN"`,
`"Infinity"`, `"-Infinity"`), or a `HexFloat32` (`0xYYYYYYYY` string
encoding the unsigned-integer representation of the IEEE 754 value).
"""

CANONICAL_NAN_HEX_FLOAT32: Final = "0x7fc00000"
"""Canonical hex form of the float32 NaN sentinel `"NaN"`.

Per spec the named `"NaN"` sentinel denotes the float with sign=0, the
most significant mantissa bit set, and all other mantissa bits zero
(the IEEE 754 default quiet NaN). Other NaN bit patterns must be
encoded with the explicit hex-string form.
"""

CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32: Final = "0x7f800000"
"""Canonical hex form of the float32 `"Infinity"` sentinel."""

CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32: Final = "0xff800000"
"""Canonical hex form of the float32 `"-Infinity"` sentinel."""


__all__ = [
    "CANONICAL_NAN_HEX_FLOAT32",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32",
    "FLOAT32_DATA_TYPE_NAME",
    "Float32DataTypeName",
    "Float32FillValue",
    "Float32SpecialFillValue",
    "HexFloat32",
    "hex_float32",
]
