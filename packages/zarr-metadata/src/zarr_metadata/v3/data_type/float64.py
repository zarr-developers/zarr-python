"""
Zarr v3 `float64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

import re
from typing import Final, Literal, NewType

FLOAT64_DATA_TYPE_NAME: Final = "float64"
"""The `data_type` value for the `float64` type."""

Float64DataTypeName = Literal["float64"]
"""Literal type of the `data_type` field for `float64`."""

Float64SpecialFillValue = Literal["NaN", "Infinity", "-Infinity"]
"""Named non-finite fill values permitted by the spec for IEEE 754 floats."""

HexFloat64 = NewType("HexFloat64", str)
"""An 18-character hex string (`0x` + 16 hex digits) encoding the
unsigned-integer representation of a float64."""

_HEX_FLOAT64_RE: Final = re.compile(r"^0x[0-9a-fA-F]{16}$")


def hex_float64(value: str) -> HexFloat64:
    """Validate `value` as a HexFloat64 and brand it.

    Raises ValueError if `value` is not exactly `0x` followed by 16 hex
    digits.
    """
    if not _HEX_FLOAT64_RE.fullmatch(value):
        raise ValueError(f"Expected '0x' followed by 16 hex digits, got {value!r}")
    return HexFloat64(value)


Float64FillValue = float | int | Float64SpecialFillValue | HexFloat64
"""Permitted JSON shape of the `fill_value` field for `float64`.

Either a JSON number, one of the named non-finite sentinels (`"NaN"`,
`"Infinity"`, `"-Infinity"`), or a `HexFloat64` (`0xYYYYYYYYYYYYYYYY`
string encoding the unsigned-integer representation of the IEEE 754
value).
"""

CANONICAL_NAN_HEX_FLOAT64: Final = "0x7ff8000000000000"
"""Canonical hex form of the float64 NaN sentinel `"NaN"`.

Per spec the named `"NaN"` sentinel denotes the float with sign=0, the
most significant mantissa bit set, and all other mantissa bits zero
(the IEEE 754 default quiet NaN). Other NaN bit patterns must be
encoded with the explicit hex-string form.
"""

CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64: Final = "0x7ff0000000000000"
"""Canonical hex form of the float64 `"Infinity"` sentinel."""

CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64: Final = "0xfff0000000000000"
"""Canonical hex form of the float64 `"-Infinity"` sentinel."""


__all__ = [
    "CANONICAL_NAN_HEX_FLOAT64",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64",
    "FLOAT64_DATA_TYPE_NAME",
    "Float64DataTypeName",
    "Float64FillValue",
    "Float64SpecialFillValue",
    "HexFloat64",
    "hex_float64",
]
