"""
Zarr v3 `float64` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

import re
from typing import Final, Literal, NewType

FLOAT64_DTYPE_NAME: Final = "float64"
"""The `data_type` value for the `float64` type."""

Float64DTypeName = Literal["float64"]
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


__all__ = [
    "FLOAT64_DTYPE_NAME",
    "Float64DTypeName",
    "Float64FillValue",
    "Float64SpecialFillValue",
    "HexFloat64",
    "hex_float64",
]
