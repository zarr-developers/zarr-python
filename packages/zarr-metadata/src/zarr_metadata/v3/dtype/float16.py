"""
Zarr v3 `float16` data type.

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

import re
from typing import Final, Literal, NewType

FLOAT16_DTYPE_NAME: Final = "float16"
"""The `data_type` value for the `float16` type."""

Float16DTypeName = Literal["float16"]
"""Literal type of the `data_type` field for `float16`."""

Float16SpecialFillValue = Literal["NaN", "Infinity", "-Infinity"]
"""Named non-finite fill values permitted by the spec for IEEE 754 floats."""

HexFloat16 = NewType("HexFloat16", str)
"""A 6-character hex string (`0x` + 4 hex digits) encoding the
unsigned-integer representation of a float16."""

_HEX_FLOAT16_RE: Final = re.compile(r"^0x[0-9a-fA-F]{4}$")


def hex_float16(value: str) -> HexFloat16:
    """Validate `value` as a HexFloat16 and brand it.

    Raises ValueError if `value` is not exactly `0x` followed by 4 hex
    digits.
    """
    if not _HEX_FLOAT16_RE.fullmatch(value):
        raise ValueError(f"Expected '0x' followed by 4 hex digits, got {value!r}")
    return HexFloat16(value)


Float16FillValue = float | int | Float16SpecialFillValue | HexFloat16
"""Permitted JSON shape of the `fill_value` field for `float16`.

Either a JSON number, one of the named non-finite sentinels (`"NaN"`,
`"Infinity"`, `"-Infinity"`), or a `HexFloat16` (`0xYYYY` string encoding
the unsigned-integer representation of the IEEE 754 value).
"""


__all__ = [
    "FLOAT16_DTYPE_NAME",
    "Float16DTypeName",
    "Float16FillValue",
    "Float16SpecialFillValue",
    "HexFloat16",
    "hex_float16",
]
