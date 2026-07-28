"""
Zarr v3 `r<N>` raw-bytes data type (parameterised by bit count).

The `data_type` value is a string of the form `r<N>` where `N` is a
positive multiple of 8 (e.g. `r8`, `r16`, `r24`).

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
"""

import re
from typing import Final, NewType

RawBytesDataTypeName = NewType("RawBytesDataTypeName", str)
"""A spec-conformant `r<N>` raw-bytes name (e.g. `"r8"`, `"r16"`)."""

_RAW_BYTES_RE: Final = re.compile(r"^r(\d+)$")


def raw_bytes_dtype_name(value: str) -> RawBytesDataTypeName:
    """Validate `value` as a `r<N>` raw-bytes name and brand it.

    Raises ValueError if `value` is not `r` followed by a positive
    multiple of 8.
    """
    match = _RAW_BYTES_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"Expected 'r' followed by a positive integer, got {value!r}")
    bits = int(match.group(1))
    if bits == 0 or bits % 8 != 0:
        raise ValueError(f"Expected 'r<N>' where N is a positive multiple of 8, got {value!r}")
    return RawBytesDataTypeName(value)


RawBytesFillValue = tuple[int, ...]
"""Permitted JSON shape of the `fill_value` field for `r<N>`.

A JSON array of N/8 integers in `[0, 255]` (one per byte).
"""


__all__ = [
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "raw_bytes_dtype_name",
]
