"""
Zarr v3 `r<N>` raw-bytes data type (parameterised by bit count).

The `data_type` value is a string of the form `r<N>` where `N` is a
positive multiple of 8 (e.g. `r8`, `r16`, `r24`).

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47; fill value: https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L97-L99)
"""

import re
from typing import Final, NewType

RawBytesDataTypeName = NewType("RawBytesDataTypeName", str)
"""A spec-conformant `r<N>` raw-bytes name (e.g. `"r8"`, `"r16"`).

"raw bits, variable size given by *, limited to be a multiple of 8":
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47
"""

RAW_BYTES_NAME_PATTERN: Final = re.compile(r"^r(\d+)$")
"""The *shape* of a raw-bytes data type name, not its validity.

Matches every `r<N>` spelling including malformed ones (`r0`, `r12`), so
that a misspelled member of this family is recognized as belonging to it
and reported as a misspelling, rather than passing as an unknown
third-party extension. `raw_bytes_dtype_name` applies the validity rule
on top. Sole owner of this grammar: other modules match through it."""


def raw_bytes_dtype_name(value: str) -> RawBytesDataTypeName:
    """Validate `value` as a `r<N>` raw-bytes name and brand it.

    Raises ValueError if `value` is not `r` followed by a positive
    multiple of 8.
    """
    match = RAW_BYTES_NAME_PATTERN.fullmatch(value)
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
    "RAW_BYTES_NAME_PATTERN",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "raw_bytes_dtype_name",
]
