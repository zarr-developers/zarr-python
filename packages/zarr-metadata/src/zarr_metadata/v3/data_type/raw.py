"""
Zarr v3 `r<N>` raw-bytes data type (parameterised by bit count).

The `data_type` value is a string of the form `r<N>` where `N` is a
positive multiple of 8 (e.g. `r8`, `r16`, `r24`).

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47; fill value: https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L97-L99)
"""

import re
from collections.abc import Iterator
from typing import Final, NewType

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import DataTypeDefinition, EmptyConfiguration

RawBytesDataTypeName = NewType("RawBytesDataTypeName", str)
"""A spec-conformant `r<N>` raw-bytes name (e.g. `"r8"`, `"r16"`).

"raw bits, variable size given by *, limited to be a multiple of 8":
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47
"""

RAW_BYTES_FAMILY: Final = "r<N>"
"""The name the raw-bytes family is filed under in a scope.

Spelled as the spec writes the family; the angle brackets keep it from
ever being a real data type's name.
"""

RAW_BYTES_NAME_PATTERN: Final = re.compile(r"^r([0-9]+)$")
"""The *shape* of a raw-bytes data type name, not its validity.

ASCII digits only: `\\d` would also match every other Unicode decimal, so
`r\uff11\uff16` would be read as sixteen bits and a genuine third-party
name spelled that way would be folded into this family.

Matches every `r<N>` spelling including malformed ones (`r0`, `r12`), so
that a misspelled member of this family is recognized as belonging to it
and reported as a misspelling, rather than passing as an unknown
third-party extension. `raw_bytes_dtype_name` applies the validity rule
on top.
"""


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


def _claims(name: str) -> bool:
    """Every `r<N>` spelling, valid or not: a malformed one is this family's to report."""
    return RAW_BYTES_NAME_PATTERN.fullmatch(name) is not None


def _name_rules(name: str) -> Iterator[ValidationProblem]:
    """ "raw bits, variable size given by *, limited to be a multiple of 8" -- and zero bits is not a type."""
    try:
        raw_bytes_dtype_name(name)
    except ValueError as error:
        yield ValidationProblem((), str(error), "invalid_value")


RAW_BYTES_DATA_TYPE: Final = DataTypeDefinition(
    name=RAW_BYTES_FAMILY,
    configuration=EmptyConfiguration,
    names=_claims,
    name_rules=_name_rules,
)
"""The raw-bytes family: one definition for every `r<N>`, whose name carries its width.

The spelling is kept rather than the bit count, so a document comes back
as it went in: `r008` is a valid and distinct way of writing `r8`.
"""


__all__ = [
    "RAW_BYTES_DATA_TYPE",
    "RAW_BYTES_FAMILY",
    "RAW_BYTES_NAME_PATTERN",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "raw_bytes_dtype_name",
]
