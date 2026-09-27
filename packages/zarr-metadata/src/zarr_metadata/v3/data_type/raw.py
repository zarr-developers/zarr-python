"""
Zarr v3 `r<N>` raw-bytes data type (parameterised by bit count).

The `data_type` value is a string of the form `r<N>` where `N` is a
positive multiple of 8 (e.g. `r8`, `r16`, `r24`).

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47; fill value: https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L97-L99)
"""

from collections.abc import Iterator
from typing import Final, NewType

from typing_extensions import TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import RAW_BYTES_NAME, RAW_BYTES_NAME_PATTERN, DataTypeDefinition

RawBytesDataTypeName = NewType("RawBytesDataTypeName", str)
"""A spec-conformant `r<N>` raw-bytes name (e.g. `"r8"`, `"r16"`).

"raw bits, variable size given by *, limited to be a multiple of 8":
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47
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


class RawBytesConfiguration(TypedDict, closed=True):
    """What a raw-bytes name carries: its size in bits, `{"bits": 16}` for `r16`.

    A document writes the name, never this object. The definition of `r*`
    reads a name into it, and the simplest spelling writes it back as the
    name.
    """

    bits: int


def _rules(configuration: RawBytesConfiguration) -> Iterator[ValidationProblem]:
    """ "raw bits, variable size given by *, limited to be a multiple of 8" -- and zero bits is not a type."""
    bits = configuration["bits"]
    if bits == 0 or bits % 8 != 0:
        yield ValidationProblem(
            ("bits",),
            f"expected a size in bits that is a positive multiple of 8, got {bits}",
            "invalid_value",
        )


RAW_BYTES_DATA_TYPE: Final = DataTypeDefinition(
    name=RAW_BYTES_NAME,
    configuration=RawBytesConfiguration,
    rules=_rules,
)
"""Raw bits, `r*`: one data type, whose name carries its size.

`r16` reads as `r*` with `{"bits": 16}`, and a size that is not a
positive multiple of 8 is a problem of the field that names it. The
simplest spelling writes the size back into the name, in decimal: `r008`
is `r8`.
"""


__all__ = [
    "RAW_BYTES_DATA_TYPE",
    "RawBytesConfiguration",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "raw_bytes_dtype_name",
]
