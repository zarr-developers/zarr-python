"""
Zarr `fixed_length_utf32` data type (fixed-length UTF-32 strings, zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/data-types/fixed_length_utf32/README.md
"""

from collections.abc import Iterator
from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import DataTypeDefinition, multi_byte

FIXED_LENGTH_UTF32_DATA_TYPE_NAME: Final = "fixed_length_utf32"
"""The `name` field value of the `fixed_length_utf32` data type."""

FixedLengthUtf32DataTypeName = Literal["fixed_length_utf32"]
"""Literal type of the `name` field of the `fixed_length_utf32` data type."""

_CODE_POINT_BYTES: Final = 4


class FixedLengthUtf32Configuration(TypedDict, closed=True):
    """
    Configuration for the `fixed_length_utf32` data type.

    Attributes
    ----------
    length_bytes
        The size in bytes of every value: a positive multiple of 4, since
        each code point takes 4 bytes.
    """

    length_bytes: ReadOnly[int]


class FixedLengthUtf32(TypedDict, closed=True):
    """`fixed_length_utf32` data type metadata."""

    name: FixedLengthUtf32DataTypeName
    configuration: FixedLengthUtf32Configuration
    must_understand: NotRequired[bool]


FixedLengthUtf32FillValue = str
"""Permitted JSON shape of the `fill_value` field for `fixed_length_utf32`: a JSON string of at most `length_bytes / 4` code points."""


def _rules(
    configuration: FixedLengthUtf32Configuration, nested: Nested
) -> Iterator[ValidationProblem]:
    """Check that `length_bytes` is a "positive integer, multiple of `4`" (https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/data-types/fixed_length_utf32/README.md?plain=1#L41)."""
    length_bytes = configuration["length_bytes"]
    if length_bytes <= 0 or length_bytes % _CODE_POINT_BYTES != 0:
        yield ValidationProblem(
            ("length_bytes",),
            f"expected a positive multiple of {_CODE_POINT_BYTES}, got {length_bytes}",
            "invalid_value",
        )


def _fill_value_rules(
    configuration: FixedLengthUtf32Configuration, nested: Nested, value: FixedLengthUtf32FillValue
) -> Iterator[ValidationProblem]:
    """Check that the fill value fits: "a JSON string of at most `length_bytes / 4` code points" (https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/data-types/fixed_length_utf32/README.md?plain=1#L124-L126)."""
    capacity = configuration["length_bytes"] // _CODE_POINT_BYTES
    if len(value) > capacity:
        yield ValidationProblem(
            (),
            f"expected at most {capacity} code points, got {len(value)}",
            "invalid_value",
        )


FIXED_LENGTH_UTF32_DATA_TYPE: Final = DataTypeDefinition(
    name=FIXED_LENGTH_UTF32_DATA_TYPE_NAME,
    configuration=FixedLengthUtf32Configuration,
    rules=_rules,
    fill_value=FixedLengthUtf32FillValue,
    fill_value_rules=_fill_value_rules,
    storage=multi_byte,
)
"""The `fixed_length_utf32` data type: NumPy's `<U` / `>U`.

Each code point is a 4-byte UTF-32 code unit, so the `bytes` codec "MUST
be configured with an explicit `endian` setting"
(https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/data-types/fixed_length_utf32/README.md?plain=1#L100-L103).
"""


__all__ = [
    "FIXED_LENGTH_UTF32_DATA_TYPE",
    "FIXED_LENGTH_UTF32_DATA_TYPE_NAME",
    "FixedLengthUtf32",
    "FixedLengthUtf32Configuration",
    "FixedLengthUtf32DataTypeName",
    "FixedLengthUtf32FillValue",
]
