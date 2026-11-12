"""The rules of a fill value written in base64, which the data types whose fill values are bytes share.

The type of one byte of such a fill value, `ByteValue`, is in
`zarr_metadata.v3._types.data_type`.
"""

import base64
from collections.abc import Iterator

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata._numpy import base64_bytes


def base64_problems(value: str) -> Iterator[ValidationProblem]:
    """Check that a fill value written as a string is standard-alphabet base64."""
    try:
        base64_bytes(value)
    except ValueError:
        yield ValidationProblem(
            (), f"expected standard-alphabet base64, got {shown(value)}", "invalid_value"
        )


def base64_fill_value_rules(
    configuration: object, nested: object, value: object
) -> Iterator[ValidationProblem]:
    """Check that a fill value written as a string is standard-alphabet base64: a data type's `fill_value_rules`.

    `Base64Bytes` is a `NewType` of `str`, so a fill value annotated with
    it accepts any string. Pass this rule as the data type's
    `fill_value_rules` to reject a string that is not base64. A value of
    any other shape is not checked here, so the rule also serves a fill
    value that may be written another way, such as
    `tuple[ByteValue, ...] | Base64Bytes`. A rule that checks more, such
    as the number of bytes, yields from this one first.
    """
    if isinstance(value, str):
        yield from base64_problems(value)


def base64_decoded(value: str) -> bytes | None:
    """The bytes that `value` encodes in standard-alphabet base64; None when it is not such base64."""
    try:
        return base64.b64decode(base64_bytes(value))
    except ValueError:
        return None


def base64_spelling(data: bytes) -> str:
    """`data` in the single spelling of a base64 fill value, standard-alphabet base64: `"QR=="` decodes to the byte that `"QQ=="` spells."""
    return base64.b64encode(data).decode("ascii")


__all__ = [
    "base64_decoded",
    "base64_fill_value_rules",
    "base64_problems",
    "base64_spelling",
]
