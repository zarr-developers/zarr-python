"""A byte value, as used in the fill values of raw bits and of `bytes`, and the rules of a fill value written in base64.

Each byte value is an integer in `[0, 255]`
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L97-L99;
the reference for the `bytes` type is https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md?plain=1#L8).
"""

import base64
from collections.abc import Iterator
from typing import Annotated

from annotated_types import Interval

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata._numpy import base64_bytes

ByteValue = Annotated[int, Interval(ge=0, le=255)]
"""One byte of a fill value: a JSON integer in `[0, 255]`."""


def base64_problems(value: str) -> Iterator[ValidationProblem]:
    """Check that a fill value written as a string is standard-alphabet base64."""
    try:
        base64_bytes(value)
    except ValueError:
        yield ValidationProblem(
            (), f"expected standard-alphabet base64, got {shown(value)}", "invalid_value"
        )


def base64_decoded(value: str) -> bytes | None:
    """The bytes that `value` encodes in standard-alphabet base64; None when it is not such base64."""
    try:
        return base64.b64decode(base64_bytes(value))
    except ValueError:
        return None


def base64_spelling(data: bytes) -> str:
    """`data` in the single spelling of a base64 fill value, standard-alphabet base64: `"QR=="` decodes to the byte that `"QQ=="` spells."""
    return base64.b64encode(data).decode("ascii")


__all__ = ["ByteValue", "base64_decoded", "base64_problems", "base64_spelling"]
