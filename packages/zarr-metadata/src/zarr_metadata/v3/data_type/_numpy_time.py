"""What the two numpy time types share: a unit, and how many of it one tick is.

Both types' configurations have these two members and the one rule on
them, so the rule is written here once and neither sibling imports it
from the other. So is how their values are stored.
"""

from collections.abc import Iterator
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import Nested, multi_byte

NUMPY_TIME_MAX_SCALE_FACTOR: Final = 2**31 - 1
"""The largest `scale_factor` numpy stores: the field is a signed int32."""

numpy_time_storage: Final = multi_byte
"""Signed 64-bit integers, in the byte order the codecs say.

Each type "is compatible with any codec that supports arrays of signed
64-bit integers"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L120),
and "the endianness of numpy.datetime64 arrays is determined by the
configuration of the codecs defined in metadata"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L83-L85);
`numpy.timedelta64` says the same.
"""


class NumpyTimeConfiguration(TypedDict):
    """The members both numpy time types' configurations have, read-only so either type fits."""

    unit: ReadOnly[str]
    scale_factor: ReadOnly[int]


def numpy_time_rules(
    configuration: NumpyTimeConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    """`scale_factor` is a positive int32."""
    scale_factor = configuration["scale_factor"]
    if not 1 <= scale_factor <= NUMPY_TIME_MAX_SCALE_FACTOR:
        yield ValidationProblem(
            ("scale_factor",),
            f"expected an integer in [1, {NUMPY_TIME_MAX_SCALE_FACTOR}], got {scale_factor}",
            "invalid_value",
        )


def numpy_time_fill_value_rules(
    configuration: NumpyTimeConfiguration, nested: Nested, value: int | str
) -> Iterator[ValidationProblem]:
    """An integer fill value is a signed 64-bit one; `"NaT"` is the other form, which the shape admits.

    https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L109-L112
    """
    if isinstance(value, int) and not -(2**63) <= value <= 2**63 - 1:
        yield ValidationProblem(
            (), f'expected a signed 64-bit integer or "NaT", got {value}', "invalid_value"
        )


__all__ = [
    "NUMPY_TIME_MAX_SCALE_FACTOR",
    "NumpyTimeConfiguration",
    "numpy_time_fill_value_rules",
    "numpy_time_rules",
    "numpy_time_storage",
]
