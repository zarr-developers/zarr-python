"""What the two numpy time types share: a unit, and how many of it one tick is.

Both types' configurations have these two members and the one rule on
them, so the rule is written here once and neither sibling imports it
from the other.
"""

from collections.abc import Iterator
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem

NUMPY_TIME_MAX_SCALE_FACTOR: Final = 2**31 - 1
"""The largest `scale_factor` numpy stores: the field is a signed int32."""


class NumpyTimeConfiguration(TypedDict):
    """The members both numpy time types' configurations have, read-only so either type fits."""

    unit: ReadOnly[str]
    scale_factor: ReadOnly[int]


def numpy_time_rules(configuration: NumpyTimeConfiguration) -> Iterator[ValidationProblem]:
    """`scale_factor` is a positive int32."""
    scale_factor = configuration["scale_factor"]
    if not 1 <= scale_factor <= NUMPY_TIME_MAX_SCALE_FACTOR:
        yield ValidationProblem(
            ("scale_factor",),
            f"expected an integer in [1, {NUMPY_TIME_MAX_SCALE_FACTOR}], got {scale_factor}",
            "invalid_value",
        )


__all__ = ["NUMPY_TIME_MAX_SCALE_FACTOR", "NumpyTimeConfiguration", "numpy_time_rules"]
