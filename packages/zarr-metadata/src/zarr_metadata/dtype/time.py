"""
Time data type configuration (datetime64, timedelta64).
"""

from typing import Literal, TypedDict

from typing_extensions import ReadOnly

DateTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes used by numpy.datetime64 / numpy.timedelta64."""


class TimeConfig(TypedDict):
    """
    Configuration for numpy.timedelta64 or numpy.datetime64 in Zarr v3.

    Attributes
    ----------
    unit
        A string encoding a unit of time.
    scale_factor
        The multiplier relative to the unit.
    """

    unit: ReadOnly[DateTimeUnit]
    scale_factor: ReadOnly[int]


__all__ = [
    "DateTimeUnit",
    "TimeConfig",
]
