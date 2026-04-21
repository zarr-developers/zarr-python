"""
Time-shaped Zarr v3 data types.

This module covers `numpy.datetime64` and `numpy.timedelta64`, both defined
in the zarr-extensions registry.

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
"""

from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

NUMPY_DATETIME64_DTYPE_NAME: Final = "numpy.datetime64"
"""The `name` field value of the `numpy.datetime64` data type."""

NumpyDatetime64DTypeName = Literal["numpy.datetime64"]
"""Literal type of the `name` field of the `numpy.datetime64` data type."""

NUMPY_TIMEDELTA64_DTYPE_NAME: Final = "numpy.timedelta64"
"""The `name` field value of the `numpy.timedelta64` data type."""

NumpyTimedelta64DTypeName = Literal["numpy.timedelta64"]
"""Literal type of the `name` field of the `numpy.timedelta64` data type."""

DateTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes used by numpy.datetime64 / numpy.timedelta64."""


class TimeConfig(TypedDict):
    """
    Configuration shared by `numpy.datetime64` and `numpy.timedelta64`.

    Attributes
    ----------
    unit
        A string encoding a unit of time.
    scale_factor
        The multiplier relative to the unit.
    """

    unit: ReadOnly[DateTimeUnit]
    scale_factor: ReadOnly[int]


class NumpyDatetime64(TypedDict):
    """`numpy.datetime64` data type metadata."""

    name: NumpyDatetime64DTypeName
    configuration: TimeConfig


class NumpyTimedelta64(TypedDict):
    """`numpy.timedelta64` data type metadata."""

    name: NumpyTimedelta64DTypeName
    configuration: TimeConfig


__all__ = [
    "NUMPY_DATETIME64_DTYPE_NAME",
    "NUMPY_TIMEDELTA64_DTYPE_NAME",
    "DateTimeUnit",
    "NumpyDatetime64",
    "NumpyDatetime64DTypeName",
    "NumpyTimedelta64",
    "NumpyTimedelta64DTypeName",
    "TimeConfig",
]
