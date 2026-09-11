"""
Zarr `numpy.timedelta64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/numpy.timedelta64
"""

from typing import Final, Literal

from typing_extensions import ReadOnly, TypedDict

NUMPY_TIMEDELTA64_DATA_TYPE_NAME: Final = "numpy.timedelta64"
"""The `name` field value of the `numpy.timedelta64` data type."""

NumpyTimedelta64DataTypeName = Literal["numpy.timedelta64"]
"""Literal type of the `name` field of the `numpy.timedelta64` data type."""

NumpyTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes used by numpy.timedelta64."""

NUMPY_TIME_UNIT: Final = (
    "Y",
    "M",
    "W",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "μs",
    "ns",
    "ps",
    "fs",
    "as",
    "generic",
)
"""Runtime tuple of the permitted `numpy.timedelta64`/`numpy.datetime64` unit strings."""


class NumpyTimedelta64Configuration(TypedDict):
    """
    Configuration for the `numpy.timedelta64` data type.

    Attributes
    ----------
    unit
        A string encoding a unit of time.
    scale_factor
        The multiplier relative to the unit.
    """

    unit: ReadOnly[NumpyTimeUnit]
    scale_factor: ReadOnly[int]


class NumpyTimedelta64(TypedDict):
    """`numpy.timedelta64` data type metadata."""

    name: NumpyTimedelta64DataTypeName
    configuration: NumpyTimedelta64Configuration


NumpyTimedelta64FillValue = int | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.timedelta64`.

Either a JSON integer (a count of `unit * scale_factor`), or the string
`"NaT"` (equivalent to the integer `-2**63`).
"""

__all__ = [
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NUMPY_TIME_UNIT",
    "NumpyTimeUnit",
    "NumpyTimedelta64",
    "NumpyTimedelta64Configuration",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
]
