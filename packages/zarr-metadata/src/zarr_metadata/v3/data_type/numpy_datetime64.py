"""
Zarr `numpy.datetime64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/numpy.datetime64
"""

from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

NUMPY_DATETIME64_DATA_TYPE_NAME: Final = "numpy.datetime64"
"""The `name` field value of the `numpy.datetime64` data type."""

NumpyDatetime64DataTypeName = Literal["numpy.datetime64"]
"""Literal type of the `name` field of the `numpy.datetime64` data type."""

DateTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes used by numpy.datetime64."""


class NumpyDatetime64Configuration(TypedDict):
    """
    Configuration for the `numpy.datetime64` data type.

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

    name: NumpyDatetime64DataTypeName
    configuration: NumpyDatetime64Configuration


NumpyDatetime64FillValue = int | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.datetime64`.

Either a JSON integer (count of `unit * scale_factor` since the epoch),
or the string `"NaT"` (equivalent to the integer `-2**63`).
"""


__all__ = [
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "DateTimeUnit",
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
]
