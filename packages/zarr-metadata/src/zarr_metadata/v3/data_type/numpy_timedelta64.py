"""
Zarr `numpy.timedelta64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/numpy.timedelta64
"""

from collections.abc import Mapping
from typing import Final, Literal

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.v3.data_type._numpy_time import (
    NUMPY_TIME_UNIT,
    NumpyTimeUnit,
    numpy_time_configuration,
    numpy_time_unit,
)

NUMPY_TIMEDELTA64_DATA_TYPE_NAME: Final = "numpy.timedelta64"
"""The `name` field value of the `numpy.timedelta64` data type."""

NumpyTimedelta64DataTypeName = Literal["numpy.timedelta64"]
"""Literal type of the `name` field of the `numpy.timedelta64` data type."""


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


def numpy_timedelta64_configuration(
    value: Mapping[str, object],
) -> NumpyTimedelta64Configuration:
    """Validate `value` as a `numpy.timedelta64` configuration and normalize it.

    The returned configuration spells the microsecond unit `"us"` even when the
    input used the equivalent `"μs"`.

    Raises TypeError if `unit` is not a string or `scale_factor` is not an
    integer. Raises ValueError if `value` does not have exactly the keys `unit`
    and `scale_factor`, if `unit` is not a `NumpyTimeUnit`, if `scale_factor` is
    outside `[1, 2**31 - 1]`, or if the `"generic"` unit is combined with a
    `scale_factor` other than 1 (NumPy's generic time type has no scale).
    """
    unit, scale_factor = numpy_time_configuration(value)
    return {"unit": unit, "scale_factor": scale_factor}


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
    "numpy_time_unit",
    "numpy_timedelta64_configuration",
]
