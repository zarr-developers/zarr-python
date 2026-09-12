"""
Zarr `numpy.datetime64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/numpy.datetime64
"""

from collections.abc import Mapping
from typing import Final, Literal

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.v3.data_type._numpy_time import (
    NumpyTimeUnit,
    numpy_time_configuration,
    numpy_time_unit,
)

NUMPY_DATETIME64_DATA_TYPE_NAME: Final = "numpy.datetime64"
"""The `name` field value of the `numpy.datetime64` data type."""

NumpyDatetime64DataTypeName = Literal["numpy.datetime64"]
"""Literal type of the `name` field of the `numpy.datetime64` data type."""


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

    unit: ReadOnly[NumpyTimeUnit]
    scale_factor: ReadOnly[int]


def numpy_datetime64_configuration(value: Mapping[str, object]) -> NumpyDatetime64Configuration:
    """Validate `value` as a `numpy.datetime64` configuration and normalize it.

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
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
    "numpy_datetime64_configuration",
    "numpy_time_unit",
]
