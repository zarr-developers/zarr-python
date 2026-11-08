"""
Zarr `numpy.timedelta64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.timedelta64/README.md
"""

from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._numpy import (
    NumpyTimeScaleFactor,
    NumpyTimeTicks,
    numpy_time_fill_value_canonical,
)
from zarr_metadata.v3._definition import DataTypeDefinition
from zarr_metadata.v3.data_type._numpy_time import NumpyTimeUnit, numpy_time_storage

NUMPY_TIMEDELTA64_DATA_TYPE_NAME: Final = "numpy.timedelta64"
"""The `name` field value of the `numpy.timedelta64` data type."""

NumpyTimedelta64DataTypeName = Literal["numpy.timedelta64"]
"""Literal type of the `name` field of the `numpy.timedelta64` data type."""


class NumpyTimedelta64Configuration(TypedDict, closed=True):
    """Configuration for the `numpy.timedelta64` data type."""

    unit: ReadOnly[NumpyTimeUnit]
    """A string encoding a unit of time.

    `"us"` and `"μs"` are equivalent spellings of microseconds; `"generic"`
    means the unit is unspecified.
    """
    scale_factor: ReadOnly[NumpyTimeScaleFactor]
    """The multiplier relative to the unit.

    An integer in `[1, 2147483647]`.
    """


class NumpyTimedelta64(TypedDict, closed=True):
    """`numpy.timedelta64` data type metadata."""

    name: NumpyTimedelta64DataTypeName
    """The data type's name, `"numpy.timedelta64"`."""
    configuration: NumpyTimedelta64Configuration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


NumpyTimedelta64FillValue = NumpyTimeTicks | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.timedelta64`.

Either a JSON integer (a count of `unit * scale_factor`), or the string
`"NaT"` (equivalent to the integer `-2**63`).
"""

NUMPY_TIMEDELTA64_DATA_TYPE: Final = DataTypeDefinition(
    name=NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    configuration=NumpyTimedelta64Configuration,
    fill_value=NumpyTimedelta64FillValue,
    fill_value_canonical=numpy_time_fill_value_canonical,
    storage=numpy_time_storage,
)
"""The `numpy.timedelta64` data type."""


__all__ = [
    "NUMPY_TIMEDELTA64_DATA_TYPE",
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NumpyTimedelta64",
    "NumpyTimedelta64Configuration",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
]
