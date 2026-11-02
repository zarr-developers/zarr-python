"""
Zarr `numpy.datetime64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md
"""

from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._numpy import (
    NumpyTimeScaleFactor,
    NumpyTimeTicks,
    numpy_time_fill_value_canonical,
)
from zarr_metadata.v3._definition import DataTypeDefinition
from zarr_metadata.v3.data_type._numpy_time import (
    NUMPY_TIME_UNIT,
    NumpyTimeUnit,
    numpy_time_storage,
)

NUMPY_DATETIME64_DATA_TYPE_NAME: Final = "numpy.datetime64"
"""The `name` field value of the `numpy.datetime64` data type."""

NumpyDatetime64DataTypeName = Literal["numpy.datetime64"]
"""Literal type of the `name` field of the `numpy.datetime64` data type."""


class NumpyDatetime64Configuration(TypedDict, closed=True):
    """Configuration for the `numpy.datetime64` data type."""

    unit: ReadOnly[NumpyTimeUnit]
    """A string encoding a unit of time.

    `"us"` and `"μs"` are equivalent spellings of microseconds; `"generic"`
    means the unit is unspecified.
    """
    scale_factor: ReadOnly[NumpyTimeScaleFactor]
    """The multiplier relative to the unit.

    An integer in `[1, 2147483647]`.
    """


class NumpyDatetime64(TypedDict, closed=True):
    """`numpy.datetime64` data type metadata."""

    name: NumpyDatetime64DataTypeName
    """The data type's name, `"numpy.datetime64"`."""
    configuration: NumpyDatetime64Configuration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


NumpyDatetime64FillValue = NumpyTimeTicks | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.datetime64`.

Either a JSON integer (count of `unit * scale_factor` since the epoch),
or the string `"NaT"` (equivalent to the integer `-2**63`).
"""

NUMPY_DATETIME64_DATA_TYPE: Final = DataTypeDefinition(
    name=NUMPY_DATETIME64_DATA_TYPE_NAME,
    configuration=NumpyDatetime64Configuration,
    fill_value=NumpyDatetime64FillValue,
    fill_value_canonical=numpy_time_fill_value_canonical,
    storage=numpy_time_storage,
)
"""The `numpy.datetime64` data type."""


__all__ = [
    "NUMPY_DATETIME64_DATA_TYPE",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NUMPY_TIME_UNIT",
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
]
