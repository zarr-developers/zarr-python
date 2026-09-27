"""
Zarr `numpy.datetime64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md
"""

from typing import Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.v3._definition import DataTypeDefinition
from zarr_metadata.v3.data_type._numpy_time import numpy_time_rules

NUMPY_DATETIME64_DATA_TYPE_NAME: Final = "numpy.datetime64"
"""The `name` field value of the `numpy.datetime64` data type."""

NumpyDatetime64DataTypeName = Literal["numpy.datetime64"]
"""Literal type of the `name` field of the `numpy.datetime64` data type."""

NumpyTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes used by numpy.datetime64."""


class NumpyDatetime64Configuration(TypedDict, closed=True):
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


class NumpyDatetime64(TypedDict, closed=True):
    """`numpy.datetime64` data type metadata."""

    name: NumpyDatetime64DataTypeName
    configuration: NumpyDatetime64Configuration
    must_understand: NotRequired[bool]


NumpyDatetime64FillValue = int | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.datetime64`.

Either a JSON integer (count of `unit * scale_factor` since the epoch),
or the string `"NaT"` (equivalent to the integer `-2**63`).
"""

NUMPY_DATETIME64_DATA_TYPE: Final = DataTypeDefinition(
    name=NUMPY_DATETIME64_DATA_TYPE_NAME,
    configuration=NumpyDatetime64Configuration,
    rules=numpy_time_rules,
)
"""The `numpy.datetime64` data type."""


__all__ = [
    "NUMPY_DATETIME64_DATA_TYPE",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
]
