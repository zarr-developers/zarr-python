"""
Zarr `numpy.timedelta64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.timedelta64/README.md
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    MemberTypes,
    StorageClass,
    is_int,
    one_of,
    problem,
)
from zarr_metadata.v3.data_type._families import NumpyTimeDataType

NUMPY_TIMEDELTA64_DATA_TYPE_NAME: Final = "numpy.timedelta64"
"""The `name` field value of the `numpy.timedelta64` data type."""

NumpyTimedelta64DataTypeName = Literal["numpy.timedelta64"]
"""Literal type of the `name` field of the `numpy.timedelta64` data type."""

NumpyTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes used by numpy.timedelta64."""

NUMPY_TIME_MAX_SCALE_FACTOR: Final = 2**31 - 1
"""The largest `scale_factor` numpy stores: the field is a signed int32."""

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


class NumpyTimedelta64Configuration(TypedDict, closed=True):
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


class NumpyTimedelta64(TypedDict, closed=True):
    """`numpy.timedelta64` data type metadata."""

    name: NumpyTimedelta64DataTypeName
    configuration: NumpyTimedelta64Configuration
    must_understand: NotRequired[bool]


NumpyTimedelta64FillValue = int | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.timedelta64`.

Either a JSON integer (a count of `unit * scale_factor`), or the string
`"NaT"` (equivalent to the integer `-2**63`).
"""

__all__ = [
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NUMPY_TIME_MAX_SCALE_FACTOR",
    "NUMPY_TIME_UNIT",
    "NumpyTimeUnit",
    "NumpyTimedelta64",
    "NumpyTimedelta64Configuration",
    "NumpyTimedelta64DataType",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
]


@dataclass(frozen=True)
class NumpyTimedelta64DataType(NumpyTimeDataType):
    """The `numpy.timedelta64` data type, coerced from its metadata."""

    unit: NumpyTimeUnit = "generic"
    scale_factor: int = 1

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    identifier: ClassVar[str] = NUMPY_TIMEDELTA64_DATA_TYPE_NAME

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {
        "unit": (True, one_of(NUMPY_TIME_UNIT)),
        "scale_factor": (True, is_int),
    }

    def problems(self) -> tuple[ValidationProblem, ...]:
        """`scale_factor` counts units per step, so it is positive.

        The upper bound is numpy's: the field is a signed 32-bit integer.
        """
        if not 1 <= self.scale_factor <= NUMPY_TIME_MAX_SCALE_FACTOR:
            return problem(
                ("scale_factor",),
                f"expected an integer in [1, {NUMPY_TIME_MAX_SCALE_FACTOR}], "
                f"got {self.scale_factor}",
                "invalid_value",
            )
        return ()

    def to_json(self) -> NumpyTimedelta64:
        return cast("NumpyTimedelta64", super().to_json())
