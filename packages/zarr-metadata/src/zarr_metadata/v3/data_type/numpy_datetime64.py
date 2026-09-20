"""
Zarr `numpy.datetime64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired, cast

from typing_extensions import ReadOnly, TypedDict, Unpack

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    MemberTypes,
    StorageClass,
    ValueRoutine,
    is_int,
    one_of,
    problem,
)
from zarr_metadata.v3.data_type._families import NumpyTimeDataType
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NUMPY_TIME_MAX_SCALE_FACTOR,
    NUMPY_TIME_UNIT,
)

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

__all__ = [
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataType",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
]


def _value_problems(
    **members: Unpack[NumpyDatetime64Configuration],
) -> tuple[ValidationProblem, ...]:
    """`scale_factor` counts units per step, so it is positive.

    The upper bound is numpy's: the field is a signed 32-bit integer.
    """
    scale_factor = members["scale_factor"]
    if not 1 <= scale_factor <= NUMPY_TIME_MAX_SCALE_FACTOR:
        return problem(
            ("scale_factor",),
            f"expected an integer in [1, {NUMPY_TIME_MAX_SCALE_FACTOR}], got {scale_factor}",
            "invalid_value",
        )
    return ()


@dataclass(frozen=True)
class NumpyDatetime64DataType(NumpyTimeDataType):
    """The `numpy.datetime64` data type, coerced from its metadata."""

    unit: NumpyTimeUnit
    scale_factor: int

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    identifier: ClassVar[str] = NUMPY_DATETIME64_DATA_TYPE_NAME

    configuration_required: ClassVar[bool] = True
    member_types: ClassVar[MemberTypes] = {
        "unit": (True, one_of(NUMPY_TIME_UNIT)),
        "scale_factor": (True, is_int),
    }

    value_problems: ClassVar[ValueRoutine] = staticmethod(_value_problems)

    def to_json(self) -> NumpyDatetime64:
        return cast("NumpyDatetime64", super().to_json())
