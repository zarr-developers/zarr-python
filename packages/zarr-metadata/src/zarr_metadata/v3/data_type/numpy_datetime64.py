"""
Zarr `numpy.datetime64` data type (zarr-extensions).

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md
"""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, NotRequired

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.model._validation import MetadataValidationError
from zarr_metadata.v3._entity import (
    StorageClass,
    problem,
)
from zarr_metadata.v3.data_type._families import (
    NUMPY_TIME_MAX_SCALE_FACTOR,
    NumpyTimeDataType,
    NumpyTimeUnit,
)

NUMPY_DATETIME64_DATA_TYPE_NAME: Final = "numpy.datetime64"
"""The `name` field value of the `numpy.datetime64` data type."""

NumpyDatetime64DataTypeName = Literal["numpy.datetime64"]
"""Literal type of the `name` field of the `numpy.datetime64` data type."""


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


@dataclass(frozen=True)
class NumpyDatetime64DataType(NumpyTimeDataType[NumpyDatetime64]):
    """The `numpy.datetime64` data type, coerced from its metadata."""

    unit: NumpyTimeUnit
    scale_factor: int

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    identifier: ClassVar[str] = NUMPY_DATETIME64_DATA_TYPE_NAME

    def __post_init__(self) -> None:
        if not 1 <= self.scale_factor <= NUMPY_TIME_MAX_SCALE_FACTOR:
            raise MetadataValidationError(
                problem(
                    ("scale_factor",),
                    f"expected an integer in [1, {NUMPY_TIME_MAX_SCALE_FACTOR}], "
                    f"got {self.scale_factor}",
                    "invalid_value",
                )
            )

    def to_json(self) -> NumpyDatetime64:
        return {
            "name": "numpy.datetime64",
            "configuration": {"unit": self.unit, "scale_factor": self.scale_factor},
        }
