"""Composition rules shared by NumPy datetime and timedelta data types."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.rules._registry import entity_rule
from zarr_metadata.v3._extension_points import DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE_NAME

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.rules._spec import ArrayParts

_ARRAY_V3 = "zarr_v3_array"
_MAX_SCALE_FACTOR = 2**31 - 1
_SCALE_FACTOR = frozenset({"scale_factor"})


def _scale_factor_is_in_range(
    configuration: Mapping[str, object], document: Mapping[str, object], incoming: ArrayParts | None
) -> tuple[ValidationProblem, ...]:
    scale_factor = cast("int", configuration["scale_factor"])
    if 1 <= scale_factor <= _MAX_SCALE_FACTOR:
        return ()
    return (
        ValidationProblem(
            ("scale_factor",),
            f"expected an integer in [1, {_MAX_SCALE_FACTOR}], got {scale_factor}",
            "invalid_value",
        ),
    )


entity_rule(_ARRAY_V3, DATA_TYPE, NUMPY_DATETIME64_DATA_TYPE_NAME, reads=_SCALE_FACTOR)(
    _scale_factor_is_in_range
)
entity_rule(_ARRAY_V3, DATA_TYPE, NUMPY_TIMEDELTA64_DATA_TYPE_NAME, reads=_SCALE_FACTOR)(
    _scale_factor_is_in_range
)
