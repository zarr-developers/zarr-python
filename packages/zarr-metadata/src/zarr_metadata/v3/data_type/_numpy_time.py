"""
Vocabulary and validation shared by the `numpy.datetime64` and `numpy.timedelta64` data types.

This module is private (underscore-prefixed); the public names are re-exported by
`zarr_metadata.v3.data_type.numpy_datetime64` and
`zarr_metadata.v3.data_type.numpy_timedelta64`.
"""

from collections.abc import Mapping
from typing import Final, Literal, cast

NumpyTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes used by numpy.datetime64 and numpy.timedelta64."""

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

MAX_NUMPY_TIME_SCALE_FACTOR: Final = 2**31 - 1
"""The largest `scale_factor` NumPy accepts for a datetime64 or timedelta64 dtype."""

_CONFIGURATION_KEYS: Final = frozenset({"unit", "scale_factor"})


def numpy_time_unit(value: str) -> NumpyTimeUnit:
    """Validate `value` as a NumPy time unit and return its canonical spelling.

    The spec lists `"us"` and `"μs"` as equivalent spellings of the microsecond
    unit; NumPy itself only ever reports `"us"`, so `"μs"` is returned as `"us"`.

    Raises ValueError if `value` is not one of `NUMPY_TIME_UNIT`.
    """
    if value not in NUMPY_TIME_UNIT:
        raise ValueError(f"Expected one of {NUMPY_TIME_UNIT}, got {value!r}")
    if value == "μs":
        return "us"
    return cast("NumpyTimeUnit", value)


def numpy_time_configuration(value: Mapping[str, object]) -> tuple[NumpyTimeUnit, int]:
    """Validate a `numpy.datetime64` / `numpy.timedelta64` configuration object.

    Returns the `(unit, scale_factor)` pair with the unit in its canonical spelling
    (see `numpy_time_unit`).

    Raises TypeError if `unit` is not a string or `scale_factor` is not an
    integer. Raises ValueError if the object has keys other than exactly `unit`
    and `scale_factor`, if `unit` is not a `NumpyTimeUnit`, if `scale_factor` is
    outside `[1, MAX_NUMPY_TIME_SCALE_FACTOR]`, or if the unit is `"generic"`
    with a `scale_factor` other than 1. NumPy's generic (unit-less) time type
    carries no scale, so any other scale factor would be silently dropped when
    the data type is materialized.
    """
    keys = frozenset(value)
    if keys != _CONFIGURATION_KEYS:
        raise ValueError(
            f"Expected exactly the keys {sorted(_CONFIGURATION_KEYS)}, got {sorted(keys)}"
        )
    raw_unit = value["unit"]
    if not isinstance(raw_unit, str):
        raise TypeError(f"Expected 'unit' to be a string, got {raw_unit!r}")
    unit = numpy_time_unit(raw_unit)
    scale_factor = value["scale_factor"]
    if isinstance(scale_factor, bool) or not isinstance(scale_factor, int):
        raise TypeError(f"Expected 'scale_factor' to be an integer, got {scale_factor!r}")
    if not 1 <= scale_factor <= MAX_NUMPY_TIME_SCALE_FACTOR:
        raise ValueError(
            f"Expected 'scale_factor' in [1, {MAX_NUMPY_TIME_SCALE_FACTOR}], got {scale_factor}"
        )
    if unit == "generic" and scale_factor != 1:
        raise ValueError(
            f"The 'generic' unit does not take a scale factor, got scale_factor={scale_factor}"
        )
    return unit, scale_factor


__all__ = [
    "MAX_NUMPY_TIME_SCALE_FACTOR",
    "NUMPY_TIME_UNIT",
    "NumpyTimeUnit",
    "numpy_time_configuration",
    "numpy_time_unit",
]
