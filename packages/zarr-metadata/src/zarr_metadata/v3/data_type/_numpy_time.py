"""Definitions shared by the two numpy time types: the number of units in one tick, a count of ticks, and how their values are stored.

The configurations of both types hold a `scale_factor` of the same type,
and the fill values of both types count ticks of the same width. Each of
these is defined once here, so that neither type module imports it from
the other. The same applies to how their values are stored.
"""

from typing import Annotated, Final

from annotated_types import Interval

from zarr_metadata.v3._definition import multi_byte

NUMPY_TIME_MAX_SCALE_FACTOR: Final = 2**31 - 1
"""The largest `scale_factor` that numpy stores, because the field is a signed int32."""

NumpyTimeScaleFactor = Annotated[int, Interval(ge=1, le=NUMPY_TIME_MAX_SCALE_FACTOR)]
"""The number of units in one tick: a positive int32."""

NumpyTimeTicks = Annotated[int, Interval(ge=-(2**63), le=2**63 - 1)]
"""A count of ticks, as a fill value writes it: a signed 64-bit integer.

https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L109-L112
"""

NOT_A_TIME_TICKS: Final = -(2**63)
"""The tick count that stores `NaT`. A fill value may write this count for `NaT`.

"`"fill_value": "NaT"` and `"fill_value": -9223372036854775808` should
be treated as equivalent representations of the same scalar value"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L114-L116).
The `numpy.timedelta64` spec says the same
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.timedelta64/README.md?plain=1#L117-L119).
"""


def numpy_time_fill_value_canonical(
    configuration: object, nested: object, value: int | str
) -> int | str:
    """Return the canonical spelling of a numpy time fill value: `"NaT"` for `NaT` however it is written, and any other count of ticks as written."""
    return "NaT" if value in ("NaT", NOT_A_TIME_TICKS) else value


numpy_time_storage: Final = multi_byte
"""The storage of the numpy time types: signed 64-bit integers, in the byte order that the codecs specify.

Each type "is compatible with any codec that supports arrays of signed
64-bit integers"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L120),
and "the endianness of numpy.datetime64 arrays is determined by the
configuration of the codecs defined in metadata"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L83-L85).
The `numpy.timedelta64` spec says the same.
"""


__all__ = [
    "NOT_A_TIME_TICKS",
    "NUMPY_TIME_MAX_SCALE_FACTOR",
    "NumpyTimeScaleFactor",
    "NumpyTimeTicks",
    "numpy_time_fill_value_canonical",
    "numpy_time_storage",
]
