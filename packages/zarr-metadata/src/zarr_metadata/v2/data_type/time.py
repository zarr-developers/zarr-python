"""The v2 time families, `datetime64` (`M`) and `timedelta64` (`m`): eight bytes, in a unit that the typestr gives in brackets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final, Literal, NotRequired

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._numpy import (
    NumpyTimeScaleFactor,
    NumpyTimeTicks,
    numpy_time_fill_value_canonical,
)
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type._rules import time_rules
from zarr_metadata.v2.data_type.scalar import (
    ZarrV2ByteOrder,
)

if TYPE_CHECKING:
    from zarr_metadata.definition._definition import Nested


class ZarrV2TimeConfiguration(TypedDict, closed=True):
    """The configuration that a time typestr carries. For example, `<M8[10ns]` is `{"byteorder": "<", "itemsize": 8, "unit": "ns", "scale_factor": 10}`."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    """The byte order that the typestr writes, `<` or `>`."""
    itemsize: ReadOnly[Annotated[int, Ge(0)]]
    """The size in bytes that the typestr writes, which must be 8."""
    unit: NotRequired[ReadOnly[str]]
    """The NumPy time unit in brackets, such as `ns`. The canonical form writes `μs` as `us`.

    An array's dtype must have one. Only a codec's dtype parameter may leave
    it out, as in `<M8`.
    """
    scale_factor: NotRequired[ReadOnly[NumpyTimeScaleFactor]]
    """The number of units in one tick, such as the `10` of `[10ns]`, or 1 when the brackets write no number."""


def _canonical(configuration: ZarrV2TimeConfiguration) -> ZarrV2TimeConfiguration:
    """Write `μs` as `us`, as NumPy writes it."""
    if configuration.get("unit") == "μs":
        return {**configuration, "unit": "us"}
    return configuration


ZarrV2TimeFillValue = NumpyTimeTicks | Literal["NaT"] | None
"""A v2 time fill value: a count of ticks, `"NaT"`, or null."""


def _fill_value_canonical(
    configuration: ZarrV2TimeConfiguration, nested: Nested, value: ZarrV2TimeFillValue
) -> ZarrV2TimeFillValue:
    """Return `NaT` for not-a-time however it is written, and any other value as written."""
    if value is None:
        return None
    return numpy_time_fill_value_canonical(configuration, nested, value)


DATETIME64_DATA_TYPE_V2: Final = ZarrV2DataTypeDefinition(
    name="datetime64",
    configuration=ZarrV2TimeConfiguration,
    rules=time_rules("M"),
    canonical=_canonical,
    fill_value=ZarrV2TimeFillValue,
    fill_value_canonical=_fill_value_canonical,
)
"""`<M8[unit]`: a point in time, as ticks of the unit since the epoch."""

TIMEDELTA64_DATA_TYPE_V2: Final = ZarrV2DataTypeDefinition(
    name="timedelta64",
    configuration=ZarrV2TimeConfiguration,
    rules=time_rules("m"),
    canonical=_canonical,
    fill_value=ZarrV2TimeFillValue,
    fill_value_canonical=_fill_value_canonical,
)
"""`<m8[unit]`: a duration, as ticks of the unit."""

__all__ = [
    "DATETIME64_DATA_TYPE_V2",
    "TIMEDELTA64_DATA_TYPE_V2",
    "ZarrV2TimeConfiguration",
    "ZarrV2TimeFillValue",
]
