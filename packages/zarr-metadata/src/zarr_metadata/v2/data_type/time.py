"""The v2 time families, `datetime64` (`M`) and `timedelta64` (`m`): eight bytes, in a unit that the typestr gives in brackets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata._numpy import (
    numpy_time_fill_value_canonical,
)
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2._types.data_type import (
    ZarrV2TimeConfiguration,
    ZarrV2TimeFillValue,
)
from zarr_metadata.v2.data_type._rules import time_rules

if TYPE_CHECKING:
    from zarr_metadata.definition._definition import Nested


def _canonical(configuration: ZarrV2TimeConfiguration) -> ZarrV2TimeConfiguration:
    """Write `μs` as `us`, as NumPy writes it."""
    if configuration.get("unit") == "μs":
        return {**configuration, "unit": "us"}
    return configuration


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
