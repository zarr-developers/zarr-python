"""The v2 time families, `datetime64` (`M`) and `timedelta64` (`m`): eight bytes, in a unit that the typestr gives in brackets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final, Literal, NotRequired

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata._numpy import (
    NUMPY_TIME_UNITS,
    NumpyTimeScaleFactor,
    NumpyTimeTicks,
    numpy_time_fill_value_canonical,
)
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type.scalar import (
    ZarrV2ByteOrder,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.definition._definition import Nested


class ZarrV2TimeConfiguration(TypedDict, closed=True):
    """The configuration that a time typestr carries. For example, `<M8[10ns]` is `{"byteorder": "<", "itemsize": 8, "unit": "ns", "scale_factor": 10}`."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    itemsize: ReadOnly[Annotated[int, Ge(0)]]
    unit: NotRequired[ReadOnly[str]]
    scale_factor: NotRequired[ReadOnly[NumpyTimeScaleFactor]]


def _rules(configuration: ZarrV2TimeConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
    """Check for eight bytes with a byte order, and for a unit, which the v2 spec requires (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L147-L150)."""
    size = configuration["itemsize"]
    if size != 8:
        yield ValidationProblem(
            ("itemsize",), f"expected a size of 8 bytes, got {size}", "invalid_value"
        )
        return
    if configuration["byteorder"] == "|":
        yield ValidationProblem(
            ("byteorder",),
            "expected a byte order '<' or '>' for a type of 8 bytes, got '|'",
            "invalid_value",
        )
    if "unit" not in configuration:
        yield ValidationProblem(
            ("unit",),
            "expected a time unit in brackets, '<M8[ns]': the v2 spec requires one",
            "missing_key",
        )
        return
    if configuration["unit"] not in NUMPY_TIME_UNITS:
        yield ValidationProblem(
            ("unit",),
            f"expected a NumPy time unit -- {', '.join(NUMPY_TIME_UNITS)} -- got {configuration['unit']!r}",
            "invalid_value",
        )


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


DATETIME64_V2: Final = ZarrV2DataTypeDefinition(
    name="datetime64",
    configuration=ZarrV2TimeConfiguration,
    rules=_rules,
    canonical=_canonical,
    fill_value=ZarrV2TimeFillValue,
    fill_value_canonical=_fill_value_canonical,
)
"""`<M8[unit]`: a point in time, as ticks of the unit since the epoch."""

TIMEDELTA64_V2: Final = ZarrV2DataTypeDefinition(
    name="timedelta64",
    configuration=ZarrV2TimeConfiguration,
    rules=_rules,
    canonical=_canonical,
    fill_value=ZarrV2TimeFillValue,
    fill_value_canonical=_fill_value_canonical,
)
"""`<m8[unit]`: a duration, as ticks of the unit."""

__all__ = [
    "DATETIME64_V2",
    "TIMEDELTA64_V2",
    "ZarrV2TimeConfiguration",
    "ZarrV2TimeFillValue",
]
