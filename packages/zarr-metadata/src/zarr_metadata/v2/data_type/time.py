"""The v2 time families, `datetime64` (`M`) and `timedelta64` (`m`): eight bytes, in a unit the typestr brackets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final, Literal, NotRequired, cast

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v2.data_type.scalar import (
    ZarrV2ByteOrder,  # noqa: TC001 - a TypedDict's annotations are evaluated at run time
)
from zarr_metadata.v3.data_type._numpy_time import (
    NumpyTimeScaleFactor,
    NumpyTimeTicks,
    numpy_time_fill_value_canonical,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.v3._definition import Nested


class ZarrV2TimeConfiguration(TypedDict, closed=True):
    """What a time typestr carries: `<M8[10ns]` is `{"byteorder": "<", "itemsize": 8, "unit": "ns", "scale_factor": 10}`."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    itemsize: ReadOnly[Annotated[int, Ge(0)]]
    unit: NotRequired[ReadOnly[str]]
    scale_factor: NotRequired[ReadOnly[NumpyTimeScaleFactor]]


UNITS: Final = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as")
"""The units NumPy gives a datetime64 or timedelta64 (https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units)."""


def _rules(configuration: ZarrV2TimeConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
    """Eight bytes, in an order; and a unit, which the v2 spec requires (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L147-L150)."""
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
    if configuration["unit"] not in UNITS:
        yield ValidationProblem(
            ("unit",),
            f"expected a NumPy time unit -- {', '.join(UNITS)} -- got {configuration['unit']!r}",
            "invalid_value",
        )


def _canonical(configuration: ZarrV2TimeConfiguration) -> ZarrV2TimeConfiguration:
    """`μs` is written `us`, as NumPy writes it."""
    if configuration.get("unit") == "μs":
        return cast("ZarrV2TimeConfiguration", {**configuration, "unit": "us"})
    return configuration


ZarrV2TimeFillValue = NumpyTimeTicks | Literal["NaT"] | None
"""A v2 time fill value: a count of ticks, `"NaT"`, or null."""


def _fill_value_canonical(
    configuration: ZarrV2TimeConfiguration, nested: Nested, value: ZarrV2TimeFillValue
) -> ZarrV2TimeFillValue:
    """`NaT` for not-a-time however it is written; any other value as written."""
    if value is None:
        return None
    return cast(
        "ZarrV2TimeFillValue", numpy_time_fill_value_canonical(configuration, nested, value)
    )


DATETIME64_V2: Final = ZarrV2DataTypeDefinition(
    name="datetime64",
    configuration=ZarrV2TimeConfiguration,
    rules=_rules,
    canonical=_canonical,
    fill_value=ZarrV2TimeFillValue,
    fill_value_canonical=_fill_value_canonical,
)
"""`<M8[unit]`: a moment, as ticks of the unit since the epoch."""

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
    "UNITS",
    "ZarrV2TimeConfiguration",
    "ZarrV2TimeFillValue",
]
