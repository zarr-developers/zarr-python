"""Rules that several v2 dtype families share: the sizes and byte orders a family allows, the unit of a time type, and a base64 fill value of the item's size."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from zarr_metadata._json import ValidationProblem, shown
from zarr_metadata._numpy import NUMPY_TIME_UNITS, base64_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.definition._definition import Nested
    from zarr_metadata.v2.data_type.fixed_width import ZarrV2Base64FillValue
    from zarr_metadata.v2.data_type.scalar import ZarrV2ScalarConfiguration
    from zarr_metadata.v2.data_type.time import ZarrV2TimeConfiguration


@dataclass(frozen=True, slots=True)
class _Sized:
    """The rules of a family whose types are `sizes` bytes wide. This is a value rather than a closure, so a scope that holds it can be pickled."""

    sizes: tuple[int, ...] | None
    orderless: frozenset[int] | None
    unit: int

    def __call__(
        self, configuration: ZarrV2ScalarConfiguration, nested: Nested
    ) -> Iterator[ValidationProblem]:
        size = configuration["itemsize"]
        if self.sizes is not None and size not in self.sizes:
            yield ValidationProblem(
                ("itemsize",),
                f"expected a size of {', '.join(map(str, self.sizes))} bytes, got {size}",
                "invalid_value",
            )
            return
        if (
            configuration["byteorder"] == "|"
            and self.orderless is not None
            and size not in self.orderless
        ):
            yield ValidationProblem(
                ("byteorder",),
                f"expected a byte order '<' or '>' for a type of {size * self.unit} bytes, got '|'",
                "invalid_value",
            )


def sized(
    sizes: tuple[int, ...] | None, orderless: frozenset[int] | None, *, unit: int = 1
) -> _Sized:
    """Return the rules of a family whose types are `sizes` bytes wide. If `sizes` is None, any width is allowed.

    `orderless` is the set of sizes at which the type has no byte order,
    which NumPy writes as `|`. This is one byte for an integer, every size
    for bytes and void, and no size for a float. None means every size.
    At any other size, the typestr must state which end comes first, `<`
    or `>`. `unit` is the number of bytes that one unit of the size takes:
    one, or four for `U`, whose size counts code points.
    """
    return _Sized(sizes, orderless, unit)


@dataclass(frozen=True, slots=True)
class _OrderlessAt:
    """The canonical spelling for a family whose types have no byte order at `sizes` bytes."""

    sizes: frozenset[int] | None

    def __call__(self, configuration: ZarrV2ScalarConfiguration) -> ZarrV2ScalarConfiguration:
        orderless = self.sizes is None or configuration["itemsize"] in self.sizes
        if orderless and configuration["byteorder"] != "|":
            return {**configuration, "byteorder": "|"}
        return configuration


def orderless_at(sizes: frozenset[int] | None) -> _OrderlessAt:
    """Return the canonical spelling for a family whose types have no byte order at `sizes` bytes. At those sizes the byte order is `|`, as NumPy writes it. None means every size."""
    return _OrderlessAt(sizes)


@dataclass(frozen=True, slots=True)
class _TimeRules:
    """The rules that `time_rules` returns. This is a value rather than a closure, so a scope that holds it can be pickled."""

    code: Literal["M", "m"]
    unit_required: bool

    def __call__(
        self, configuration: ZarrV2TimeConfiguration, nested: Nested
    ) -> Iterator[ValidationProblem]:
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
            if self.unit_required:
                yield ValidationProblem(
                    ("unit",),
                    f"expected a time unit in brackets, '<{self.code}8[ns]': the v2 spec "
                    "requires one",
                    "missing_key",
                )
            return
        if configuration["unit"] not in NUMPY_TIME_UNITS:
            yield ValidationProblem(
                ("unit",),
                f"expected a NumPy time unit -- {', '.join(NUMPY_TIME_UNITS)} -- got {configuration['unit']!r}",
                "invalid_value",
            )


def time_rules(code: Literal["M", "m"], *, unit_required: bool = True) -> _TimeRules:
    """Return the rules of the time family whose type code is `code`: eight bytes with a byte order, and a NumPy time unit in brackets.

    The v2 spec requires the unit of an array's dtype
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L147-L150).
    With `unit_required` False, the unit may be left out, as in `<M8`.
    numcodecs writes a codec's dtype parameter that way for a NumPy time
    type without a unit.
    """
    return _TimeRules(code, unit_required)


def sized_base64_rules(
    value: ZarrV2Base64FillValue, size: int | None, *, exact: bool
) -> Iterator[ValidationProblem]:
    """Check that a fill value that is not null is standard-alphabet base64 of exactly `size` bytes, or of at most `size` bytes when `exact` is False. Any size passes when `size` is None, which happens only when the item size is unknown."""
    if value is None:
        return
    try:
        base64_bytes(value)
    except ValueError:
        yield ValidationProblem(
            (), f"expected standard-alphabet base64, got {shown(value)}", "invalid_value"
        )
        return
    if size is None:
        return
    held = base64.b64decode(value)
    if (len(held) != size) if exact else (len(held) > size):
        bound = "" if exact else "at most "
        yield ValidationProblem(
            (),
            f"expected base64 of {bound}{size} bytes, the item's size, got {len(held)} bytes",
            "invalid_value",
        )


__all__ = ["orderless_at", "sized", "sized_base64_rules", "time_rules"]
