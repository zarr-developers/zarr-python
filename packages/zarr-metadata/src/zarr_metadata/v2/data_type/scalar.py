"""The v2 scalar families: `bool`, `int`, `uint`, `float` and `complex`, filed by family and read for every typestr of the family."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from zarr_metadata.v3._definition import Nested

ZarrV2ByteOrder = Literal["<", ">", "|"]
"""The byte orders a typestr writes: little-endian, big-endian, and not relevant."""


class ZarrV2ScalarConfiguration(TypedDict, closed=True):
    """What a scalar typestr carries: its byte order and its size in bytes, `{"byteorder": "<", "itemsize": 4}` for `<f4`."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    itemsize: ReadOnly[Annotated[int, Ge(0)]]


@dataclass(frozen=True, slots=True)
class _Sized:
    """The rules of a family whose types are `sizes` bytes wide: a value rather than a closure, so a scope holding it pickles."""

    sizes: tuple[int, ...] | None
    orderless: frozenset[int] | None

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
                f"expected a byte order '<' or '>' for a type of {size} bytes, got '|'",
                "invalid_value",
            )


def sized(sizes: tuple[int, ...] | None, orderless: frozenset[int] | None) -> _Sized:
    """The rules of a family whose types are `sizes` bytes wide, any width when None.

    `orderless` is the sizes at which the type has no byte order, which
    NumPy writes as `|`: one byte for an integer, every size for bytes
    and void, none for a float; None is every size. At any other size
    the typestr says which end comes first, `<` or `>`.
    """
    return _Sized(sizes, orderless)


@dataclass(frozen=True, slots=True)
class _OrderlessAt:
    """The canonical spelling of a family whose types of `sizes` bytes have no byte order."""

    sizes: frozenset[int] | None

    def __call__(self, configuration: ZarrV2ScalarConfiguration) -> ZarrV2ScalarConfiguration:
        orderless = self.sizes is None or configuration["itemsize"] in self.sizes
        if orderless and configuration["byteorder"] != "|":
            return cast("ZarrV2ScalarConfiguration", {**configuration, "byteorder": "|"})
        return configuration


def orderless_at(sizes: frozenset[int] | None) -> _OrderlessAt:
    """The canonical spelling of a family whose types of `sizes` bytes have no byte order: `|`, as NumPy writes it; None is every size."""
    return _OrderlessAt(sizes)


_ONE_BYTE: Final = frozenset({1})


@dataclass(frozen=True, slots=True)
class _InRange:
    """The rule that an integer fill value lies in the range of the type's size."""

    signed: bool

    def __call__(
        self, configuration: ZarrV2ScalarConfiguration, nested: Nested, value: int | None
    ) -> Iterator[ValidationProblem]:
        if value is None:
            return
        bits = 8 * configuration["itemsize"]
        if self.signed:
            low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        else:
            low, high = 0, 2**bits - 1
        if not low <= value <= high:
            yield ValidationProblem(
                (), f"expected an integer in [{low}, {high}], got {value}", "invalid_value"
            )


ZarrV2FloatSpecial = Literal["NaN", "Infinity", "-Infinity"]
"""The non-finite values the v2 spec spells by name (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L178-L190)."""

ZarrV2FloatFillValue = float | int | ZarrV2FloatSpecial | None
"""A v2 float fill value: a number, a named non-finite value, or null."""

ZarrV2ComplexComponent = float | int | ZarrV2FloatSpecial
"""One component of a complex fill value: a float fill value that is not null."""

ZarrV2ComplexFillValue = tuple[ZarrV2ComplexComponent, ZarrV2ComplexComponent] | None
"""A v2 complex fill value: `[real, imag]`, each a float fill value, or null."""


def _float_canonical(
    configuration: ZarrV2ScalarConfiguration, nested: Nested, value: ZarrV2FloatFillValue
) -> ZarrV2FloatFillValue:
    """An integer written for a float is the float, `0` and `0.0` one value, and a number past the largest float of the dtype's width is the infinity of its sign, as the v3 float types read one and NumPy stores it."""
    return _held_in(value, configuration["itemsize"])


def _held_in(value: ZarrV2FloatFillValue, itemsize: int) -> ZarrV2FloatFillValue:
    """`value` as a float of `itemsize` bytes holds it: itself as a float, or the infinity of its sign past the largest such float; a width no float has keeps the value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    try:
        held = float(value)
        narrowed = held
        if itemsize in _FLOAT_CODES:
            # `struct` refuses a float16 past the largest, and packs a
            # float32 past the largest as an infinity.
            code = _FLOAT_CODES[itemsize]
            (narrowed,) = struct.unpack(code, struct.pack(code, held))
    except OverflowError:
        return "-Infinity" if value < 0 else "Infinity"
    if math.isinf(narrowed):
        return "-Infinity" if value < 0 else "Infinity"
    return held


_FLOAT_CODES: Final[Mapping[int, str]] = {2: "e", 4: "f", 8: "d"}
"""The `struct` code of the float each size in bytes is: what refuses a value past the largest."""


def _complex_canonical(
    configuration: ZarrV2ScalarConfiguration, nested: Nested, value: ZarrV2ComplexFillValue
) -> ZarrV2ComplexFillValue:
    """Each component in the float's canonical spelling."""
    if value is None:
        return None
    real, imag = value
    width = configuration["itemsize"] // 2
    return (
        cast("ZarrV2ComplexComponent", _held_in(real, width)),
        cast("ZarrV2ComplexComponent", _held_in(imag, width)),
    )


BOOL_V2: Final = ZarrV2DataTypeDefinition(
    name="bool",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized((1,), None),
    canonical=orderless_at(None),
    fill_value=bool | None,
)
"""`|b1`: one byte, true or false."""

INT_V2: Final = ZarrV2DataTypeDefinition(
    name="int",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized((1, 2, 4, 8), _ONE_BYTE),
    canonical=orderless_at(_ONE_BYTE),
    fill_value=int | None,
    fill_value_rules=_InRange(True),
)
"""`i1` to `i8`: signed integers, the fill value in the type's range."""

UINT_V2: Final = ZarrV2DataTypeDefinition(
    name="uint",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized((1, 2, 4, 8), _ONE_BYTE),
    canonical=orderless_at(_ONE_BYTE),
    fill_value=int | None,
    fill_value_rules=_InRange(False),
)
"""`u1` to `u8`: unsigned integers, the fill value in the type's range."""

FLOAT_V2: Final = ZarrV2DataTypeDefinition(
    name="float",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized((2, 4, 8), frozenset()),
    fill_value=ZarrV2FloatFillValue,
    fill_value_canonical=_float_canonical,
)
"""`f2`, `f4`, `f8`: IEEE 754 floats; the fill value a number or a named non-finite value."""

COMPLEX_V2: Final = ZarrV2DataTypeDefinition(
    name="complex",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized((8, 16), frozenset()),
    fill_value=ZarrV2ComplexFillValue,
    fill_value_canonical=_complex_canonical,
)
"""`c8`, `c16`: complex floats; the fill value `[real, imag]`."""

__all__ = [
    "BOOL_V2",
    "COMPLEX_V2",
    "FLOAT_V2",
    "INT_V2",
    "UINT_V2",
    "ZarrV2ByteOrder",
    "ZarrV2ComplexComponent",
    "ZarrV2ComplexFillValue",
    "ZarrV2FloatFillValue",
    "ZarrV2FloatSpecial",
    "ZarrV2ScalarConfiguration",
    "orderless_at",
    "sized",
]
