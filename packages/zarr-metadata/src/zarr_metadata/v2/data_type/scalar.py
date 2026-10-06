"""The v2 scalar families: `bool`, `int`, `uint`, `float` and `complex`, filed by family and read for every typestr of the family."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Annotated, Final, Literal, cast

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition
from zarr_metadata.v3._definition import Nested

ZarrV2ByteOrder = Literal["<", ">", "|"]
"""The byte orders a typestr writes: little-endian, big-endian, and not relevant."""


class ZarrV2ScalarConfiguration(TypedDict, closed=True):
    """What a scalar typestr carries: its byte order and its size in bytes, `{"byteorder": "<", "itemsize": 4}` for `<f4`."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    itemsize: ReadOnly[Annotated[int, Ge(0)]]


Rules = Callable[[ZarrV2ScalarConfiguration, Nested], Iterator[ValidationProblem]]


def sized(sizes: tuple[int, ...] | None, orderless: frozenset[int] | None) -> Rules:
    """The rules of a family whose types are `sizes` bytes wide, any width when None.

    `orderless` is the sizes at which the type has no byte order, which
    NumPy writes as `|`: one byte for an integer, every size for bytes
    and void, none for a float; None is every size. At any other size
    the typestr says which end comes first, `<` or `>`.
    """

    def rules(
        configuration: ZarrV2ScalarConfiguration, nested: Nested
    ) -> Iterator[ValidationProblem]:
        size = configuration["itemsize"]
        if sizes is not None and size not in sizes:
            yield ValidationProblem(
                ("itemsize",),
                f"expected a size of {', '.join(map(str, sizes))} bytes, got {size}",
                "invalid_value",
            )
            return
        if configuration["byteorder"] == "|" and orderless is not None and size not in orderless:
            yield ValidationProblem(
                ("byteorder",),
                f"expected a byte order '<' or '>' for a type of {size} bytes, got '|'",
                "invalid_value",
            )

    return rules


def orderless_at(
    sizes: frozenset[int] | None,
) -> Callable[[ZarrV2ScalarConfiguration], ZarrV2ScalarConfiguration]:
    """The canonical spelling of a family whose types of `sizes` bytes have no byte order: `|`, as NumPy writes it; None is every size."""

    def canonical(configuration: ZarrV2ScalarConfiguration) -> ZarrV2ScalarConfiguration:
        if (sizes is None or configuration["itemsize"] in sizes) and configuration[
            "byteorder"
        ] != "|":
            return cast("ZarrV2ScalarConfiguration", {**configuration, "byteorder": "|"})
        return configuration

    return canonical


_ONE_BYTE: Final = frozenset({1})


def _in_range(
    signed: bool,
) -> Callable[[ZarrV2ScalarConfiguration, Nested, int | None], Iterator[ValidationProblem]]:
    """The rule that an integer fill value lies in the range of the type's size."""

    def rules(
        configuration: ZarrV2ScalarConfiguration, nested: Nested, value: int | None
    ) -> Iterator[ValidationProblem]:
        if value is None:
            return
        bits = 8 * configuration["itemsize"]
        low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)
        if not low <= value <= high:
            yield ValidationProblem(
                (), f"expected an integer in [{low}, {high}], got {value}", "invalid_value"
            )

    return rules


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
    """An integer written for a float is the float: `0` and `0.0` are one value."""
    return float(value) if isinstance(value, int) and not isinstance(value, bool) else value


def _complex_canonical(
    configuration: ZarrV2ScalarConfiguration, nested: Nested, value: ZarrV2ComplexFillValue
) -> ZarrV2ComplexFillValue:
    """Each component in the float's canonical spelling."""
    if value is None:
        return None
    real, imag = value
    return (
        cast("ZarrV2ComplexComponent", _float_canonical(configuration, nested, real)),
        cast("ZarrV2ComplexComponent", _float_canonical(configuration, nested, imag)),
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
    fill_value_rules=_in_range(True),
)
"""`i1` to `i8`: signed integers, the fill value in the type's range."""

UINT_V2: Final = ZarrV2DataTypeDefinition(
    name="uint",
    configuration=ZarrV2ScalarConfiguration,
    rules=sized((1, 2, 4, 8), _ONE_BYTE),
    canonical=orderless_at(_ONE_BYTE),
    fill_value=int | None,
    fill_value_rules=_in_range(False),
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
