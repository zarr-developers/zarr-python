"""The fill value rules floating-point numbers share, and complex numbers built of them.

A float's fill value is a JSON number, which a reader rounds to the
nearest value the type represents; one of `"NaN"`, `"Infinity"` and
`"-Infinity"`; or `"0x"` and the hex digits of the value's bytes read as
an unsigned integer, as many as the type has bytes
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79).
A complex fill value is a pair of such components, real then imaginary
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L88-L91).
"""

import dataclasses
import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any, Final, Literal, get_args

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem, choices, shown
from zarr_metadata.v3._definition import EmptyConfiguration, Nested

FloatSpecialFillValue = Literal["NaN", "Infinity", "-Infinity"]
"""The named non-finite fill values every IEEE 754 floating-point type takes."""


def float_fill_value_rules(
    name: str, hex_form: Callable[[str], str]
) -> Callable[[EmptyConfiguration, Nested, float | str], Iterator[ValidationProblem]]:
    """The fill value rules of the floating-point type `name`, whose hex strings `hex_form` accepts.

    A number takes any value, since a reader rounds it. A string is one of
    the named values, or a hex string of the type's own width: the
    type-checked shape takes any string there, and `hex_form` raises
    `ValueError` for one that is not.
    """
    return _FloatFillValue(name, hex_form)


@dataclass(frozen=True, slots=True)
class _FloatFillValue:
    """A float fill value: a value rather than a closure, so a definition holding it is equal to itself after a pickle or a deep copy."""

    name: str
    hex_form: Callable[[str], str]

    def __call__(
        self, configuration: EmptyConfiguration, nested: Nested, value: float | str
    ) -> Iterator[ValidationProblem]:
        if not isinstance(value, str) or value in get_args(FloatSpecialFillValue):
            return
        try:
            self.hex_form(value)
        except ValueError:
            yield ValidationProblem(
                (),
                f"expected a number, {choices(get_args(FloatSpecialFillValue))}, or a "
                f"{self.name} hex string, got {shown(value)}",
                "invalid_value",
            )


FloatWidth = Literal[16, 32, 64]
"""The widths, in bits, of the IEEE 754 binary formats the float types store."""

_FORMATS: Final[dict[FloatWidth, tuple[str, int]]] = {16: ("e", 10), 32: ("f", 23), 64: ("d", 52)}
"""Each width's `struct` format, and how many bits of it hold the fraction."""


def float_fill_value_canonical(
    width: FloatWidth,
) -> Callable[[EmptyConfiguration, Nested, float | str], float | str]:
    """The canonical spelling of a fill value of the floating-point type `width` bits wide.

    A fill value spells a value of the type, as `float_bits` reads one: a
    number, a named value, or the value's bits. Its canonical spelling is
    the named value for an infinity, or for the NaN the spec names
    `"NaN"`; the hex string of its bits, in lower case, for any other NaN;
    and, for any other value, the shortest number that rounds to it, the
    nearest of those, as numpy and `repr` spell one -- `0.1` for the
    `float32` nearest `0.1` -- `-0.0`, a value of its own, among them.
    """
    return _FloatCanonical(width)


@dataclass(frozen=True, slots=True)
class _FloatCanonical:
    """A float fill value's canonical spelling: a value rather than a closure, as `_FloatFillValue` is."""

    width: FloatWidth

    def __call__(
        self, configuration: EmptyConfiguration, nested: Nested, value: float | str
    ) -> float | str:
        return _spelled(float_bits(value, self.width), self.width)


def float_bits(value: float | str, width: FloatWidth) -> int:
    """The bits of the value `value`, a float fill value the rules allow, spells in the type `width` bits wide.

    A number is read as a float64, as a JSON parser reads one -- an integer
    of more digits than a float64 holds is rounded to one -- and rounded to
    the nearest value the type represents, ties to even, and to an infinity
    past the largest: as zarrs reads one, `as_f64` then `as f32`
    (https://github.com/zarrs/zarrs/blob/8d68f8522b382d050b768f84bce64c2935de4523/zarrs_metadata/src/v3/array/fill_value.rs#L160),
    as tensorstore does, `static_cast<T>` of `get<double>()`
    (https://github.com/google/tensorstore/blob/692d2798c51a76d2eed0b4aee85cad5fd4be950a/tensorstore/driver/zarr3/metadata.cc#L165),
    and as numpy casts a float64. So an integer and a number with a
    fraction that read as one float64 spell one value; and an integer past
    2**53 whose float64 sits halfway between two values of the type --
    2**60 + 2**36 + 1, for float32 -- is the even one, 2**60, as those
    readers store it, not the value nearest the integer itself.
    """
    code, fraction = _FORMATS[width]
    exponent = width - 1 - fraction
    infinity = ((1 << exponent) - 1) << fraction
    sign = 1 << (width - 1)
    if isinstance(value, str):
        named = {
            "NaN": infinity | (1 << (fraction - 1)),
            "Infinity": infinity,
            "-Infinity": sign | infinity,
        }
        return named[value] if value in named else int(value, 16)
    try:
        held = float(value)
    except OverflowError:
        # An integer past the largest float64 reads as an infinity.
        return (sign if value < 0 else 0) | infinity
    try:
        return int.from_bytes(struct.pack(f">{code}", held), "big")
    except OverflowError:
        # `struct` refuses a float64 that rounds past the type's largest value.
        return (sign if held < 0 else 0) | infinity


def _spelled(bits: int, width: FloatWidth) -> float | str:
    """The canonical spelling of the value whose bits, in the type `width` bits wide, are `bits`."""
    code, fraction = _FORMATS[width]
    exponent = width - 1 - fraction
    infinity = ((1 << exponent) - 1) << fraction
    sign = 1 << (width - 1)
    if bits & infinity == infinity:
        if bits & ((1 << fraction) - 1) == 0:
            return "-Infinity" if bits & sign else "Infinity"
        if bits == infinity | (1 << (fraction - 1)):
            return "NaN"
        return f"0x{bits:0{width // 4}x}"
    value: float = struct.unpack(f">{code}", bits.to_bytes(width // 8, "big"))[0]
    if width == 64 or value == 0:
        # A float64 is its own shortest spelling, as `repr` writes it, and
        # so is a zero of either sign.
        return value
    return _shortest(value, bits, width)


def _shortest(value: float, bits: int, width: FloatWidth) -> float:
    """The shortest number that rounds to `value`, whose bits in the type `width` bits wide are `bits`: of the fewest significant digits, the nearest to it.

    Of each number of digits the nearest is tried, and then the one past
    `value` from it: at a power of two the numbers that round to it reach
    twice as far above it as below, so the nearest may miss where the next
    one does not.
    """
    exact = Decimal(value)
    for digits in range(1, 18):
        with localcontext() as context:
            context.prec = digits
            context.rounding = ROUND_HALF_EVEN
            nearest = +exact
        step = Decimal((0, (1,), nearest.adjusted() - digits + 1))
        beyond = nearest + step if nearest < exact else nearest - step
        for candidate in (nearest, beyond):
            if float_bits(float(candidate), width) == bits:
                return float(candidate)
    # Seventeen significant digits tell every float64 from every other.
    return value


def complex_fill_value_rules(
    component: Callable[[EmptyConfiguration, Nested, Any], Iterable[ValidationProblem]],
) -> Callable[
    [EmptyConfiguration, Nested, tuple[float | str, float | str]], Iterator[ValidationProblem]
]:
    """The fill value rules of a complex type, whose components `component` judges, each at its index.

    `component` is the fill value rules of the component's own float type.
    """
    return _ComplexFillValue(component)


@dataclass(frozen=True, slots=True)
class _ComplexFillValue:
    """A complex fill value, each component judged at its index: a value rather than a closure, as `_FloatFillValue` is."""

    component: Callable[[EmptyConfiguration, Nested, Any], Iterable[ValidationProblem]]

    def __call__(
        self,
        configuration: EmptyConfiguration,
        nested: Nested,
        value: tuple[float | str, float | str],
    ) -> Iterator[ValidationProblem]:
        for index, part in enumerate(value):
            for found in self.component(configuration, nested, part):
                yield dataclasses.replace(found, loc=(index, *found.loc))


def complex_fill_value_canonical(
    component: Callable[[EmptyConfiguration, Nested, Any], JSONValue],
) -> Callable[[EmptyConfiguration, Nested, tuple[float | str, float | str]], JSONValue]:
    """The canonical spelling of a complex fill value: each component in the canonical spelling `component`, its float type's, gives it."""
    return _ComplexCanonical(component)


@dataclass(frozen=True, slots=True)
class _ComplexCanonical:
    """A complex fill value's canonical spelling: a value rather than a closure, as `_FloatFillValue` is."""

    component: Callable[[EmptyConfiguration, Nested, Any], JSONValue]

    def __call__(
        self,
        configuration: EmptyConfiguration,
        nested: Nested,
        value: tuple[float | str, float | str],
    ) -> JSONValue:
        return tuple(self.component(configuration, nested, part) for part in value)


__all__ = [
    "FloatSpecialFillValue",
    "FloatWidth",
    "complex_fill_value_canonical",
    "complex_fill_value_rules",
    "float_bits",
    "float_fill_value_canonical",
    "float_fill_value_rules",
]
