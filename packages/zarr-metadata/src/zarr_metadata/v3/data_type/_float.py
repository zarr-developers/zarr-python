"""The fill value rules shared by floating-point types and by the complex types built from them.

A float fill value is one of the following: a JSON number, which a
reader rounds to the nearest value the type represents; one of `"NaN"`,
`"Infinity"` and `"-Infinity"`; or `"0x"` followed by the hex digits of
the value's bytes read as an unsigned integer, with as many bytes as
the type has
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79).
A complex fill value is a pair of such components: the real part, then
the imaginary part
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L88-L91).
"""

import dataclasses
import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Final, Literal, get_args

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ValidationProblem, choices, shown
from zarr_metadata.definition._definition import (
    EmptyConfiguration,
    Nested,
)

FloatSpecialFillValue = Literal["NaN", "Infinity", "-Infinity"]
"""The named non-finite fill values every IEEE 754 floating-point type takes."""

FLOAT_SPECIAL_FILL_VALUES: Final[tuple[FloatSpecialFillValue, ...]] = get_args(
    FloatSpecialFillValue
)
"""The strings that the fill value rules accept for a float fill value in place of a number."""


def float_fill_value_rules(
    name: str, hex_form: Callable[[str], str]
) -> Callable[[EmptyConfiguration, Nested, float | str], Iterator[ValidationProblem]]:
    """Return the fill value rules of the floating-point type `name`. `hex_form` accepts its hex strings.

    A number may have any value, because a reader rounds it. A string
    must be one of the named values, or a hex string of the type's own
    width. The type-checked shape accepts any string here, so `hex_form`
    raises `ValueError` for a string that is not a valid hex string.
    """
    return _FloatFillValue(name, hex_form)


@dataclass(frozen=True, slots=True)
class _FloatFillValue:
    """The rules for a float fill value. This is a value rather than a closure, so a definition that holds it is equal to itself after a pickle or a deep copy."""

    name: str
    hex_form: Callable[[str], str]

    def __call__(
        self, configuration: EmptyConfiguration, nested: Nested, value: float | str
    ) -> Iterator[ValidationProblem]:
        if not isinstance(value, str) or value in FLOAT_SPECIAL_FILL_VALUES:
            return
        try:
            self.hex_form(value)
        except ValueError:
            yield ValidationProblem(
                (),
                f"expected a number, {choices(FLOAT_SPECIAL_FILL_VALUES)}, or a "
                f"{self.name} hex string, got {shown(value)}",
                "invalid_value",
            )


FloatWidth = Literal[16, 32, 64]
"""The widths, in bits, of the IEEE 754 binary formats the float types store."""

_FORMATS: Final[dict[FloatWidth, tuple[str, int]]] = {16: ("e", 10), 32: ("f", 23), 64: ("d", 52)}
"""The `struct` format for each width, and the number of bits that hold the fraction."""


def float_fill_value_canonical(
    width: FloatWidth,
) -> Callable[[EmptyConfiguration, Nested, float | str], float | str]:
    """Return the canonical spelling of a fill value of the floating-point type that is `width` bits wide.

    A fill value spells a value of the type, as `float_bits` reads it: a
    number, a named value, or the bits of the value. The canonical
    spelling is the named value for an infinity, or for the NaN that the
    spec names `"NaN"`. For any other NaN, it is the hex string of its
    bits, in lower case. For any other value, it is the shortest number
    that rounds to the value; if several numbers are equally short, it
    is the nearest of them. This matches how numpy and `repr` spell a
    value, for example `0.1` for the `float32` nearest `0.1`. `-0.0` is
    treated as a separate value with its own spelling.
    """
    return _FloatCanonical(width)


@dataclass(frozen=True, slots=True)
class _FloatCanonical:
    """The canonical spelling of a float fill value. This is a value rather than a closure, for the same reason as `_FloatFillValue`."""

    width: FloatWidth

    def __call__(
        self, configuration: EmptyConfiguration, nested: Nested, value: float | str
    ) -> float | str:
        return _spelled(float_bits(value, self.width), self.width)


def float_bits(value: float | str, width: FloatWidth) -> int:
    """Return the bits of the value that `value` spells in the type that is `width` bits wide.

    `value` is a float fill value that the rules allow. A number is read
    as a float64, as a JSON parser reads it. An integer with more digits
    than a float64 holds is rounded to a float64. The float64 is then
    rounded to the nearest value the type represents, with ties to even,
    and to an infinity if it is past the largest value. This matches how
    zarrs reads a number, `as_f64` then `as f32`
    (https://github.com/zarrs/zarrs/blob/8d68f8522b382d050b768f84bce64c2935de4523/zarrs_metadata/src/v3/array/fill_value.rs#L160),
    how tensorstore reads one, `static_cast<T>` of `get<double>()`
    (https://github.com/google/tensorstore/blob/692d2798c51a76d2eed0b4aee85cad5fd4be950a/tensorstore/driver/zarr3/metadata.cc#L165),
    and how numpy casts a float64. So an integer and a number with a
    fraction spell the same value if they read as the same float64. An
    integer past 2**53 whose float64 lies halfway between two values of
    the type (2**60 + 2**36 + 1, for float32) reads as the even value,
    2**60, as those readers store it. It does not read as the value
    nearest the integer itself.
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
        # `struct` raises for a float64 that rounds past the type's largest value.
        return (sign if held < 0 else 0) | infinity


def _spelled(bits: int, width: FloatWidth) -> float | str:
    """Return the canonical spelling of the value whose bits are `bits`, in the type that is `width` bits wide."""
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
        # `repr` already writes the shortest spelling of a float64. A zero
        # of either sign is also already its shortest spelling.
        return value
    return _shortest(value, bits, width)


def _shortest(value: float, bits: int, width: FloatWidth) -> float:
    """Return the shortest number that rounds to `value`. Among numbers with the fewest significant digits, return the nearest one.

    `bits` are the bits of `value` in the type that is `width` bits wide.
    For each number of digits, this function tries the nearest number
    first, and then the next number on the other side of `value`. At a
    power of two, the numbers that round to `value` reach twice as far
    above it as below it. So the nearest number can miss when the next
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
    # Seventeen significant digits are enough to distinguish every float64.
    return value


def complex_fill_value_rules(
    component: Callable[[EmptyConfiguration, Nested, float | str], Iterable[ValidationProblem]],
) -> Callable[
    [EmptyConfiguration, Nested, tuple[float | str, float | str]], Iterator[ValidationProblem]
]:
    """Return the fill value rules of a complex type. `component` checks each component at its index.

    `component` is the fill value rules of the component's own float type.
    """
    return _ComplexFillValue(component)


@dataclass(frozen=True, slots=True)
class _ComplexFillValue:
    """The rules for a complex fill value, which check each component at its index. This is a value rather than a closure, for the same reason as `_FloatFillValue`."""

    component: Callable[[EmptyConfiguration, Nested, float | str], Iterable[ValidationProblem]]

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
    component: Callable[[EmptyConfiguration, Nested, float | str], JSONValue],
) -> Callable[[EmptyConfiguration, Nested, tuple[float | str, float | str]], JSONValue]:
    """Return the canonical spelling of a complex fill value. Each component is spelled by `component`, the canonical spelling of its float type."""
    return _ComplexCanonical(component)


@dataclass(frozen=True, slots=True)
class _ComplexCanonical:
    """The canonical spelling of a complex fill value. This is a value rather than a closure, for the same reason as `_FloatFillValue`."""

    component: Callable[[EmptyConfiguration, Nested, float | str], JSONValue]

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
