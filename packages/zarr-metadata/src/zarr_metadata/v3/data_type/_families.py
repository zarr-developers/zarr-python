"""What each family of data types accepts as a fill value.

A fill value is judged against the data type, which makes it a question
the data type answers: `DataTypeEntity.fill_value_problems`. The families
here exist because the answer is the same for every width in a family
apart from one number -- the range, the hex parser, the component type --
so each family is written once and parameterised by that number.

Written per family rather than kept in a table keyed by name, so that
what `int32` accepts is stated where `int32` is, with nothing elsewhere
to keep in step with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._entity import (
    Configuration,
    Configured,
    DataTypeEntity,
    StorageClass,
    is_integer,
    problem,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from zarr_metadata.v3._entity import Loc

FLOAT_SPECIALS: Final = ("NaN", "Infinity", "-Infinity")
"""The three non-finite floats the spec spells as strings."""


def as_sequence(value: object) -> tuple[object, ...] | None:
    """`value` as a tuple if it is a JSON array, else None.

    A string is a sequence in Python and never a JSON array, so it is
    excluded.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    # `isinstance` narrows to `Sequence[Unknown]`; its elements are objects.
    return tuple(cast("Sequence[object]", value))


def byte_values(value: object, expected: int | None, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An array of `expected` integers in [0, 255], or any length if None."""
    items = as_sequence(value)
    if items is None:
        return problem(loc, f"expected an array of byte values, got {value!r}", "invalid_value")
    if expected is not None and len(items) != expected:
        return problem(loc, f"expected {expected} byte values, got {len(items)}", "invalid_value")
    return tuple(
        found
        for index, item in enumerate(items)
        if not (is_integer(item) and 0 <= item <= 255)
        for found in problem(
            (*loc, index), f"expected integers in [0, 255], got {item!r}", "invalid_value"
        )
    )


@dataclass(frozen=True)
class IntegerDataType(DataTypeEntity):
    """A fixed-width integer. The width is the whole difference."""

    bounds: ClassVar[tuple[int, int]]

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        low, high = type(self).bounds
        if not is_integer(value):
            return problem(loc, f"expected an integer, got {value!r}", "invalid_value")
        if not low <= value <= high:
            return problem(
                loc, f"expected an integer in [{low}, {high}], got {value!r}", "invalid_value"
            )
        return ()


@dataclass(frozen=True)
class FloatDataType(DataTypeEntity):
    """A binary float. A fill value may be a number, a named non-finite, or hex."""

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    hex_parser: ClassVar[Callable[[str], object]]

    largest: ClassVar[float | None]
    """The largest finite magnitude this width holds, or None for float64.

    None because a Python float *is* a float64, so no literal that reaches
    here can exceed it.
    """

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        if is_integer(value) or isinstance(value, float):
            largest = type(self).largest
            if largest is not None and abs(value) > largest:
                return problem(
                    loc,
                    f"expected a {type(self).identifier} value, got {value!r}",
                    "invalid_value",
                )
            return ()
        if not isinstance(value, str):
            return problem(loc, f"expected a number or string, got {value!r}", "invalid_value")
        if value in FLOAT_SPECIALS:
            return ()
        try:
            type(self).hex_parser(value)
        except ValueError:
            return problem(
                loc,
                f"expected a number, one of 'NaN'/'Infinity'/'-Infinity', or a "
                f"{type(self).identifier} hex string, got {value!r}",
                "invalid_value",
            )
        return ()


@dataclass(frozen=True)
class ComplexDataType(DataTypeEntity):
    """A complex number: a `[real, imag]` pair of the component float type."""

    scalar_storage: ClassVar[StorageClass] = "multi_byte"
    component: ClassVar[type[FloatDataType]]

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        pair = as_sequence(value)
        if pair is None or len(pair) != 2:
            return problem(loc, f"expected a [real, imag] pair, got {value!r}", "invalid_value")
        component = type(self).component()
        return tuple(
            ValidationProblem(found.loc, f"invalid component: {found.message}", found.kind)
            for index, part in enumerate(pair)
            for found in component.fill_value_problems(part, (*loc, index))
        )


NumpyTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""Time unit codes shared by `numpy.datetime64` and `numpy.timedelta64`."""

NUMPY_TIME_UNIT: Final = (
    "Y",
    "M",
    "W",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "μs",
    "ns",
    "ps",
    "fs",
    "as",
    "generic",
)
"""Tuple of the permitted `unit` values, in numpy's order from coarse to fine."""

NUMPY_TIME_MAX_SCALE_FACTOR: Final = 2**31 - 1
"""The largest `scale_factor` numpy stores: the field is a signed int32."""


@dataclass(frozen=True)
class NumpyTimeOptions(Configuration):
    """What a numpy time type is configured with: a unit, and how many of it one tick is."""

    unit: NumpyTimeUnit
    scale_factor: int

    def problems(self) -> Iterator[ValidationProblem]:
        if not 1 <= self.scale_factor <= NUMPY_TIME_MAX_SCALE_FACTOR:
            yield ValidationProblem(
                ("scale_factor",),
                f"expected an integer in [1, {NUMPY_TIME_MAX_SCALE_FACTOR}], "
                f"got {self.scale_factor}",
                "invalid_value",
            )


@dataclass(frozen=True)
class NumpyTimeDataType(DataTypeEntity, Configured):
    """A numpy time scalar: a signed 64-bit count of units, or `NaT`.

    The two time types share their configuration -- a unit and a scale
    factor -- and the rule on it, so both live here with the family and
    neither sibling imports them from the other.
    """

    configuration: NumpyTimeOptions

    scalar_storage: ClassVar[StorageClass] = "multi_byte"

    @property
    def unit(self) -> NumpyTimeUnit:
        return self.configuration.unit

    @property
    def scale_factor(self) -> int:
        return self.configuration.scale_factor

    def fill_value_problems(self, value: object, loc: Loc = ()) -> tuple[ValidationProblem, ...]:
        if value == "NaT":
            return ()
        if not is_integer(value):
            return problem(
                loc, f"expected a signed 64-bit integer or 'NaT', got {value!r}", "invalid_value"
            )
        if not -(2**63) <= value <= 2**63 - 1:
            return problem(loc, f"expected a signed 64-bit integer, got {value!r}", "invalid_value")
        return ()


__all__ = [
    "FLOAT_SPECIALS",
    "NUMPY_TIME_MAX_SCALE_FACTOR",
    "NUMPY_TIME_UNIT",
    "ComplexDataType",
    "FloatDataType",
    "IntegerDataType",
    "NumpyTimeDataType",
    "NumpyTimeOptions",
    "NumpyTimeUnit",
    "as_sequence",
    "byte_values",
]
