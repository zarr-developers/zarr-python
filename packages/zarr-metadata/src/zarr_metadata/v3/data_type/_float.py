"""The fill value rules floating-point numbers share, and complex numbers built of them.

A float's fill value is a JSON number, which a reader rounds to the
nearest value the type represents; one of `"NaN"`, `"Infinity"` and
`"-Infinity"`; or `"0x"` and the hex digits of the value's bytes read as
an unsigned integer, as many as the type has bytes
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79).
A complex fill value is a pair of such components, real then imaginary
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L88-L91).
"""

import functools
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, get_args

from zarr_metadata._json import ValidationProblem
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
    `ValueError` for one that is not. A partial application of a
    module-level function, so a definition holding it pickles.
    """
    return functools.partial(_float_fill_value, name, hex_form)


def _float_fill_value(
    name: str,
    hex_form: Callable[[str], str],
    configuration: EmptyConfiguration,
    nested: Nested,
    value: float | str,
) -> Iterator[ValidationProblem]:
    if not isinstance(value, str) or value in get_args(FloatSpecialFillValue):
        return
    try:
        hex_form(value)
    except ValueError:
        yield ValidationProblem(
            (),
            f"expected a number, one of {get_args(FloatSpecialFillValue)!r}, or a {name} "
            f"hex string, got {value!r}",
            "invalid_value",
        )


def complex_fill_value_rules(
    component: Callable[[EmptyConfiguration, Nested, Any], Iterable[ValidationProblem]],
) -> Callable[
    [EmptyConfiguration, Nested, tuple[float | str, float | str]], Iterator[ValidationProblem]
]:
    """The fill value rules of a complex type, whose components `component` judges, each at its index.

    `component` is the fill value rules of the component's own float type.
    """
    return functools.partial(_complex_fill_value, component)


def _complex_fill_value(
    component: Callable[[EmptyConfiguration, Nested, Any], Iterable[ValidationProblem]],
    configuration: EmptyConfiguration,
    nested: Nested,
    value: tuple[float | str, float | str],
) -> Iterator[ValidationProblem]:
    for index, part in enumerate(value):
        for found in component(configuration, nested, part):
            yield ValidationProblem((index, *found.loc), found.message, found.kind)


__all__ = ["FloatSpecialFillValue", "complex_fill_value_rules", "float_fill_value_rules"]
