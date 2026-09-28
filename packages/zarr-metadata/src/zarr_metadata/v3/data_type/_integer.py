"""The fill value rules integers share: a JSON integer within a range.

The integer data types take one within their own range, and the byte
values of `r*` and `bytes` fill values are integers in `[0, 255]`
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L59-L61).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v3._definition import EmptyConfiguration, Nested


def integer_fill_value_rules(
    low: int, high: int
) -> Callable[[EmptyConfiguration, Nested, int], Iterator[ValidationProblem]]:
    """The fill value rules of an integer data type whose range is `[low, high]`."""
    return _InRange(low, high)


@dataclass(frozen=True, slots=True)
class _InRange:
    """An integer in `[low, high]`: a value rather than a closure, so a definition holding it is equal to itself after a pickle or a deep copy."""

    low: int
    high: int

    def __call__(
        self, configuration: EmptyConfiguration, nested: Nested, value: int
    ) -> Iterator[ValidationProblem]:
        if not self.low <= value <= self.high:
            yield ValidationProblem(
                (),
                f"expected an integer in [{self.low}, {self.high}], got {value}",
                "invalid_value",
            )


def byte_value_problems(values: tuple[int, ...]) -> Iterator[ValidationProblem]:
    """Each of `values` that is not a byte, an integer in `[0, 255]`, located at its index."""
    for index, value in enumerate(values):
        if not 0 <= value <= 255:
            yield ValidationProblem(
                (index,), f"expected an integer in [0, 255], got {value}", "invalid_value"
            )


__all__ = ["byte_value_problems", "integer_fill_value_rules"]
