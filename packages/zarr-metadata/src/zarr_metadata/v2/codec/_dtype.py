"""A rule for the dtype parameters of a v2 codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from zarr_metadata._json import ValidationProblem
from zarr_metadata.v2._definition import parse_typestr, typestr_problem

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from zarr_metadata.v3._definition import Nested


@dataclass(frozen=True, slots=True)
class _DtypeParameter:
    """The rule `dtype_parameter` gives: a value rather than a closure, so a scope holding it pickles."""

    keys: tuple[str, ...]
    float_only: bool

    def __call__(
        self, configuration: Mapping[str, object], nested: Nested
    ) -> Iterator[ValidationProblem]:
        for key in self.keys:
            if key not in configuration:
                continue
            written = cast("str", configuration[key])
            bad = typestr_problem(written, (key,))
            if bad is not None:
                yield bad
                continue
            parsed = parse_typestr(written)
            if self.float_only and (parsed is None or parsed[0] != "f"):
                yield ValidationProblem(
                    (key,), f"expected a float typestr, '<f8', got {written!r}", "invalid_value"
                )


def dtype_parameter(*keys: str, float_only: bool = False) -> _DtypeParameter:
    """The rule that each of `keys`, when written, is a typestr: numcodecs writes `np.dtype(x).str` there, and a struct is no element type.

    With `float_only`, each must name a float, as `quantize` requires.
    """
    return _DtypeParameter(keys, float_only)


__all__ = ["dtype_parameter"]
