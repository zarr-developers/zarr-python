"""A rule for the dtype parameters of a v2 codec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import resolve
from zarr_metadata.v2._context import ZarrV2Context
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition, parse_typestr
from zarr_metadata.v2.data_type import DATA_TYPES_V2

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from zarr_metadata.definition._definition import Nested

_DATA_TYPES: Final = ZarrV2Context(data_types=DATA_TYPES_V2)
"""The scope a dtype parameter is read in: the data types that zarr-python 2.x writes. numcodecs reads the parameter with NumPy, whatever scope the array is read in."""


@dataclass(frozen=True, slots=True)
class _DtypeParameter:
    """The rule that `dtype_parameter` returns. It is a value rather than a closure, so a scope that holds it can be pickled."""

    keys: tuple[str, ...]
    float_only: bool

    def __call__(
        self, configuration: Mapping[str, object], nested: Nested
    ) -> Iterator[ValidationProblem]:
        for key in self.keys:
            written = configuration.get(key)
            if not isinstance(written, str):
                # Omitted, or null for an `astype` that numcodecs takes from `dtype`.
                continue
            _, problems = resolve(written, ZarrV2DataTypeDefinition, _DATA_TYPES, (key,))
            if len(problems) != 0:
                yield from problems
                continue
            parsed = parse_typestr(written)
            if self.float_only and (parsed is None or parsed[0] != "f"):
                yield ValidationProblem(
                    (key,), f"expected a float typestr, '<f8', got {written!r}", "invalid_value"
                )


def dtype_parameter(*keys: str, float_only: bool = False) -> _DtypeParameter:
    """Return a rule that checks that each of `keys`, when present and not null, is a typestr that an array's dtype could be, by the rules of its family in `DATA_TYPES_V2`. So `<f3` and `|f4` are refused. numcodecs writes `np.dtype(x).str` there, and a struct cannot be an element type.

    With `float_only`, each value must name a float type, as `quantize` requires.
    """
    return _DtypeParameter(keys, float_only)


__all__ = ["dtype_parameter"]
