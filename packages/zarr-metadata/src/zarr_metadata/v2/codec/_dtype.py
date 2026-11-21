"""A rule for the dtype parameters of a v2 codec."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import canonicalize, resolve
from zarr_metadata.v2._context import ZarrV2Context
from zarr_metadata.v2._definition import ZarrV2DataTypeDefinition, parse_typestr
from zarr_metadata.v2.data_type import DATA_TYPES_V2
from zarr_metadata.v2.data_type._rules import time_rules
from zarr_metadata.v2.data_type.time import DATETIME64_DATA_TYPE_V2, TIMEDELTA64_DATA_TYPE_V2

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from zarr_metadata.definition._definition import Nested

_DTYPE_CONTEXT: Final = ZarrV2Context(
    data_types=[
        *(
            definition
            for definition in DATA_TYPES_V2
            if definition.name not in {DATETIME64_DATA_TYPE_V2.name, TIMEDELTA64_DATA_TYPE_V2.name}
        ),
        replace(DATETIME64_DATA_TYPE_V2, rules=time_rules("M", unit_required=False)),
        replace(TIMEDELTA64_DATA_TYPE_V2, rules=time_rules("m", unit_required=False)),
    ]
)
"""The scope a dtype parameter is read in: the data types that zarr-python 2.x writes, `DATA_TYPES_V2`, whatever scope the array is read in, because numcodecs reads the parameter with NumPy. A time type may leave out its unit here, as in `<M8`: numcodecs writes `np.dtype("M8").str` for a NumPy time type without a unit, and runs it. The v2 spec requires the unit only of an array's dtype."""


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
            _, problems = resolve(written, ZarrV2DataTypeDefinition, _DTYPE_CONTEXT, (key,))
            if len(problems) != 0:
                yield from problems
                continue
            parsed = parse_typestr(written)
            if self.float_only and (parsed is None or parsed[0] != "f"):
                yield ValidationProblem(
                    (key,), f"expected a float typestr, '<f8', got {written!r}", "invalid_value"
                )


def dtype_parameter(*keys: str, float_only: bool = False) -> _DtypeParameter:
    """Return a rule that checks that each of `keys`, when present and not null, is a typestr that an array's dtype could be, by the rules of its family in `DATA_TYPES_V2`. So `<f3` and `|f4` are refused. numcodecs writes `np.dtype(x).str` there, and a struct cannot be an element type. A time type may leave out its unit, as in `<M8`, because numcodecs writes and runs it.

    The parameter is checked against the core v2 data types, not against
    the scope the array is read in. So a scope that widens a family, such
    as one that reads `<f16`, does not widen what a codec's dtype
    parameter may be.

    With `float_only`, each value must name a float type, as `quantize` requires.
    """
    return _DtypeParameter(keys, float_only)


def canonical_typestr(written: str) -> str:
    """Return `written`, a dtype parameter that `dtype_parameter` accepts, in the canonical spelling of its data type, which is how an array's dtype compares. This is how NumPy writes it: `<i1` is `|i1`, and `<M8[μs]` is `<M8[us]`."""
    canonical, _ = canonicalize(written, ZarrV2DataTypeDefinition, _DTYPE_CONTEXT)
    return canonical if isinstance(canonical, str) else written


__all__ = ["canonical_typestr", "dtype_parameter"]
