"""JSON checked against a TypedDict: the public entry point.

This package describes every Zarr document and configuration as a
TypedDict, such as `ZarrV3ArrayMetadataJSON` or `GzipCodecConfiguration`.
`check` turns any of them into a runtime check. It type-checks a JSON
value against the TypedDict and returns a value of the TypedDict (or
None) and every problem, each located in the value:

    import json

    from zarr_metadata import ZarrV3ArrayMetadataJSON
    from zarr_metadata.typed_json import check

    document, problems = check(json.loads(raw), ZarrV3ArrayMetadataJSON)
    for problem in problems:
        print(problem.loc, problem.kind, problem.message)

`check` reads a TypedDict as the typing spec defines it, regardless of
how its module writes annotations:

- A key is required when its `Required` or `NotRequired` qualifier says
  so. Otherwise the `total` setting of the class that declared the key
  decides. The runtime's `__required_keys__` cannot see a qualifier
  written as a string (and `from __future__ import annotations` makes
  every annotation a string). `typeddict_keys` reads the evaluated
  annotations, so it can see the qualifier.
- A key the TypedDict does not declare is handled according to `closed`
  or `extra_items`, or, when the class sets neither, according to its
  bases. In a closed TypedDict the key is reported as `unknown_key` and
  left out, and the value is still returned. With `extra_items=` the key
  is checked as that type. In an open TypedDict the key is kept.
- Each annotation is evaluated in the module of the class that wrote it,
  as the spec requires. `get_type_hints`, by contrast, evaluates
  inherited annotations in the subclass's module.
- `Required`, `NotRequired` and `ReadOnly` are removed wherever they are
  written. A `NewType` is checked as its supertype, and a type alias as
  the type it stands for. A TypedDict or alias that refers to itself is
  checked as deep as the value goes.
- A number type may carry bounds, written with annotated-types as
  pydantic reads them: `Gt`, `Ge`, `Lt`, `Le` and `Interval`, at any
  depth. For example, `tuple[Annotated[int, Ge(1)], ...]` bounds each
  element. A value outside the bounds is an `invalid_value` problem. Its
  message states what the type allows, and its `ctx` holds the bounds.
  `Annotated` may also carry a note (a string or a `Doc`). Any other
  metadata raises `TypeError`, because `check` would not enforce a
  constraint it does not support.
- In a union of TypedDicts where each requires a key as a `Literal` of
  values unique to that branch, the value of that key selects the
  branch, and the problems are that branch's. Otherwise the value is
  checked by the first branch that reports no problem (among TypedDicts,
  the one that declares the most of the value's keys). If every branch
  reports problems, the result is from the branch with the fewest
  problems.

The returned value is a value of the TypedDict: arrays as tuples, and
each object a new dict of the keys its type allows. Problems are values,
not exceptions: `ValidationProblem(loc, message, kind)`, where `kind` is
one of `missing_key`, `invalid_type`, `invalid_value`, `unknown_key` and
`invalid_json`. A caller that tolerates undeclared keys can therefore
distinguish them from wrong values. Each problem carries the data in its
message, as pydantic's errors and zod's issues do: `input` is the JSON
in the value at `loc`, and `ctx` is what was expected there (a type's
bounds, or the values of a `Literal`).

`zarr_metadata.json_schema.json_schema` writes the rules `check` applies
as a JSON Schema.

`check` supports only annotations that describe JSON shapes: `int`,
`float` for any number, `bool`, `str`, `None`, `JSONValue`, a `Literal`,
`tuple[T, ...]` and `tuple[T1, T2]`, a union, a TypedDict,
`Mapping[str, V]`, a `NewType` and a type alias. A TypedDict that
contains anything else raises `TypeError`, naming the member and the
TypedDict that contains it. The same happens for a TypedDict the spec
itself forbids, such as one with a key that is both `Required` and
`NotRequired`, and for a generic TypedDict, because no runtime records
how its type arguments bind in its bases' keys. A string annotation
resolves only in its module, so a TypedDict written inside a function
cannot refer to a class defined in that function. On Python 3.11,
`typing.TypedDict` does not record a class's bases, so a subclass
evaluates its inherited annotations in its own module.
`typing_extensions.TypedDict` records the bases on every version.
"""

from zarr_metadata._common import JSONValue
from zarr_metadata._json import ProblemKind, ValidationProblem
from zarr_metadata._typed_json import (
    Loc,
    check,
)

__all__ = [
    "JSONValue",
    "Loc",
    "ProblemKind",
    "ValidationProblem",
    "check",
]
