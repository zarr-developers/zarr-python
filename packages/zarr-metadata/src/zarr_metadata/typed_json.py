"""JSON checked against a TypedDict: the public door.

Every Zarr document and configuration this package describes is a
TypedDict -- `ZarrV3ArrayMetadataJSON`, `GzipCodecConfiguration` -- and
`check` makes any of them executable. It type-checks a JSON value
against the TypedDict and gives back a value of it, or None, and every
problem, each located in the value:

    import json

    from zarr_metadata import ZarrV3ArrayMetadataJSON
    from zarr_metadata.typed_json import check

    document, problems = check(json.loads(raw), ZarrV3ArrayMetadataJSON)
    for problem in problems:
        print(problem.loc, problem.kind, problem.message)

A TypedDict reads as the typing spec defines it, however its module
writes annotations:

- A key is required when its `Required` or `NotRequired` qualifier says
  so, and otherwise when the `total` of the class that declared it does.
  The runtime's `__required_keys__` cannot see a qualifier written as a
  string, as `from __future__ import annotations` writes every one;
  `typeddict_keys` reads the annotations evaluated, and can.
- A key the TypedDict does not declare is what `closed`, `extra_items`
  or, when the class says neither, its bases make it: in a closed
  TypedDict it is reported, as `unknown_key`, and left out, and the value
  still comes back; with `extra_items=` it is checked as that type; in
  an open one it is kept.
- Each annotation is evaluated in the module of the class that wrote
  it, as the spec has it, where `get_type_hints` would read what a
  subclass inherited in the subclass's module.
- `ReadOnly` and `Annotated` are peeled wherever they are written; a
  `NewType` reads as the type it names, a type alias as the type it
  stands for, and a TypedDict or alias that holds itself as deep as the
  value goes.
- A union of TypedDicts that each require a key as a `Literal` of values
  of their own is read by the branch that key names, and its problems
  are that branch's. Otherwise a value is read by the first branch it
  fits with no problem -- among TypedDicts, the one declaring the most of
  its keys -- and failing that, reported by the branch with the fewest
  problems.

What comes back is a value of the TypedDict: arrays as tuples, and each
object a new dict of the keys its type admits. Problems are values, not
exceptions: `ValidationProblem(loc, message, kind)`, with `kind` one of
`missing_key`, `invalid_type`, `invalid_value`, `unknown_key` and
`invalid_json`, so a caller that tolerates a key the TypedDict does not
declare can tell it from a wrong value.

`check` reads the shapes JSON takes and no others -- `int`, `float` for
any number, `bool`, `str`, `None`, `JSONValue`, a `Literal`,
`tuple[T, ...]` and `tuple[T1, T2]`, a union, a TypedDict,
`Mapping[str, V]`, a `NewType` and a type alias -- and a TypedDict
holding anything else is a `TypeError` naming the member, down to the
TypedDict that holds it. So is one the spec itself refuses, such as a
key both `Required` and `NotRequired`, and a generic TypedDict, since no
runtime records what its type arguments bind in its bases' keys. A
string annotation resolves only in its module, so a TypedDict written
inside a function cannot name a class of that function. `typing.TypedDict`
on Python 3.11 does not record a class's bases, so there a subclass
evaluates what it inherits in its own module; `typing_extensions.TypedDict`
records them on every version.
"""

from zarr_metadata._common import JSONValue
from zarr_metadata._typed_json import Loc, TypedDictKeys, check, typeddict_keys
from zarr_metadata.model._validation import ProblemKind, ValidationProblem

__all__ = [
    "JSONValue",
    "Loc",
    "ProblemKind",
    "TypedDictKeys",
    "ValidationProblem",
    "check",
    "typeddict_keys",
]
