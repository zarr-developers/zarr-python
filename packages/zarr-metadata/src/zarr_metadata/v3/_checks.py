"""The member checks every entity needs, and the walk that applies them.

One check is a function of a value and its location that returns the
problems it found -- none, for a value of the right type. The scalars,
a closed set of names, a homogeneous sequence, and a nested metadata
field cover what a configuration member can be; `coerce_members` walks
a configuration with a table of them, distinguishing an unknown key, an
optional member that failed, and a required one that did. `within` and
`named_configuration` are how an entity's problems and envelope are read
from the document that holds it.
"""

from __future__ import annotations

# Runtime imports, not `TYPE_CHECKING` ones: the string type aliases below
# (`TypeCheck`, `MemberTypes`) are resolved by `get_type_hints` at class
# creation, and a name that exists only for the type checker is a NameError
# then -- for this package and for any tool introspecting an entity.
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    TypeAlias,
    cast,
)

from typing_extensions import TypeIs

from zarr_metadata.model._validation import (
    ValidationProblem,
    is_json,
)

if TYPE_CHECKING:
    from zarr_metadata.model._validation import ProblemKind


Loc: TypeAlias = "tuple[str | int, ...]"


TypeCheck: TypeAlias = "Callable[[object, Loc], tuple[ValidationProblem, ...]]"
"""Whether one value has the type a member declares, and where if not."""


MemberTypes: TypeAlias = "Mapping[str, tuple[bool, TypeCheck]]"
"""Per configuration member: whether it is required, and its type check."""


def problem(
    loc: Loc, message: str, kind: ProblemKind = "invalid_type"
) -> tuple[ValidationProblem, ...]:
    """One problem, as the one-element tuple every check returns.

    A tuple so that a check can return it directly and a rule can
    `found.extend(problem(...))` and raise `MetadataValidationError(found)`
    once. The default `kind` names a type mismatch; a value rule passes
    `"invalid_value"`.
    """
    return (ValidationProblem(loc, message, kind),)


def is_integer(value: object) -> TypeIs[int]:
    """A JSON integer: an `int`, and not a `bool`.

    `True` is an `int` in Python and `true` is not a number in JSON, so
    the two have to be told apart everywhere a number is expected.
    """
    return not isinstance(value, bool) and isinstance(value, int)


def is_int(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An integer, and not a bool -- JSON `true` is not the integer 1."""
    if not is_integer(value):
        return problem(loc, f"expected an integer, got {value!r}")
    return ()


def is_str(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, str):
        return problem(loc, f"expected a string, got {value!r}")
    return ()


def is_bool(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    if not isinstance(value, bool):
        return problem(loc, f"expected a boolean, got {value!r}")
    return ()


def is_number(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """A JSON number: an `int` or a `float`, and not a `bool`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return problem(loc, f"expected a number, got {value!r}")
    return ()


def is_json_value(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """Any JSON value at all -- the widest type a member can declare."""
    if not is_json(value):
        return problem(loc, f"expected a JSON value, got {value!r}")
    return ()


def one_of(allowed: tuple[str, ...]) -> TypeCheck:
    """A member whose type is a closed set of names."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if value not in allowed:
            return problem(loc, f"expected one of {allowed!r}, got {value!r}", "invalid_value")
        return ()

    return check


def sequence_of(element: TypeCheck) -> TypeCheck:
    """A member whose type is a sequence, checked element by element."""

    def check(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(value, (list, tuple)):
            return problem(loc, f"expected a sequence, got {value!r}")
        elements: tuple[object, ...] = tuple(cast("list[object] | tuple[object, ...]", value))
        return tuple(
            found for index, entry in enumerate(elements) for found in element(entry, (*loc, index))
        )

    return check


def object_of(value: TypeCheck) -> TypeCheck:
    """A member whose type is an object with any keys, checked value by value.

    The open counterpart of `mapping_of`: a `Mapping[str, V]` says nothing
    about which keys there are, only what each value must be.
    """

    def check(candidate: object, loc: Loc) -> tuple[ValidationProblem, ...]:
        if not isinstance(candidate, Mapping):
            return problem(loc, f"expected an object, got {candidate!r}")
        entries = cast("Mapping[str, object]", candidate)
        return tuple(found for key, entry in entries.items() for found in value(entry, (*loc, key)))

    return check


def _as_tuples(value: object) -> object:
    """Every JSON array in `value`, at any depth, as a tuple.

    The TypedDicts spell a JSON array as a tuple throughout, so a member
    taken straight from parsed JSON would otherwise hold a list where its
    own type says tuple -- and two documents differing only in that would
    compare unequal.
    """
    if isinstance(value, (list, tuple)):
        entries = cast("list[object] | tuple[object, ...]", value)
        return tuple(_as_tuples(entry) for entry in entries)
    if isinstance(value, Mapping):
        entries = cast("Mapping[str, object]", value)
        return {key: _as_tuples(entry) for key, entry in entries.items()}
    return value


def coerce_members(
    configuration: Mapping[str, object], types: MemberTypes
) -> tuple[dict[str, object], tuple[ValidationProblem, ...]]:
    """The members `types` declares, taken from `configuration`.

    Returns what was read and every problem found. A key the entity does
    not declare says the value carries something extra, not that it is
    wrong, so the member it sits beside is still read; a member of the
    wrong type, or a required one missing, is reported and left out --
    and an entity is never built around the hole, because its rules are
    written over a whole configuration.
    """
    problems: list[ValidationProblem] = []
    members: dict[str, object] = {}
    for key in configuration:
        if key not in types:
            problems.extend(
                problem(("configuration", key), f"unexpected key {key!r}", "unknown_key")
            )
    for key, (required, check) in types.items():
        if key not in configuration:
            if required:
                problems.extend(
                    problem(("configuration", key), f"missing required key {key!r}", "missing_key")
                )
            continue
        # Normalized before the check, so a check only ever sees the tuples
        # the TypedDicts declare -- never the lists raw JSON arrives as.
        value = _as_tuples(configuration[key])
        found = check(value, ("configuration", key))
        problems.extend(found)
        if all(entry.kind == "unknown_key" for entry in found):
            members[key] = value
    return members, tuple(problems)


def is_metadata_field(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """A nested metadata field: a bare name or a named-configuration object.

    Only the envelope's shape. Which entity the name denotes, and whether
    its configuration is well formed, is settled when the containing
    entity reads it in scope.
    """
    if not isinstance(value, (str, Mapping)):
        return problem(loc, f"expected a metadata field, got {value!r}")
    return ()


def within(prefix: Loc, problems: Sequence[ValidationProblem]) -> tuple[ValidationProblem, ...]:
    """One entity's problems, located in the document that holds it.

    An entity reports relative to its own `configuration`, so that is what
    goes between the field and the member. A problem with an empty
    location is about the entity itself -- a malformed `r<N>` name, a
    codec that cannot encode what reaches it -- and lands on the field.
    """
    return tuple(
        ValidationProblem(
            (*prefix, *(("configuration", *found.loc) if len(found.loc) != 0 else ())),
            found.message,
            found.kind,
        )
        for found in problems
    )


def named_configuration(
    value: object,
) -> tuple[str | None, Mapping[str, object] | None, bool]:
    """Split metadata into `(name, configuration, must_understand)`.

    The shared shape every entity arrives in: a bare name, or an object
    carrying one. A `None` name means the value is not a metadata field at
    all; a `None` configuration means the bare spelling was used.
    """
    if isinstance(value, str):
        return value, None, True
    if not isinstance(value, Mapping):
        return None, None, True
    entry = cast("Mapping[str, object]", value)
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, True
    configuration = entry.get("configuration")
    must_understand = entry.get("must_understand", True)
    return (
        name,
        cast("Mapping[str, object]", configuration) if isinstance(configuration, Mapping) else None,
        must_understand if isinstance(must_understand, bool) else True,
    )


__all__ = [
    "Loc",
    "MemberTypes",
    "TypeCheck",
    "coerce_members",
    "is_bool",
    "is_int",
    "is_integer",
    "is_json_value",
    "is_metadata_field",
    "is_number",
    "is_str",
    "named_configuration",
    "one_of",
    "problem",
    "sequence_of",
    "within",
]
