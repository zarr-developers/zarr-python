"""What every metadata entity can do for itself.

A codec, data type, chunk grid or chunk key encoding is three things at
once: a JSON shape, a set of constraints on the values in that shape, and
a canonical spelling. Keeping the three apart put one entity's knowledge
in four modules and needed a table per axis plus a drift test per table to
hold them together. Here they are one class per entity, and the class is
where methods bind:

- `coerce` is **type-space**: raw metadata in, the entity or the reasons
  it is not that entity out.
- `problems` is **value-space**: the entity is well-typed by construction,
  so this only asks whether its values are in range.
- `to_json` is **canonical**: the simplest spelling meaning the same, typed
  as the entity's own object TypedDict.

The TypedDicts stay: they model the JSON form, and the correspondence is
exact in both directions. A configuration TypedDict unpacked is the
dataclass constructor's signature, and the object TypedDict is what
`to_json` returns. `tests/v3/test_entities.py` asserts the first, so the
two cannot drift.

Everything that needs the document or the codec chain stays outside, in
`zarr_metadata.rules`, because an entity cannot answer it alone.

`coerce` takes a `Context`: the entities in scope for this reading. Most
entities ignore it -- a `gzip` codec is a `gzip` codec whatever else is
registered -- but the ones whose configuration contains other entities do
not. A `struct` data type holds field data types and a `sharding_indexed`
codec holds two codec pipelines, and neither can coerce its own
configuration without knowing what names are in scope inside it.

The shared plumbing lives here too: the member checks every entity needs
and the walk over a configuration that applies them. What stays with the
entity is the table saying which members it has -- that is the part that
is about blosc rather than about entities.
"""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeAlias, TypeVar, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._extension_points import canonical_name

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Self

    from zarr_metadata.model._validation import ProblemKind
    from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
    from zarr_metadata.v3._extension_points import ExtensionPointField

EntityT = TypeVar("EntityT", bound="MetadataEntity")

# A real alias, not a string one: entity modules subscript it as
# `Coerced[Self]` in a return annotation, and not all of them defer
# annotation evaluation.
Coerced: TypeAlias = tuple[EntityT | None, tuple[ValidationProblem, ...]]
"""The entity, or None and every reason the metadata is not one.

A caller that only wants a verdict reads the problems; one that wants to
go on reading the entity checks for None. Both never happen at once.
"""

Loc: TypeAlias = "tuple[str | int, ...]"

TypeCheck: TypeAlias = "Callable[[object, Loc], tuple[ValidationProblem, ...]]"
"""Whether one value has the type a member declares, and where if not."""

MemberTypes: TypeAlias = "Mapping[str, tuple[bool, TypeCheck]]"
"""Per configuration member: whether it is required, and its type check."""


def problem(
    loc: Loc, message: str, kind: ProblemKind = "invalid_type"
) -> tuple[ValidationProblem, ...]:
    """One problem, as the tuple every check returns."""
    return (ValidationProblem(loc, message, kind),)


def is_int(value: object, loc: Loc) -> tuple[ValidationProblem, ...]:
    """An integer, and not a bool -- JSON `true` is not the integer 1."""
    if isinstance(value, bool) or not isinstance(value, int):
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


def coerce_members(
    configuration: Mapping[str, object], types: MemberTypes
) -> tuple[dict[str, object], tuple[ValidationProblem, ...]]:
    """The members `types` declares, taken from `configuration`.

    Returns what was accepted and every problem found: a missing required
    member, a member of the wrong type, and a key the entity does not
    declare. Only a key it does not declare is survivable -- a caller can
    report it without abandoning the entity -- so problems are returned
    rather than raised and the caller decides.
    """
    problems: list[ValidationProblem] = []
    members: dict[str, object] = {}
    for key in configuration:
        if key not in types:
            problems.extend(problem(("configuration",), f"unexpected key {key!r}", "unknown_key"))
    for key, (required, check) in types.items():
        if key not in configuration:
            if required:
                problems.extend(
                    problem(("configuration", key), f"missing required key {key!r}", "missing_key")
                )
            continue
        found = check(configuration[key], ("configuration", key))
        problems.extend(found)
        if len(found) == 0:
            members[key] = configuration[key]
    return members, tuple(problems)


@dataclass(frozen=True, slots=True)
class Context:
    """The entities in scope while metadata is being read.

    A scope is not a property of the entities, it is a choice the reader
    makes: judging against the specification alone, or against the
    specification plus what `zarr-extensions` registers. The two live in
    `zarr_metadata.v3._registry`.
    """

    entities: Mapping[ExtensionPointField, Mapping[str, type[MetadataEntity]]]

    def resolve(self, field: ExtensionPointField, name: str) -> type[MetadataEntity] | None:
        """The entity `name` denotes at `field`, or None if out of scope.

        Out of scope is not an error: an unknown name may be an extension
        this reader does not model, and openness means leaving it unjudged.
        """
        return self.entities.get(field, {}).get(canonical_name(field, name))


@dataclass(frozen=True, slots=True)
class MetadataEntity:
    """One named entity, coerced from its metadata.

    Subclasses add their configuration members as fields, which is what
    makes them well-typed by construction: an instance exists only if
    `coerce` accepted the metadata that produced it.
    """

    must_understand: bool = True

    identifier: ClassVar[str]
    """The name this entity is registered under.

    Usually the `name` the metadata carries. The raw-bytes data types are
    the exception: every `r<N>` spelling is one family, so the family gets
    an invented identifier that no real name can collide with.
    """

    member_types: ClassVar[MemberTypes] = {}
    """The configuration members, and the type each one takes.

    The same keys as the configuration TypedDict, which is the same as the
    constructor signature; `tests/v3/test_entities.py` holds the three
    together.
    """

    @classmethod
    def coerce(cls, value: object, context: Context) -> Coerced[Self]:
        """`value` as this entity, or the reasons it is not one.

        `context` is the scope this reading is happening in; most entities
        have no use for it and ignore it.
        """
        raise NotImplementedError  # pragma: no cover - subclasses implement

    def problems(self) -> tuple[ValidationProblem, ...]:
        """Every value of this entity the spec disallows.

        Locations are relative to the entity's `configuration`. Default:
        an entity whose type admits only valid values has nothing to add.
        """
        return ()

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        """This entity in its simplest equivalent spelling.

        Subclasses narrow the return type to their own object TypedDict,
        which is the JSON form this dataclass models.
        """
        raise NotImplementedError  # pragma: no cover - subclasses implement


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
    if not isinstance(value, _Mapping):
        return None, None, True
    entry = cast("Mapping[str, object]", value)
    name = entry.get("name")
    if not isinstance(name, str):
        return None, None, True
    configuration = entry.get("configuration")
    must_understand = entry.get("must_understand", True)
    return (
        name,
        cast("Mapping[str, object]", configuration)
        if isinstance(configuration, _Mapping)
        else None,
        must_understand if isinstance(must_understand, bool) else True,
    )


__all__ = [
    "Coerced",
    "Context",
    "Loc",
    "MemberTypes",
    "MetadataEntity",
    "TypeCheck",
    "coerce_members",
    "is_bool",
    "is_int",
    "is_str",
    "named_configuration",
    "one_of",
    "problem",
    "sequence_of",
]
