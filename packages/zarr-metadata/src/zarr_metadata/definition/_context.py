"""How a format's scope files its definitions.

No global registry files definitions. Reading a field means
deciding which definition its name refers to. That decision depends on
the *scope* the field is read in, not on the definitions themselves. A
scope is a value built from definitions. `Context` is the protocol a
reader needs from one. `ZarrV3Context` and `ZarrV2Context`, in each
format's `definition` module, are the scopes of each format. They are
separate classes. Each holds its definitions as tables: one table per
kind, keyed by the name each definition is filed under. The functions
here file, look up, update and join such tables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias, cast, get_args, get_origin, get_type_hints

from typing_extensions import TypeIs, TypeVar

from zarr_metadata.definition._definition import (
    Context,
    Definition,
    as_kind,
    described,
    filed_under,
    format_of,
    kind_of,
)
from zarr_metadata.definition._scope import (
    Conflict,
    ScopeConflictError,
    disagreements_of,
    kind_name,
)

if TYPE_CHECKING:
    from zarr_metadata.definition._definition import D
    from zarr_metadata.definition._scope import Claims, Disagreements
    from zarr_metadata.v2._context import ZarrV2Context
    from zarr_metadata.v3._context import ZarrV3Context


Tables: TypeAlias = Mapping[type[Definition[Any]], Mapping[str, Definition[Any]]]
"""A scope's definitions: one table for each kind its format has, keyed by the name each definition is filed under. A table holds definitions of one kind only."""

Given: TypeAlias = Iterable[Definition[Any]] | Mapping[str, Definition[Any]]
"""The definitions given for one slot of a scope: an iterable of them, or a mapping from the name each is filed under to it, as a scope's slot is."""

K = TypeVar("K", bound=Definition[Any])


def slot_kinds(
    slots: type, accepts: Callable[[type[Definition[Any]]], TypeIs[type[K]]]
) -> Mapping[str, type[K]]:
    """The kind each slot of a scope holds, read from `slots`: the TypedDict a scope's constructor takes, whose keys are the slots, each annotated `Iterable[K] | Mapping[str, K]` for a kind `K` that `accepts`, the check for a kind of the scope's format. A scope class derives its slots from it, so they are declared once.

    Raises `TypeError`, when the scope class is defined, for a member
    annotated any other way, or with a kind that is not of the format.
    """
    kinds: dict[str, type[K]] = {}
    for slot, annotation in get_type_hints(slots).items():
        held = _held(annotation)
        if held is None:
            msg = f"{slots.__name__}.{slot}: a slot is annotated Iterable[K] | Mapping[str, K] for a kind K, got {annotation!r}"
            raise TypeError(msg)
        kind = as_kind(held)
        if not accepts(kind):
            msg = f"{slots.__name__}.{slot}: {kind.__name__} is not a kind of the scope's format"
            raise TypeError(msg)
        kinds[slot] = kind
    return MappingProxyType(kinds)


def _held(annotation: object) -> object:
    """The kind `K` that a slot annotated `Iterable[K] | Mapping[str, K]` holds, or None for any other annotation."""
    arms = get_args(annotation)
    if len(arms) != 2 or get_origin(arms[0]) is not Iterable or len(get_args(arms[0])) != 1:
        return None
    (held,) = get_args(arms[0])
    return held if arms[1] == Mapping[str, held] else None


def filed(
    kinds: Mapping[str, type[Definition[Any]]],
    given: Mapping[str, Given],
    zarr_format: int,
) -> Tables:
    """The tables of the definitions `given` for each slot of a scope whose slots hold `kinds`: one table per kind, each kind of the format present.

    Raises `TypeError` for a slot the scope does not have, for a
    definition of another kind than its slot holds, and for a slot given
    as a mapping with a key that is not the name of its definition, and
    `ScopeConflictError` for two different definitions given under one
    name in a slot.
    """
    unknown = sorted(set(given) - set(kinds))
    if len(unknown) != 0:
        msg = f"a v{zarr_format} scope has no slot {', '.join(map(repr, unknown))}"
        raise TypeError(msg)
    tables: dict[type[Definition[Any]], Mapping[str, Definition[Any]]] = {}
    conflicts: list[Conflict] = []
    for slot, kind in kinds.items():
        table: dict[str, Definition[Any]] = {}
        definitions = given.get(slot, ())
        if _is_keyed(definitions):
            _refuse_misnamed(slot, definitions)
            definitions = definitions.values()
        for definition in definitions:
            misfiled = _misfiled(slot, kind, definition)
            if misfiled is not None:
                raise TypeError(misfiled)
            held = table.get(definition.name)
            if held is not None and held != definition:
                conflicts.append(Conflict((kind, definition.name), held, definition))
                continue
            table[definition.name] = definition
        tables[kind] = MappingProxyType(table)
    if len(conflicts) != 0:
        raise ScopeConflictError(conflicts)
    return MappingProxyType(tables)


def updated(
    tables: Tables,
    kinds: Mapping[str, type[Definition[Any]]],
    given: Mapping[str, Given],
    zarr_format: int,
) -> Tables:
    """`tables`, with the table of each slot `given` replaced by the definitions given for it. The tables of the other slots are kept. Raises as `filed` does."""
    replacing = filed(kinds, given, zarr_format)
    replaced = {kinds[slot] for slot in given}
    return MappingProxyType(
        {kind: replacing[kind] if kind in replaced else table for kind, table in tables.items()}
    )


def joined_tables(kinds: Mapping[str, type[Definition[Any]]], many: Iterable[Tables]) -> Tables:
    """The smallest tables that file everything each of `many`, tables of the format whose slots hold `kinds`, files: their join.

    Raises `ScopeConflictError` when two of them file different
    definitions under one name of a kind.
    """
    merged: dict[type[Definition[Any]], dict[str, Definition[Any]]] = {
        kind: {} for kind in kinds.values()
    }
    conflicts: list[Conflict] = []
    for tables in many:
        for kind, table in tables.items():
            for name, definition in table.items():
                held = merged[kind].get(name)
                if held is not None and held != definition:
                    conflicts.append(Conflict((kind, name), held, definition))
                    continue
                merged[kind][name] = definition
    if len(conflicts) != 0:
        raise ScopeConflictError(conflicts)
    return MappingProxyType({kind: MappingProxyType(table) for kind, table in merged.items()})


def definitions_in(tables: Tables, kind: type[D]) -> Mapping[str, D]:
    """The table of `kind` in `tables`, by the name each definition is filed under.

    Tables hold every kind of their format, so a kind they do not hold is
    a kind of the other format, and raises `TypeError`, as anything that is
    not a kind does.
    """
    asked = as_kind(kind)
    if asked not in tables:
        msg = (
            f"a v{format_of(next(iter(tables)))} scope holds the kinds of its format, "
            f"got {kind_name(asked)} definitions, a kind of v{format_of(asked)}"
        )
        raise TypeError(msg)
    return cast("Mapping[str, D]", tables[asked])


def claimant_in(tables: Tables, kind: type[D], name: str) -> D | None:
    """The definition of `kind` in `tables` that reads `name` (a name as a document uses it), or None: the one filed under the name `filed_under` maps `name` to."""
    table = definitions_in(tables, kind)
    filed_name, _ = filed_under(as_kind(kind), name)
    return None if filed_name is None else table.get(filed_name)


def disagreements(context: ZarrV3Context | ZarrV2Context, claims: Claims) -> Disagreements:
    """Where `context` reads `claims`, a reading's claims, differently: the names it would gain and the names it conflicts with, as `Disagreements` describes them. A claim is keyed by the name its definition is filed under, so it is looked up by that name. Raises `TypeError` for a claim of the other format's kind, as `definitions_in` does.

    A model calls this to check a scope it is moved to. It is not a method
    of the scope classes, whose public API is the definitions they hold.
    """
    # This module files, looks up and compares the tables a scope holds,
    # so it reads them here.
    tables = context._tables  # pyright: ignore[reportPrivateUsage]
    return disagreements_of(lambda kind, name: definitions_in(tables, kind).get(name), claims)


def slot_contents(
    tables: Tables, kinds: Mapping[str, type[Definition[Any]]]
) -> dict[str, tuple[Definition[Any], ...]]:
    """The definitions in `tables`, by the slot each kind is given in: a scope's constructor arguments, which it pickles as."""
    return {slot: tuple(tables[kind].values()) for slot, kind in kinds.items()}


def tables_key(
    tables: Tables,
) -> frozenset[tuple[type[Definition[Any]], frozenset[tuple[str, Definition[Any]]]]]:
    """What `tables` file, as a value to hash: each kind, with its names and definitions."""
    return frozenset((kind, frozenset(table.items())) for kind, table in tables.items())


def _is_keyed(given: Given) -> TypeIs[Mapping[str, Definition[Any]]]:
    """Whether the definitions `given` for a slot are a mapping by name rather than an iterable of definitions. A mapping keyed some other way is caught by `_refuse_misnamed`."""
    return isinstance(given, Mapping)


def _refuse_misnamed(slot: str, definitions: Mapping[str, object]) -> None:
    """Raise `TypeError` for a key of `definitions`, a slot given as a mapping, that is not the name of the definition it holds. A value that is not a definition is left for `_misfiled` to refuse."""
    for key, value in definitions.items():
        if isinstance(value, Definition) and key != value.name:
            msg = f"{slot} files each definition under its name, got {value.name!r} under {key!r}"
            raise TypeError(msg)


def _misfiled(slot: str, kind: type[Definition[Any]], value: object) -> str | None:
    """Why `value` does not belong in `slot`, which holds definitions of `kind`; None when it does. `value` is checked as a value of any type, for a caller without a type checker."""
    found = kind_of(value)
    if found is kind:
        return None
    if not isinstance(value, Definition):
        return f"{slot} holds {kind_name(kind)} definitions, got {value!r}"
    of = "a definition of no kind" if found is None else f"a {kind_name(found)} definition"
    return f"{slot} holds {kind_name(kind)} definitions, got {value.name!r}, {of}"


def refuse_others(contexts: Iterable[object], scope: type[ZarrV2Context | ZarrV3Context]) -> None:
    """Raise `TypeError` unless each of `contexts` is a `scope`, the class of one format's scopes, so a scope's `union` takes only scopes of its own format."""
    for context in contexts:
        if not isinstance(context, scope):
            zarr_format = format_of(next(iter(scope.kinds.values())))
            msg = (
                f"a scope reads documents of one Zarr format; a union of v{zarr_format} scopes "
                f"takes v{zarr_format} scopes, got {described(context)}"
            )
            raise TypeError(msg)


C = TypeVar("C", bound=Context)


def scoped(context: object, default: C) -> C:
    """The scope for a reader of `default`'s format: `context`, or `default` when `context` is None. Raises `TypeError` for anything else, such as a scope of the other format, or a value that is not one of this package's scopes."""
    if context is None:
        return default
    if not isinstance(context, type(default)):
        msg = (
            f"a v{default.zarr_format} document is read in a scope of that format, a "
            f"{type(default).__name__}, got {described(context)}"
        )
        raise TypeError(msg)
    return context


__all__ = [
    "Given",
    "Tables",
    "claimant_in",
    "definitions_in",
    "disagreements",
    "filed",
    "joined_tables",
    "refuse_others",
    "scoped",
    "slot_contents",
    "slot_kinds",
    "tables_key",
    "updated",
]
