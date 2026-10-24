"""How a format's scope files its definitions.

This is the only registry the package needs. Reading a field means
deciding which definition its name refers to. That decision depends on
the *scope* the field is read in, not on the definitions themselves. A
scope is a value built from definitions. `Context` is the protocol a
reader needs from one. `ZarrV3Context` and `ZarrV2Context`, in each
format's `definition` module, are the scopes of each format. They are
separate classes: each holds a `Slots`, one table per kind of field its
format has, and leaves the filing, lookups, equality and pickling to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import TypeVar

from zarr_metadata.definition._definition import (
    Context,
    Definition,
    as_kind,
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
    from collections.abc import Iterable, Mapping, Sequence

    from zarr_metadata.definition._definition import D
    from zarr_metadata.definition._scope import Claims, Disagreements


_NOTHING: Mapping[str, Definition[Any]] = MappingProxyType({})


@dataclass(frozen=True, slots=True, eq=False)
class Slots:
    """The definitions a format's scope files: one table per slot, keyed by the name each definition is filed under.

    `kinds` gives the kind each slot holds. Nothing holds definitions of
    two kinds together. Two `Slots` are equal when they file the same
    definitions in each slot, and equal ones have equal hashes. Build one
    with `of`, which checks each definition against its slot.
    """

    kinds: Mapping[str, type[Definition[Any]]]
    tables: Mapping[str, Mapping[str, Definition[Any]]]
    _hash: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The tables are immutable, so the hash is computed once.
        filed = frozenset((slot, frozenset(table.items())) for slot, table in self.tables.items())
        object.__setattr__(self, "_hash", hash(filed))

    @classmethod
    def of(
        cls,
        kinds: Mapping[str, type[Definition[Any]]],
        given: Mapping[str, Iterable[Definition[Any]]],
        zarr_format: int,
    ) -> Slots:
        """File the definitions `given` for each slot of a scope whose slots hold `kinds`.

        Raises `TypeError` for a slot the scope does not have, and for a
        definition of another kind than its slot holds, and
        `ScopeConflictError` for two different definitions given under one
        name in a slot.
        """
        unknown = sorted(set(given) - set(kinds))
        if len(unknown) != 0:
            msg = f"a v{zarr_format} scope has no slot {', '.join(map(repr, unknown))}"
            raise TypeError(msg)
        tables: dict[str, Mapping[str, Definition[Any]]] = {}
        conflicts: list[Conflict] = []
        for slot, kind in kinds.items():
            table: dict[str, Definition[Any]] = {}
            for definition in given.get(slot, ()):
                misfiled = _misfiled(slot, kind, definition)
                if misfiled is not None:
                    raise TypeError(misfiled)
                held = table.get(definition.name)
                if held is not None and held != definition:
                    conflicts.append(Conflict((kind, definition.name), held, definition))
                    continue
                table[definition.name] = definition
            tables[slot] = MappingProxyType(table)
        if len(conflicts) != 0:
            raise ScopeConflictError(conflicts)
        return cls(kinds, MappingProxyType(tables))

    def definitions_of(self, kind: type[D]) -> Mapping[str, D]:
        """The definitions of `kind`, by the name each is filed under: the slot that holds `kind`. Empty for a kind no slot holds."""
        asked = as_kind(kind)
        for slot, held in self.kinds.items():
            if held is asked:
                return cast("Mapping[str, D]", self.tables[slot])
        return cast("Mapping[str, D]", _NOTHING)

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` that reads `name` (a name as a document uses it), or None: the one filed under the name `filed_under` maps `name` to."""
        filed, _ = filed_under(as_kind(kind), name)
        if filed is None:
            return None
        return self.definitions_of(kind).get(filed)

    def disagreements(self, claims: Claims) -> Disagreements:
        """Where these slots read `claims` differently, looking each claim up by the name its definition is filed under."""
        return disagreements_of(lambda kind, name: self.definitions_of(kind).get(name), claims)

    def extended(self, given: Mapping[str, Iterable[Definition[Any]]], zarr_format: int) -> Slots:
        """These slots, with the definitions `given` filed in their slots, each replacing any definition already filed there under its name."""
        merged: dict[str, list[Definition[Any]]] = {}
        for slot, table in self.tables.items():
            added = list(given.get(slot, ()))
            names = {definition.name for definition in added}
            kept = [held for name, held in table.items() if name not in names]
            merged[slot] = [*kept, *added]
        return Slots.of(self.kinds, merged, zarr_format)

    @classmethod
    def joined(
        cls, kinds: Mapping[str, type[Definition[Any]]], many: Sequence[Slots], zarr_format: int
    ) -> Slots:
        """The smallest slots that file everything each of `many` files. Raises `TypeError` for slots of another format's kinds, and `ScopeConflictError` when two of them file different definitions under one name in a slot."""
        for slots in many:
            if slots.kinds != kinds:
                other = format_of(next(iter(slots.kinds.values())))
                msg = (
                    f"a scope reads documents of one Zarr format; a v{zarr_format} scope "
                    f"joins v{zarr_format} scopes, got a v{other} scope"
                )
                raise TypeError(msg)
        merged: dict[str, dict[str, Definition[Any]]] = {slot: {} for slot in kinds}
        conflicts: list[Conflict] = []
        for slots in many:
            for slot, table in slots.tables.items():
                for name, definition in table.items():
                    held = merged[slot].get(name)
                    if held is not None and held != definition:
                        conflicts.append(Conflict((kinds[slot], name), held, definition))
                        continue
                    merged[slot][name] = definition
        if len(conflicts) != 0:
            raise ScopeConflictError(conflicts)
        return cls.of(
            kinds, {slot: list(table.values()) for slot, table in merged.items()}, zarr_format
        )

    def contents(self) -> dict[str, tuple[Definition[Any], ...]]:
        """The definitions in each slot, by slot: what a scope is pickled as."""
        return {slot: tuple(table.values()) for slot, table in self.tables.items()}

    def __len__(self) -> int:
        return sum(len(table) for table in self.tables.values())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Slots):
            return NotImplemented
        return self.tables == other.tables

    def __hash__(self) -> int:
        return self._hash


def _misfiled(slot: str, kind: type[Definition[Any]], value: object) -> str | None:
    """Why `value` does not belong in `slot`, which holds definitions of `kind`; None when it does. `value` is checked as a value of any type, for a caller without a type checker."""
    found = kind_of(value)
    if found is kind:
        return None
    if not isinstance(value, Definition):
        return f"{slot} holds {kind_name(kind)} definitions, got {value!r}"
    of = "a definition of no kind" if found is None else f"a {kind_name(found)} definition"
    return f"{slot} holds {kind_name(kind)} definitions, got {value.name!r}, {of}"


C = TypeVar("C", bound=Context)


def scoped(context: Context | None, default: C) -> C:
    """The scope for a reader of `default`'s format: `context`, or `default` when `context` is None. Raises `TypeError` for a scope of another class, which reads another format."""
    if context is None:
        return default
    if not isinstance(context, type(default)):
        msg = (
            f"a v{default.zarr_format} document is read in a scope of that format, "
            f"got a v{context.zarr_format} scope"
        )
        raise TypeError(msg)
    return context


__all__ = [
    "Slots",
    "scoped",
]
