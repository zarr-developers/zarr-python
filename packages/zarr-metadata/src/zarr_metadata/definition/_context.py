"""Which definitions are in scope while a document is read.

This is the only registry the package needs. Reading a field means
deciding which definition its name refers to. That decision depends on
the *scope* the field is read in, not on the definitions themselves. A
scope is a value built from definitions. `Context` is what the scopes of
both formats share. `ZarrV3Context` and `ZarrV2Context`, in each
format's `definition` module, hold one slot per kind of field that a
document of the format contains, and file each definition there under
its name.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeGuard, cast

from typing_extensions import TypeVar

from zarr_metadata.definition._definition import (
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
    from collections.abc import Iterable, Mapping

    from zarr_metadata.definition._definition import D
    from zarr_metadata.definition._scope import Claims, Disagreements


_NOTHING: Mapping[str, Definition[Any]] = MappingProxyType({})


class Context:
    """The definitions in scope while metadata of one Zarr format is read: the base of `ZarrV3Context` and `ZarrV2Context`.

    A scope is a value and does no reading itself. `resolve` reads a field
    in a scope. The question a scope answers for a reader is `claimant`:
    which definition a name belongs to. Each format's scope has one slot
    per kind of field, named like the document member that holds such
    fields (`codecs`, `data_types`, ...), and files each definition in
    its slot under its name. Build a scope with the format's class, and
    add definitions with `extended_with`. Two scopes are equal when they
    are of one format and file the same definitions, and equal scopes have
    equal hashes.
    """

    __slots__ = ("_definitions", "_filed", "_tables")

    kinds: ClassVar[Mapping[str, type[Definition[Any]]]]
    """The kind of definition each slot of a scope of this format holds, by slot."""

    _definitions: tuple[Definition[Any], ...]
    _filed: frozenset[tuple[type[Definition[Any]], str, Definition[Any]]]
    _tables: Mapping[type[Definition[Any]], Mapping[str, Definition[Any]]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Every slot of a format's scope holds a kind of that format, which
        # is the scope's format.
        super().__init_subclass__(**kwargs)
        formats = sorted({format_of(kind) for kind in cls.kinds.values()})
        if len(formats) != 1:
            msg = f"{cls.__name__} holds kinds of Zarr formats {formats}; a scope is of one format"
            raise TypeError(msg)

    @property
    def zarr_format(self) -> Literal[2, 3]:
        """The Zarr format of the documents this scope reads: the format of every kind it holds."""
        return format_of(next(iter(self.kinds.values())))

    def __init__(self, slots: Mapping[str, Iterable[Definition[Any]]]) -> None:
        """File the definitions given for each slot. A format's class calls this with its own slots.

        Raises `TypeError` for a definition of another kind than its slot
        holds, and `ScopeConflictError` for two different definitions given
        under one name in a slot.
        """
        if type(self) is Context:
            msg = "a scope reads documents of one Zarr format; build a ZarrV3Context or a ZarrV2Context"
            raise TypeError(msg)
        unknown = sorted(set(slots) - set(self.kinds))
        if len(unknown) != 0:
            msg = f"a v{self.zarr_format} scope has no slot {', '.join(map(repr, unknown))}"
            raise TypeError(msg)
        tables: dict[type[Definition[Any]], Mapping[str, Definition[Any]]] = {}
        conflicts: list[Conflict] = []
        for slot, kind in self.kinds.items():
            table: dict[str, Definition[Any]] = {}
            for definition in slots.get(slot, ()):
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
        # A scope is immutable, so what it files is computed once: every
        # definition, and every definition with the kind and name it is
        # filed under, which scopes are compared by.
        self._tables = MappingProxyType(tables)
        self._definitions = tuple(entry for table in tables.values() for entry in table.values())
        self._filed = frozenset(
            (kind, name, definition)
            for kind, table in tables.items()
            for name, definition in table.items()
        )

    @classmethod
    def _built(cls, slots: Mapping[str, Iterable[Definition[Any]]]) -> Self:
        """A scope of this class, with `slots` filed as the constructor files them."""
        context = cls.__new__(cls)
        Context.__init__(context, slots)
        return context

    def _held(self, kind: type[Definition[Any]]) -> Mapping[str, Definition[Any]]:
        """The definitions of `kind` in scope, keyed by the name each is filed under. Empty for a kind of no slot."""
        return self._tables.get(kind, _NOTHING)

    @property
    def definitions(self) -> tuple[Definition[Any], ...]:
        """Every definition in scope, slot by slot."""
        return self._definitions

    def _extended(self, slots: Mapping[str, Iterable[Definition[Any]]]) -> Self:
        """This scope, with the definitions in `slots` filed in their slots, each replacing any definition already filed there under its name."""
        merged: dict[str, list[Definition[Any]]] = {}
        for slot, kind in self.kinds.items():
            given = list(slots.get(slot, ()))
            names = {definition.name for definition in given}
            kept = [held for name, held in self._held(kind).items() if name not in names]
            merged[slot] = [*kept, *given]
        return self._built(merged)

    def _slot_contents(self) -> dict[str, tuple[Definition[Any], ...]]:
        """The definitions in each slot, by slot name."""
        return {slot: tuple(self._held(kind).values()) for slot, kind in self.kinds.items()}

    def __repr__(self) -> str:
        # Keep this short, because it appears in signatures as a default
        # argument. A full repr would list every definition, and `help` on a
        # validator would run to pages.
        return f"{type(self).__name__}(<{len(self._definitions)} definitions>)"

    def __reduce__(self) -> tuple[object, tuple[dict[str, tuple[Definition[Any], ...]]]]:
        # A scope is defined by its definitions, so pickle it as them. This
        # lets a scope go to another process along with the documents it
        # will read there.
        return (type(self)._built, (self._slot_contents(),))

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        # A scope is immutable, so a copy can be the same object.
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Context):
            return NotImplemented
        return type(self) is type(other) and self._filed == other._filed

    def __hash__(self) -> int:
        return hash((type(self), self._filed))

    def disagreements(self, claims: Claims) -> Disagreements:
        """The places where this scope reads `claims` (a reading's claims) differently: the names it would gain and the names it conflicts with, as described by `Disagreements`.

        A claim is keyed by the name its definition is filed under (raw
        bits under `r*`), so it is looked up by the filed name, not by the
        name a document uses.
        """
        return disagreements_of(lambda kind, name: self._held(kind).get(name), claims)

    @classmethod
    def joined(cls, *contexts: Self) -> Self:
        """The smallest scope that files everything each of `contexts` files (their join), of the format they share.

        Raises `ScopeConflictError` when two of them file different
        definitions under the same kind and name. To replace a definition
        on purpose, use `extended_with`. Raises `TypeError` for scopes of
        two formats, and for no scope when called on `Context`, whose
        format is unknown.
        """
        classes = {type(context) for context in contexts}
        if len(classes) > 1:
            formats = " and ".join(sorted(f"v{found.zarr_format}" for found in classes))
            msg = (
                f"a scope reads documents of one Zarr format; these scopes are of formats {formats}"
            )
            raise TypeError(msg)
        target = classes.pop() if len(classes) == 1 else cls
        if target is Context:
            msg = "join scopes of one format, or call joined on ZarrV3Context or ZarrV2Context"
            raise TypeError(msg)
        filed: dict[tuple[type[Definition[Any]], str], Definition[Any]] = {}
        conflicts: list[Conflict] = []
        for context in contexts:
            for kind, table in context._tables.items():
                for name, definition in table.items():
                    held = filed.get((kind, name))
                    if held is not None and held != definition:
                        conflicts.append(Conflict((kind, name), held, definition))
                        continue
                    filed[kind, name] = definition
        if len(conflicts) != 0:
            raise ScopeConflictError(conflicts)
        slots = {
            slot: [definition for (held, _), definition in filed.items() if held is kind]
            for slot, kind in target.kinds.items()
        }
        return target._built(slots)

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` in scope that reads `name` (a name as a document uses it), or None if there is none.

        This is the definition filed under the name that `filed_under`
        maps `name` to. That is `name` itself, except for raw bits, where
        `r16` is read by the definition of `r*`.
        """
        asked = as_kind(kind)
        filed, _ = filed_under(asked, name)
        if filed is None:
            return None
        return cast("D | None", self._held(asked).get(filed))


def _misfiled(slot: str, kind: type[Definition[Any]], value: object) -> str | None:
    """Why `value` does not belong in `slot`, which holds definitions of `kind`; None when it does. `value` is checked as a value of any type, for a caller without a type checker."""
    found = kind_of(value)
    if found is kind:
        return None
    if not isinstance(value, Definition):
        return f"{slot} holds {kind_name(kind)} definitions, got {value!r}"
    of = "a definition of no kind" if found is None else f"a {kind_name(found)} definition"
    return f"{slot} holds {kind_name(kind)} definitions, got {value.name!r}, {of}"


def is_context(value: object) -> TypeGuard[Context]:
    """Whether `value` is a scope, of any format."""
    return isinstance(value, Context)


C = TypeVar("C", bound=Context)


def scoped(context: Context | None, default: C) -> C:
    """The scope for a reader of `default`'s format: `context`, or `default` when `context` is None. Raises `TypeError` for a scope of the other format."""
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
    "Context",
    "is_context",
    "scoped",
]
