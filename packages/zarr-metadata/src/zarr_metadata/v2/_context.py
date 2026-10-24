"""The scope a Zarr v2 document is read in: one slot for each kind of v2 field."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypeAlias, get_args, get_origin

from typing_extensions import TypedDict, TypeIs, TypeVar, Unpack

from zarr_metadata.definition._context import (
    Tables,
    claimant_in,
    definitions_in,
    disagreements_in,
    extended,
    filed,
    joined_tables,
    slot_contents,
    slot_kinds,
    tables_key,
)
from zarr_metadata.v2._definition import ZarrV2CodecDefinition, ZarrV2DataTypeDefinition

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.definition._definition import D, Definition
    from zarr_metadata.definition._scope import Claims, Disagreements

ZarrV2Kind: TypeAlias = ZarrV2DataTypeDefinition[Any] | ZarrV2CodecDefinition[Any]
"""A definition of a kind of v2 field."""

_K = TypeVar("_K", bound=ZarrV2Kind)
"""A kind of v2 field, so a type checker refuses a kind of the other format."""

_KINDS: Final[frozenset[type[Definition[Any]]]] = frozenset(
    get_origin(member) for member in get_args(ZarrV2Kind)
)


def is_v2_kind(kind: type[Definition[Any]]) -> TypeIs[type[ZarrV2Kind]]:
    """Whether `kind` is a kind of v2 field: one of the kinds in `ZarrV2Kind`."""
    return kind in _KINDS


class ZarrV2ContextSlots(TypedDict, total=False, closed=True):
    """The definitions in each slot of a v2 scope, as `ZarrV2Context` and its `extended_with` take them.

    This is where a v2 scope's slots are declared: each key is a slot,
    and its value the definitions of the kind that slot holds.
    `ZarrV2Context.kinds` is read from these annotations, so a type
    checker checks every slot a caller names, and every definition given
    in it.
    """

    data_types: Iterable[ZarrV2DataTypeDefinition[Any]]
    """Data types, read from `dtype`."""
    codecs: Iterable[ZarrV2CodecDefinition[Any]]
    """Codecs, read from `compressor` and `filters`."""


class ZarrV2Context:
    """A scope that reads Zarr v2 documents. Every v2 reader takes one.

    It implements `Context`, and has one slot for each kind of field a v2
    document holds, as `ZarrV2ContextSlots` declares them: `data_types`,
    read from `dtype`, and `codecs`, read from `compressor` and `filters`.
    The constructor takes an iterable of definitions for each slot, and
    each slot is a read-only mapping from name to definition. A definition
    of another kind than its slot holds raises `TypeError`, and two
    different definitions under one name in a slot raise
    `ScopeConflictError`. To replace a definition, use `extended_with`.
    `ZarrV2Context()` is the scope that claims nothing. Two scopes are
    equal when they file the same definitions in each slot, and equal
    scopes have equal hashes.
    """

    __slots__ = ("_tables",)

    kinds: ClassVar[Mapping[str, type[ZarrV2Kind]]] = slot_kinds(ZarrV2ContextSlots, is_v2_kind)
    """The kind of definition each slot holds, by slot, as `ZarrV2ContextSlots` declares them."""

    _tables: Tables

    def __init__(self, **slots: Unpack[ZarrV2ContextSlots]) -> None:
        self._tables = filed(self.kinds, slots, 2)

    @classmethod
    def _of(cls, tables: Tables) -> ZarrV2Context:
        """A scope of `tables`, which were filed for this class's slots."""
        context = object.__new__(cls)
        context._tables = tables
        return context

    @classmethod
    def _restored(cls, contents: Mapping[str, Iterable[Definition[Any]]]) -> ZarrV2Context:
        """A scope rebuilt from the definitions in each slot, as `__reduce__` gives them."""
        return cls._of(filed(cls.kinds, contents, 2))

    @property
    def zarr_format(self) -> Literal[2]:
        """The Zarr format of the documents this scope reads: 2."""
        return 2

    @property
    def data_types(self) -> Mapping[str, ZarrV2DataTypeDefinition[Any]]:
        """The data type definitions in scope, by the NumPy family each reads."""
        return self.definitions_of(ZarrV2DataTypeDefinition)

    @property
    def codecs(self) -> Mapping[str, ZarrV2CodecDefinition[Any]]:
        """The codec definitions in scope, by numcodecs id."""
        return self.definitions_of(ZarrV2CodecDefinition)

    def definitions_of(self, kind: type[_K]) -> Mapping[str, _K]:
        """The definitions of `kind` in scope, by the name each is filed under: the slot that holds `kind`. Raises `TypeError` for a kind of the other format."""
        return definitions_in(self._tables, kind)

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` in scope that reads `name` (a name as a document uses it), or None if there is none. Raises `TypeError` for a kind of the other format.

        This is the definition filed under the name that `filed_under`
        maps `name` to. That is `name` itself, except for raw bits, where
        `r16` is read by the definition of `r*`.
        """
        return claimant_in(self._tables, kind, name)

    def disagreements(self, claims: Claims) -> Disagreements:
        """The places where this scope reads `claims` (a reading's claims) differently: the names it would gain and the names it conflicts with, as described by `Disagreements`. A claim is keyed by the name its definition is filed under, so it is looked up by that name."""
        return disagreements_in(self._tables, claims)

    def extended_with(self, **slots: Unpack[ZarrV2ContextSlots]) -> ZarrV2Context:
        """This scope, plus definitions of your own, each in its slot.

        A definition given here replaces any definition already filed in
        its slot under the same name. This is how a reader substitutes its
        own reading of a definition the package already defines.
        """
        return self._of(extended(self._tables, self.kinds, slots, 2))

    @classmethod
    def joined(cls, *contexts: ZarrV2Context) -> ZarrV2Context:
        """The smallest scope that files everything each of `contexts` files (their join).

        Raises `ScopeConflictError` when two of them file different
        definitions under one name in a slot. To replace a definition on
        purpose, use `extended_with`. Raises `TypeError` for a scope of
        another format.
        """
        return cls._of(joined_tables(cls.kinds, [context._tables for context in contexts], 2))

    def __repr__(self) -> str:
        # Keep this short, because it appears in signatures as a default
        # argument. A full repr would list every definition, and `help` on a
        # validator would run to pages.
        count = sum(len(table) for table in self._tables.values())
        return f"ZarrV2Context(<{count} definitions>)"

    def __reduce__(self) -> tuple[object, tuple[dict[str, tuple[Definition[Any], ...]]]]:
        # A scope is defined by the definitions in its slots, so pickle it as
        # them. This lets a scope go to another process along with the
        # documents it will read there.
        return (type(self)._restored, (slot_contents(self._tables, self.kinds),))

    def __copy__(self) -> ZarrV2Context:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ZarrV2Context:
        # A scope is immutable, so a copy can be the same object.
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZarrV2Context):
            return NotImplemented
        return self._tables == other._tables

    def __hash__(self) -> int:
        return hash((ZarrV2Context, tables_key(self._tables)))


__all__ = ["ZarrV2Context", "ZarrV2ContextSlots", "ZarrV2Kind", "is_v2_kind"]
