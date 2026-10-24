"""The scope a Zarr v3 document is read in: one slot for each kind of v3 field."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from typing_extensions import TypedDict, Unpack

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
from zarr_metadata.v3._definition import (
    ArrayExtraFieldDefinition,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    GroupExtraFieldDefinition,
    StorageTransformerDefinition,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.definition._definition import D, Definition
    from zarr_metadata.definition._scope import Claims, Disagreements


class ZarrV3ContextSlots(TypedDict, total=False, closed=True):
    """The definitions in each slot of a v3 scope, as `ZarrV3Context` and its `extended_with` take them.

    This is where a v3 scope's slots are declared: each key is a slot,
    and its value the definitions of the kind that slot holds.
    `ZarrV3Context.kinds` is read from these annotations, so a type
    checker checks every slot a caller names, and every definition given
    in it.
    """

    data_types: Iterable[DataTypeDefinition[Any]]
    """Data types, read from `data_type`."""
    chunk_grids: Iterable[ChunkGridDefinition[Any]]
    """Chunk grids, read from `chunk_grid`."""
    chunk_key_encodings: Iterable[ChunkKeyEncodingDefinition[Any]]
    """Chunk key encodings, read from `chunk_key_encoding`."""
    codecs: Iterable[CodecDefinition[Any]]
    """Codecs, read from `codecs` and from the codecs a codec holds."""
    storage_transformers: Iterable[StorageTransformerDefinition[Any]]
    """Storage transformers, read from `storage_transformers`."""
    array_extra_fields: Iterable[ArrayExtraFieldDefinition[Any]]
    """Members of an array document that the spec does not define, by key."""
    group_extra_fields: Iterable[GroupExtraFieldDefinition[Any]]
    """Members of a group document that the spec does not define, by key."""


class ZarrV3Context:
    """A scope that reads Zarr v3 documents. Every v3 reader takes one.

    It implements `Context`, and has one slot for each kind of field a v3
    document holds, as `ZarrV3ContextSlots` declares them. The slots for
    the spec's extension points are named like the document member that
    holds such fields: `data_types`, `chunk_grids`, `chunk_key_encodings`,
    `codecs` and `storage_transformers`. `array_extra_fields` and
    `group_extra_fields` hold the definitions of members the spec does not
    define, by key. The constructor takes an iterable of definitions for
    each slot, and each slot is a read-only mapping from the name each
    definition is filed under (`r*` for raw bits) to the definition. A
    definition of another kind than its slot holds raises `TypeError`, and
    two different definitions under one name in a slot raise
    `ScopeConflictError`. To replace a definition, use `extended_with`.
    `ZarrV3Context()` is the scope that claims nothing. Two scopes are
    equal when they file the same definitions in each slot, and equal
    scopes have equal hashes.
    """

    __slots__ = ("_tables",)

    kinds: ClassVar[Mapping[str, type[Definition[Any]]]] = slot_kinds(ZarrV3ContextSlots)
    """The kind of definition each slot holds, by slot, as `ZarrV3ContextSlots` declares them."""

    _tables: Tables

    def __init__(self, **slots: Unpack[ZarrV3ContextSlots]) -> None:
        self._tables = filed(self.kinds, slots, 3)

    @classmethod
    def _of(cls, tables: Tables) -> ZarrV3Context:
        """A scope of `tables`, which were filed for this class's slots."""
        context = object.__new__(cls)
        context._tables = tables
        return context

    @classmethod
    def _restored(cls, contents: Mapping[str, Iterable[Definition[Any]]]) -> ZarrV3Context:
        """A scope rebuilt from the definitions in each slot, as `__reduce__` gives them."""
        return cls._of(filed(cls.kinds, contents, 3))

    @property
    def zarr_format(self) -> Literal[3]:
        """The Zarr format of the documents this scope reads: 3."""
        return 3

    @property
    def data_types(self) -> Mapping[str, DataTypeDefinition[Any]]:
        """The data type definitions in scope, by the name each is filed under (`r*` for raw bits)."""
        return self.definitions_of(DataTypeDefinition)

    @property
    def chunk_grids(self) -> Mapping[str, ChunkGridDefinition[Any]]:
        """The chunk grid definitions in scope, by name."""
        return self.definitions_of(ChunkGridDefinition)

    @property
    def chunk_key_encodings(self) -> Mapping[str, ChunkKeyEncodingDefinition[Any]]:
        """The chunk key encoding definitions in scope, by name."""
        return self.definitions_of(ChunkKeyEncodingDefinition)

    @property
    def codecs(self) -> Mapping[str, CodecDefinition[Any]]:
        """The codec definitions in scope, by name."""
        return self.definitions_of(CodecDefinition)

    @property
    def storage_transformers(self) -> Mapping[str, StorageTransformerDefinition[Any]]:
        """The storage transformer definitions in scope, by name."""
        return self.definitions_of(StorageTransformerDefinition)

    @property
    def array_extra_fields(self) -> Mapping[str, ArrayExtraFieldDefinition[Any]]:
        """The definitions of members of an array document that the spec does not define, by key."""
        return self.definitions_of(ArrayExtraFieldDefinition)

    @property
    def group_extra_fields(self) -> Mapping[str, GroupExtraFieldDefinition[Any]]:
        """The definitions of members of a group document that the spec does not define, by key."""
        return self.definitions_of(GroupExtraFieldDefinition)

    def definitions_of(self, kind: type[D]) -> Mapping[str, D]:
        """The definitions of `kind` in scope, by the name each is filed under: the slot that holds `kind`. Empty for a kind that no slot holds, such as a kind of the other format."""
        return definitions_in(self._tables, kind)

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` in scope that reads `name` (a name as a document uses it), or None if there is none.

        This is the definition filed under the name that `filed_under`
        maps `name` to. That is `name` itself, except for raw bits, where
        `r16` is read by the definition of `r*`.
        """
        return claimant_in(self._tables, kind, name)

    def disagreements(self, claims: Claims) -> Disagreements:
        """The places where this scope reads `claims` (a reading's claims) differently: the names it would gain and the names it conflicts with, as described by `Disagreements`. A claim is keyed by the name its definition is filed under, so it is looked up by that name."""
        return disagreements_in(self._tables, claims)

    def extended_with(self, **slots: Unpack[ZarrV3ContextSlots]) -> ZarrV3Context:
        """This scope, plus definitions of your own, each in its slot.

        A definition given here replaces any definition already filed in
        its slot under the same name. This is how a reader substitutes its
        own reading of a definition the package already defines.
        """
        return self._of(extended(self._tables, self.kinds, slots, 3))

    @classmethod
    def joined(cls, *contexts: ZarrV3Context) -> ZarrV3Context:
        """The smallest scope that files everything each of `contexts` files (their join).

        Raises `ScopeConflictError` when two of them file different
        definitions under one name in a slot. To replace a definition on
        purpose, use `extended_with`. Raises `TypeError` for a scope of
        another format.
        """
        return cls._of(joined_tables(cls.kinds, [context._tables for context in contexts], 3))

    def __repr__(self) -> str:
        # Keep this short, because it appears in signatures as a default
        # argument. A full repr would list every definition, and `help` on a
        # validator would run to pages.
        count = sum(len(table) for table in self._tables.values())
        return f"ZarrV3Context(<{count} definitions>)"

    def __reduce__(self) -> tuple[object, tuple[dict[str, tuple[Definition[Any], ...]]]]:
        # A scope is defined by the definitions in its slots, so pickle it as
        # them. This lets a scope go to another process along with the
        # documents it will read there.
        return (type(self)._restored, (slot_contents(self._tables, self.kinds),))

    def __copy__(self) -> ZarrV3Context:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ZarrV3Context:
        # A scope is immutable, so a copy can be the same object.
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZarrV3Context):
            return NotImplemented
        return self._tables == other._tables

    def __hash__(self) -> int:
        return hash((ZarrV3Context, tables_key(self._tables)))


__all__ = ["ZarrV3Context", "ZarrV3ContextSlots"]
