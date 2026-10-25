"""The scope a Zarr v3 document is read in: one slot for each kind of v3 field."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    TypeAlias,
    final,
    get_args,
    get_origin,
)

from typing_extensions import TypedDict, TypeIs, TypeVar, Unpack

from zarr_metadata.definition._context import (
    Tables,
    claimant_in,
    definitions_in,
    filed,
    joined_tables,
    refuse_others,
    slot_contents,
    slot_kinds,
    tables_key,
    updated,
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
    from zarr_metadata.definition._definition import D, Definition

ZarrV3Kind: TypeAlias = (
    DataTypeDefinition[Any]
    | ChunkGridDefinition[Any]
    | ChunkKeyEncodingDefinition[Any]
    | CodecDefinition[Any]
    | StorageTransformerDefinition[Any]
    | ArrayExtraFieldDefinition[Any]
    | GroupExtraFieldDefinition[Any]
)
"""A definition of a kind of v3 field."""

_K = TypeVar("_K", bound=ZarrV3Kind)
"""A kind of v3 field, so a type checker refuses a kind of the other format."""

_KINDS: Final[frozenset[type[Definition[Any]]]] = frozenset(
    get_origin(member) for member in get_args(ZarrV3Kind)
)


def is_v3_kind(kind: type[Definition[Any]]) -> TypeIs[type[ZarrV3Kind]]:
    """Whether `kind` is a kind of v3 field: one of the kinds in `ZarrV3Kind`."""
    return kind in _KINDS


class ZarrV3ContextSlots(TypedDict, total=False, closed=True):
    """The definitions in each slot of a v3 scope, as `ZarrV3Context` and its `update` take them.

    This is where a v3 scope's slots are declared: each key is a slot,
    and its value the definitions of the kind that slot holds, as an
    iterable of them or as a mapping from each one's name to it.
    `ZarrV3Context.kinds` is read from these annotations, so a type
    checker checks every slot a caller names, and every definition given
    in it.
    """

    data_types: Iterable[DataTypeDefinition[Any]] | Mapping[str, DataTypeDefinition[Any]]
    """Data types, read from `data_type`."""
    chunk_grids: Iterable[ChunkGridDefinition[Any]] | Mapping[str, ChunkGridDefinition[Any]]
    """Chunk grids, read from `chunk_grid`."""
    chunk_key_encodings: (
        Iterable[ChunkKeyEncodingDefinition[Any]] | Mapping[str, ChunkKeyEncodingDefinition[Any]]
    )
    """Chunk key encodings, read from `chunk_key_encoding`."""
    codecs: Iterable[CodecDefinition[Any]] | Mapping[str, CodecDefinition[Any]]
    """Codecs, read from `codecs` and from the codecs a codec holds."""
    storage_transformers: (
        Iterable[StorageTransformerDefinition[Any]]
        | Mapping[str, StorageTransformerDefinition[Any]]
    )
    """Storage transformers, read from `storage_transformers`."""
    array_extra_fields: (
        Iterable[ArrayExtraFieldDefinition[Any]] | Mapping[str, ArrayExtraFieldDefinition[Any]]
    )
    """Members of an array document that the spec does not define, by key."""
    group_extra_fields: (
        Iterable[GroupExtraFieldDefinition[Any]] | Mapping[str, GroupExtraFieldDefinition[Any]]
    )
    """Members of a group document that the spec does not define, by key."""


@final
class ZarrV3Context:
    """A scope that reads Zarr v3 documents. Every v3 reader takes one.

    It implements `Context`, and has one slot for each kind of field a v3
    document holds, as `ZarrV3ContextSlots` declares them. The slots for
    the spec's extension points are named like the document member that
    holds such fields: `data_types`, `chunk_grids`, `chunk_key_encodings`,
    `codecs` and `storage_transformers`. `array_extra_fields` and
    `group_extra_fields` hold the definitions of members the spec does not
    define, by key. Each slot is a read-only mapping from the name each
    definition is filed under (`r*` for raw bits) to the definition. The
    constructor and `update` take, for each slot, an iterable of
    definitions or such a mapping, so one scope's slot can be given to
    another. A definition of another kind than its slot holds, and a key
    that is not its definition's name, raise `TypeError`, and two
    different definitions under one name in a slot raise
    `ScopeConflictError`. To replace a definition, or a whole slot, use
    `update`.
    `ZarrV3Context()` is the scope that claims nothing. Two scopes are
    equal when they file the same definitions in each slot, and equal
    scopes have equal hashes.
    """

    __slots__ = ("_tables",)

    kinds: ClassVar[Mapping[str, type[ZarrV3Kind]]] = slot_kinds(ZarrV3ContextSlots, is_v3_kind)
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

    def update(self, **slots: Unpack[ZarrV3ContextSlots]) -> ZarrV3Context:
        """This scope, with each slot given here holding only the definitions given for it. The other slots are kept.

        A slot can be given as a mapping from each definition's name to
        it, so a slot of this scope passes back in changed:
        `context.update(codecs={**context.codecs, "gzip": MY_GZIP})`
        replaces the gzip definition and keeps the other codecs, and
        `context.update(codecs=[*context.codecs.values(), MY_CODEC])` adds
        one, raising `ScopeConflictError` if the scope already files
        another definition under its name.
        """
        return self._of(updated(self._tables, self.kinds, slots, 3))

    def union(self, *others: ZarrV3Context) -> ZarrV3Context:
        """The smallest scope that files everything this scope and each of `others` file.

        Raises `ScopeConflictError` when two of them file different
        definitions under one name in a slot. To replace a definition on
        purpose, use `update`. Raises `TypeError` for a scope of another
        format.
        """
        refuse_others(others, ZarrV3Context, 3)
        return self._of(
            joined_tables(self.kinds, [self._tables, *(other._tables for other in others)])
        )

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


__all__ = ["ZarrV3Context", "ZarrV3ContextSlots", "ZarrV3Kind", "is_v3_kind"]
