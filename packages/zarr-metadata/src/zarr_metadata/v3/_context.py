"""The scope a Zarr v3 document is read in: one slot for each kind of v3 field."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

from zarr_metadata.definition._context import Context
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
    from collections.abc import Iterable, Mapping

    from zarr_metadata.definition._definition import Definition


class ZarrV3Context(Context):
    """A scope that reads Zarr v3 documents. Every v3 reader takes one.

    It has one slot for each kind of field a v3 document holds. The slots
    for the spec's extension points are named like the document member
    that holds such fields: `data_types`, `chunk_grids`,
    `chunk_key_encodings`, `codecs` and `storage_transformers`.
    `array_extra_fields` and `group_extra_fields` hold the definitions of
    members the spec does not define, by key. The constructor takes an iterable of
    definitions for each slot, and each slot is a read-only mapping from
    the name each definition is filed under (`r*` for raw bits) to the
    definition. A definition of another kind than its slot holds raises
    `TypeError`, and two different definitions under one name in a slot
    raise `ScopeConflictError`. To replace a definition, use
    `extended_with`. `ZarrV3Context()` is the scope that claims nothing.
    """

    __slots__ = ()

    kinds: ClassVar[Mapping[str, type[Definition[Any]]]] = MappingProxyType(
        {
            "data_types": DataTypeDefinition,
            "chunk_grids": ChunkGridDefinition,
            "chunk_key_encodings": ChunkKeyEncodingDefinition,
            "codecs": CodecDefinition,
            "storage_transformers": StorageTransformerDefinition,
            "array_extra_fields": ArrayExtraFieldDefinition,
            "group_extra_fields": GroupExtraFieldDefinition,
        }
    )

    def __init__(
        self,
        *,
        data_types: Iterable[DataTypeDefinition[Any]] = (),
        chunk_grids: Iterable[ChunkGridDefinition[Any]] = (),
        chunk_key_encodings: Iterable[ChunkKeyEncodingDefinition[Any]] = (),
        codecs: Iterable[CodecDefinition[Any]] = (),
        storage_transformers: Iterable[StorageTransformerDefinition[Any]] = (),
        array_extra_fields: Iterable[ArrayExtraFieldDefinition[Any]] = (),
        group_extra_fields: Iterable[GroupExtraFieldDefinition[Any]] = (),
    ) -> None:
        super().__init__(
            {
                "data_types": data_types,
                "chunk_grids": chunk_grids,
                "chunk_key_encodings": chunk_key_encodings,
                "codecs": codecs,
                "storage_transformers": storage_transformers,
                "array_extra_fields": array_extra_fields,
                "group_extra_fields": group_extra_fields,
            }
        )

    @property
    def data_types(self) -> Mapping[str, DataTypeDefinition[Any]]:
        """The data type definitions in scope, by the name each is filed under."""
        return cast("Mapping[str, DataTypeDefinition[Any]]", self._held(DataTypeDefinition))

    @property
    def chunk_grids(self) -> Mapping[str, ChunkGridDefinition[Any]]:
        """The chunk grid definitions in scope, by name."""
        return cast("Mapping[str, ChunkGridDefinition[Any]]", self._held(ChunkGridDefinition))

    @property
    def chunk_key_encodings(self) -> Mapping[str, ChunkKeyEncodingDefinition[Any]]:
        """The chunk key encoding definitions in scope, by name."""
        return cast(
            "Mapping[str, ChunkKeyEncodingDefinition[Any]]",
            self._held(ChunkKeyEncodingDefinition),
        )

    @property
    def codecs(self) -> Mapping[str, CodecDefinition[Any]]:
        """The codec definitions in scope, by name."""
        return cast("Mapping[str, CodecDefinition[Any]]", self._held(CodecDefinition))

    @property
    def storage_transformers(self) -> Mapping[str, StorageTransformerDefinition[Any]]:
        """The storage transformer definitions in scope, by name."""
        return cast(
            "Mapping[str, StorageTransformerDefinition[Any]]",
            self._held(StorageTransformerDefinition),
        )

    @property
    def array_extra_fields(self) -> Mapping[str, ArrayExtraFieldDefinition[Any]]:
        """The definitions of members of an array document that the spec does not define, by key."""
        return cast(
            "Mapping[str, ArrayExtraFieldDefinition[Any]]", self._held(ArrayExtraFieldDefinition)
        )

    @property
    def group_extra_fields(self) -> Mapping[str, GroupExtraFieldDefinition[Any]]:
        """The definitions of members of a group document that the spec does not define, by key."""
        return cast(
            "Mapping[str, GroupExtraFieldDefinition[Any]]", self._held(GroupExtraFieldDefinition)
        )

    def extended_with(
        self,
        *,
        data_types: Iterable[DataTypeDefinition[Any]] = (),
        chunk_grids: Iterable[ChunkGridDefinition[Any]] = (),
        chunk_key_encodings: Iterable[ChunkKeyEncodingDefinition[Any]] = (),
        codecs: Iterable[CodecDefinition[Any]] = (),
        storage_transformers: Iterable[StorageTransformerDefinition[Any]] = (),
        array_extra_fields: Iterable[ArrayExtraFieldDefinition[Any]] = (),
        group_extra_fields: Iterable[GroupExtraFieldDefinition[Any]] = (),
    ) -> ZarrV3Context:
        """This scope, plus definitions of your own, each in its slot.

        A definition given here replaces any definition already filed in
        its slot under the same name. This is how a reader substitutes its
        own reading of a codec the package already defines, or of raw bits
        (by defining `r*`).
        """
        return self._extended(
            {
                "data_types": data_types,
                "chunk_grids": chunk_grids,
                "chunk_key_encodings": chunk_key_encodings,
                "codecs": codecs,
                "storage_transformers": storage_transformers,
                "array_extra_fields": array_extra_fields,
                "group_extra_fields": group_extra_fields,
            }
        )


__all__ = ["ZarrV3Context"]
