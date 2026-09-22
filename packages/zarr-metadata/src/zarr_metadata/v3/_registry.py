"""Which entities are in scope when raw JSON is coerced.

The only registry the package needs. A validator reading a document has
to decide, for each kind of entity, which identifier maps to which class
— and that decision is the *scope* it validates against, not a property
of the entities themselves. Which of a document's fields holds which
kind is the document's own knowledge, in `_document`.

Two scopes, because the question "is this document valid?" has two useful
answers. `CORE` is what the Zarr v3 specification itself defines, so a
document validating against it uses nothing an implementation could
refuse for being optional. `CORE_AND_EXTENSIONS` adds what
`zarr-extensions` registers and this package models. A name in neither is
not rejected — extension openness — it is simply not judged.

An entity is registered under its `identifier`, which is the `name` the
metadata carries -- except for a family, which covers many names with
one class (every `r<N>` spelling is one data-type family) and registers
under an invented one. Resolution asks each entity of a kind whether a
name is its own, through `accepts`; the identifier is the key that
`extended_with` takes a name over with.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from zarr_metadata.v3._entity import (
    KINDS,
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
    CodecEntity,
    EntityT,
    MetadataEntity,
    kind_of,
    unreadable,
)
from zarr_metadata.v3._typed_json import is_class_var, own_annotations
from zarr_metadata.v3.chunk_grid.rectilinear import RectilinearChunkGrid
from zarr_metadata.v3.chunk_grid.regular import RegularChunkGrid
from zarr_metadata.v3.chunk_key_encoding.default import DefaultChunkKeyEncoding
from zarr_metadata.v3.chunk_key_encoding.v2 import V2ChunkKeyEncoding
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.codec.bytes import BytesCodec
from zarr_metadata.v3.codec.cast_value import CastValueCodec
from zarr_metadata.v3.codec.crc32c import Crc32cCodec
from zarr_metadata.v3.codec.gzip import GzipCodec
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodec
from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexedCodec
from zarr_metadata.v3.codec.transpose import TransposeCodec
from zarr_metadata.v3.codec.zstd import ZstdCodec
from zarr_metadata.v3.data_type.bool import BoolDataType
from zarr_metadata.v3.data_type.bytes import BytesDataType
from zarr_metadata.v3.data_type.complex64 import Complex64DataType
from zarr_metadata.v3.data_type.complex128 import Complex128DataType
from zarr_metadata.v3.data_type.float16 import Float16DataType
from zarr_metadata.v3.data_type.float32 import Float32DataType
from zarr_metadata.v3.data_type.float64 import Float64DataType
from zarr_metadata.v3.data_type.int8 import Int8DataType
from zarr_metadata.v3.data_type.int16 import Int16DataType
from zarr_metadata.v3.data_type.int32 import Int32DataType
from zarr_metadata.v3.data_type.int64 import Int64DataType
from zarr_metadata.v3.data_type.numpy_datetime64 import NumpyDatetime64DataType
from zarr_metadata.v3.data_type.numpy_timedelta64 import NumpyTimedelta64DataType
from zarr_metadata.v3.data_type.raw import RawBytesDataType
from zarr_metadata.v3.data_type.string import StringDataType
from zarr_metadata.v3.data_type.struct import StructDataType
from zarr_metadata.v3.data_type.uint8 import Uint8DataType
from zarr_metadata.v3.data_type.uint16 import Uint16DataType
from zarr_metadata.v3.data_type.uint32 import Uint32DataType
from zarr_metadata.v3.data_type.uint64 import Uint64DataType

Tables = Mapping[type[MetadataEntity], Mapping[str, type[MetadataEntity]]]
"""By kind, then by the identifier each entity is registered under."""


@dataclass(frozen=True, slots=True)
class Context:
    """The entities in scope while metadata is being read.

    A value, with no reading of its own: `resolve` reads a field in it,
    and `claimant` is the one question it answers, which class a name
    belongs to. Built from classes with `Context.of`; extended with more
    by `extended_with`. What each class is registered as is read off it
    -- its kind is its base class, its key is its `identifier` -- so
    there is nothing to misfile.
    """

    tables: Tables

    @classmethod
    def of(cls, *entities: type[MetadataEntity]) -> Context:
        """A scope of exactly these entities; a later one takes over an identifier from an earlier."""
        tables: dict[type[MetadataEntity], dict[str, type[MetadataEntity]]] = {
            kind: {} for kind in KINDS
        }
        for entity in entities:
            kind = _registrable(entity)
            tables[kind][entity.identifier] = entity
        return cls(
            MappingProxyType({kind: MappingProxyType(table) for kind, table in tables.items()})
        )

    def extended_with(self, *entities: type[MetadataEntity]) -> Context:
        """This scope, plus entities of your own.

        A name already registered under the same kind is taken over by
        what is passed here, which is how a reader substitutes its own
        reading of a codec the package already models.
        """
        return Context.of(*self.entities(), *entities)

    def entities(self) -> tuple[type[MetadataEntity], ...]:
        """Every entity in scope, kind by kind."""
        return tuple(entity for table in self.tables.values() for entity in table.values())

    def claimant(self, kind: type[EntityT], name: str) -> type[EntityT] | None:
        """The class in scope that claims `name` as an entity of `kind`; None if none does.

        Asks each class registered under the kind's kind whether the name
        is its own -- a family claims every `r<N>` -- rather than looking
        a key up, so the identifier keys exist for `extended_with` to
        take a name over, not for lookup. A class that claims the name
        but is not a `kind` -- `transpose` asked for as a
        `BytesBytesCodec` -- is none; `resolve` asks with the kind's kind
        to tell that case from a name nothing claims.
        """
        registered = kind_of(kind)
        if registered is None:
            return None
        table = self.tables.get(registered, {})
        entity = next((candidate for candidate in table.values() if candidate.accepts(name)), None)
        if entity is None or not issubclass(entity, kind):
            return None
        return entity


def _registrable(entity: type[MetadataEntity]) -> type[MetadataEntity]:
    """The kind `entity` is registered under; `TypeError` for a class no scope can use.

    The one moment an entity is refused: every way of writing one that
    type-checks cleanly and then fails somewhere that will not name the
    class -- no kind, a codec skipping the kind classes, no `@dataclass`,
    a field `coerce` could not parse, a `__post_init__` `coerce` would
    never ask, a class variable owed and unset, what a kind leaves
    abstract -- with a message that says what to write.
    """
    kind = kind_of(entity)
    if kind is None:
        msg = (
            f"{entity.__name__} is of no kind; subclass a codec kind, DataTypeEntity, "
            "ChunkGridEntity, ChunkKeyEncodingEntity or StorageTransformerEntity"
        )
        raise TypeError(msg)
    if issubclass(entity, CodecEntity) and not issubclass(
        entity, (ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec)
    ):
        # The kind classes say what a codec does to the array, and what
        # each must answer is abstract on them.
        msg = (
            f"{entity.__name__} subclasses CodecEntity directly; subclass ArrayArrayCodec, "
            "ArrayBytesCodec or BytesBytesCodec, which says what the codec does to the array"
        )
        raise TypeError(msg)
    if "__dataclass_fields__" not in vars(entity) and any(
        not is_class_var(annotation) for annotation in own_annotations(entity).values()
    ):
        msg = (
            f"{entity.__name__} declares fields but is not a dataclass; decorate it with "
            "@dataclass(frozen=True), which is what makes its fields the configuration"
        )
        raise TypeError(msg)
    refused = unreadable(entity)
    if refused is not None:
        raise TypeError(refused)
    if inspect.isabstract(entity):
        left = ", ".join(sorted(entity.__abstractmethods__))
        msg = f"{entity.__name__} does not define {left}, which its base leaves abstract"
        raise TypeError(msg)
    return kind


_CORE: Final[tuple[type[MetadataEntity], ...]] = (
    BloscCodec,
    BytesCodec,
    Crc32cCodec,
    GzipCodec,
    ShardingIndexedCodec,
    TransposeCodec,
    BoolDataType,
    Int8DataType,
    Int16DataType,
    Int32DataType,
    Int64DataType,
    Uint8DataType,
    Uint16DataType,
    Uint32DataType,
    Uint64DataType,
    Float16DataType,
    Float32DataType,
    Float64DataType,
    Complex64DataType,
    Complex128DataType,
    RawBytesDataType,
    RegularChunkGrid,
    DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding,
)
"""What the Zarr v3 specification itself defines."""

_EXTENSIONS: Final[tuple[type[MetadataEntity], ...]] = (
    CastValueCodec,
    ScaleOffsetCodec,
    ZstdCodec,
    BytesDataType,
    StringDataType,
    NumpyDatetime64DataType,
    NumpyTimedelta64DataType,
    StructDataType,
    RectilinearChunkGrid,
)
"""What `zarr-extensions` registers and this package models."""

CORE: Final = Context.of(*_CORE)
"""Only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final = Context.of(*_CORE, *_EXTENSIONS)
"""What the specification defines, plus what `zarr-extensions` registers."""
