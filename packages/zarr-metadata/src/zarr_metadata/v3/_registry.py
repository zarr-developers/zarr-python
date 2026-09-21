"""Which entities are in scope when raw JSON is coerced.

The only registry the package needs. A validator reading a document has
to decide, for each extension point, which identifier maps to which
entity — and that decision is the *scope* it validates against, not a
property of the entities themselves.

Two scopes, because the question "is this document valid?" has two useful
answers. `CORE` is what the Zarr v3 specification itself defines, so a
document validating against it uses nothing an implementation could
refuse for being optional. `CORE_AND_EXTENSIONS` adds what
`zarr-extensions` registers and this package models. A name in neither is
not rejected — extension openness — it is simply not judged.

Identifiers are the `name` the metadata carries, with one exception. A
family covers many names with one class -- every `r<N>` spelling is one
data-type family -- so it registers under an invented identifier that no
real name can collide with, and recognizes its own names through
`accepts`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, overload

from typing_extensions import TypedDict, Unpack

from zarr_metadata.model._validation import (
    ValidationProblem,
    validate_metadata_field_v3,
)
from zarr_metadata.v3._compile import is_class_var, own_annotations
from zarr_metadata.v3._entity import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    STORAGE_TRANSFORMERS,
    ChunkGridEntity,
    CodecEntity,
    DataTypeEntity,
    MetadataEntity,
    Opaque,
    named_configuration,
)
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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.v3._entity import ExtensionPointField, Loc


class EntityTables(TypedDict):
    """Which entity is registered under which name, at each extension point.

    Typed per point rather than as one mapping, because an entity's kind
    is a fact about where it may be registered: a codec at `data_type`
    would satisfy a `Mapping[str, type[MetadataEntity]]` and then fail
    the moment anything asked it for a storage class. Spelling the
    correspondence here is what makes `resolve`'s per-point return type
    true rather than asserted, and what turns a misfiling into an error
    where the table is written.
    """

    data_type: Mapping[str, type[DataTypeEntity]]
    codecs: Mapping[str, type[CodecEntity]]
    chunk_grid: Mapping[str, type[ChunkGridEntity]]
    chunk_key_encoding: Mapping[str, type[MetadataEntity]]
    storage_transformers: Mapping[str, type[MetadataEntity]]


class PartialEntityTables(TypedDict, total=False):
    """`EntityTables` with every point optional: what `extended_with` takes.

    A reader registering a codec of its own says so and nothing else;
    the points it does not name keep whatever the scope it extended had.
    """

    data_type: Mapping[str, type[DataTypeEntity]]
    codecs: Mapping[str, type[CodecEntity]]
    chunk_grid: Mapping[str, type[ChunkGridEntity]]
    chunk_key_encoding: Mapping[str, type[MetadataEntity]]
    storage_transformers: Mapping[str, type[MetadataEntity]]


_ENTITY_KINDS: Final[Mapping[ExtensionPointField, type[MetadataEntity]]] = {
    DATA_TYPE: DataTypeEntity,
    CODECS: CodecEntity,
    CHUNK_GRID: ChunkGridEntity,
    CHUNK_KEY_ENCODING: MetadataEntity,
    STORAGE_TRANSFORMERS: MetadataEntity,
}
"""The base every entity at a point must derive from.

`EntityTables` says the same thing to the type checker, which is where a
table written out in source is caught. This is for the one built at run
time -- from a plugin entry point, from configuration -- where there was
no type to check.
"""


@dataclass(frozen=True, slots=True)
class Context:
    """The entities in scope while metadata is being read.

    Passed to every `coerce`, and most entities ignore it: a `gzip` codec
    is a `gzip` codec whatever else is in scope. The ones that do not
    ignore it hold other entities inside their own configuration -- a
    `struct` data type holds field data types, a `sharding_indexed` codec
    holds two codec pipelines -- and cannot read those without knowing
    what is in scope inside them.

    A scope is not a property of the entities, it is a choice the reader
    makes: judging against the specification alone, or against the
    specification plus what `zarr-extensions` registers.
    """

    entities: EntityTables

    def __post_init__(self) -> None:
        """Refuse a table an entity does not belong in.

        Two ways it can fail to. The entity may be of the wrong kind for
        the point -- a codec under `data_type` -- which `EntityTables`
        catches in source and this catches in a scope assembled at run
        time. Or its key may not be its identifier.

        `resolve` finds a candidate by key and then asks the entity
        whether the name is really one of its own, so a key that is not
        the entity's `identifier` can never resolve. If the two disagree
        -- a typo, or a rename that missed one of the two places the name
        is written -- registration appears to succeed, validation runs,
        and the verdict is clean. Indistinguishable from extension
        openness, and the easiest way to ship a broken extension.

        The key is the identifier, not a name a document writes: the
        raw-bytes family registers under an invented one that `accepts`
        deliberately refuses.
        """
        for field, table in self.tables().items():
            for key, entity in table.items():
                if not issubclass(entity, _ENTITY_KINDS[field]):
                    point = entity.extension_point
                    msg = (
                        f"{entity.__name__} is registered at {field!r}, which takes "
                        f"{_ENTITY_KINDS[field].__name__} entities"
                        + (f"; register it at {point!r}" if point is not None else "")
                    )
                    raise TypeError(msg)
                if key != entity.identifier:
                    msg = (
                        f"{entity.__name__} is registered at {field!r} under {key!r} "
                        f"but its identifier is {entity.identifier!r}; key the table by "
                        f"{entity.__name__}.identifier"
                    )
                    raise ValueError(msg)
                if inspect.isabstract(entity):
                    # What a kind leaves abstract -- `transition`, `grid`,
                    # `fill_value_problems` -- the entity answers, or it
                    # is not one this scope can use.
                    left = ", ".join(sorted(entity.__abstractmethods__))
                    msg = (
                        f"{entity.__name__} does not define {left}, which its base leaves "
                        "abstract; define it, if only to return the same thing as `incoming` or ()"
                    )
                    raise TypeError(msg)
                if "__dataclass_fields__" not in vars(entity) and any(
                    not is_class_var(annotation) for annotation in own_annotations(entity).values()
                ):
                    # Class creation runs before `@dataclass` and cannot
                    # see whether it was applied; this is the next place
                    # the entity passes through before `coerce` builds it.
                    msg = (
                        f"{entity.__name__} declares fields but is not a dataclass; decorate "
                        "it with @dataclass(frozen=True), which is what `coerce` builds it with"
                    )
                    raise TypeError(msg)

    def extended_with(self, **entities: Unpack[PartialEntityTables]) -> Context:
        """This scope, plus entities of your own at the points named.

        The merge is per point, so naming `codecs` adds codecs rather
        than replacing the ones already in scope. A name already
        registered is taken over by what is passed here, which is how a
        reader substitutes its own reading of a codec the package
        already models.
        """
        return Context(
            {
                DATA_TYPE: {**self.entities["data_type"], **entities.get("data_type", {})},
                CODECS: {**self.entities["codecs"], **entities.get("codecs", {})},
                CHUNK_GRID: {**self.entities["chunk_grid"], **entities.get("chunk_grid", {})},
                CHUNK_KEY_ENCODING: {
                    **self.entities["chunk_key_encoding"],
                    **entities.get("chunk_key_encoding", {}),
                },
                STORAGE_TRANSFORMERS: {
                    **self.entities["storage_transformers"],
                    **entities.get("storage_transformers", {}),
                },
            }
        )

    def tables(self) -> Mapping[ExtensionPointField, Mapping[str, type[MetadataEntity]]]:
        """Every point's table, under the one kind all entities share.

        `entities` gives each point its own entity kind, which is the
        point of it, and a key that is not a literal loses that. So the
        widening is written out here, once, by hand rather than asserted
        with a `cast`: reading each member by its own key is what makes
        the result checked rather than promised. Anything that asks the
        scope what is in it, rather than asking it about one point, wants
        this.
        """
        return {
            DATA_TYPE: self.entities["data_type"],
            CODECS: self.entities["codecs"],
            CHUNK_GRID: self.entities["chunk_grid"],
            CHUNK_KEY_ENCODING: self.entities["chunk_key_encoding"],
            STORAGE_TRANSFORMERS: self.entities["storage_transformers"],
        }

    @overload
    def resolve(self, field: Literal["data_type"], name: str) -> type[DataTypeEntity] | None: ...

    @overload
    def resolve(self, field: Literal["codecs"], name: str) -> type[CodecEntity] | None: ...

    @overload
    def resolve(self, field: Literal["chunk_grid"], name: str) -> type[ChunkGridEntity] | None: ...

    @overload
    def resolve(self, field: ExtensionPointField, name: str) -> type[MetadataEntity] | None: ...

    def resolve(self, field: ExtensionPointField, name: str) -> type[MetadataEntity] | None:
        """The entity `name` denotes at `field`, or None if out of scope.

        Out of scope is not an error: an unknown name may be an extension
        this reader does not model, and openness means leaving it unjudged.

        The entity has the last word, via `accepts`. A name that is a key
        still has to be claimed, because a family's key is an invented
        identifier that no document may write; and a name that is not a
        key may still belong to a family, which is what the scan is for.
        A third party registers one the same way, with no table of
        spellings anywhere in this package.
        """
        table = self.tables()[field]
        entity = table.get(name)
        if entity is not None:
            return entity if entity.accepts(name) else None
        # A family covers many names with one class, so its entry cannot
        # be keyed by all of them; it is keyed by an invented identifier
        # and recognizes its own. Asked only when the name is not a key,
        # so the common case stays a lookup. First match wins, and two
        # entities claiming one name is a scope that contradicts itself.
        return next((candidate for candidate in table.values() if candidate.accepts(name)), None)

    @overload
    def coerce(
        self,
        field: Literal["data_type"],
        value: object,
        loc: Loc = (),
        *,
        envelope_judged: bool = False,
    ) -> tuple[DataTypeEntity | Opaque, tuple[ValidationProblem, ...]]: ...

    @overload
    def coerce(
        self,
        field: Literal["codecs"],
        value: object,
        loc: Loc = (),
        *,
        envelope_judged: bool = False,
    ) -> tuple[CodecEntity | Opaque, tuple[ValidationProblem, ...]]: ...

    @overload
    def coerce(
        self,
        field: Literal["chunk_grid"],
        value: object,
        loc: Loc = (),
        *,
        envelope_judged: bool = False,
    ) -> tuple[ChunkGridEntity | Opaque, tuple[ValidationProblem, ...]]: ...

    @overload
    def coerce(
        self,
        field: ExtensionPointField,
        value: object,
        loc: Loc = (),
        *,
        envelope_judged: bool = False,
    ) -> tuple[MetadataEntity | Opaque, tuple[ValidationProblem, ...]]: ...

    def coerce(
        self,
        field: ExtensionPointField,
        value: object,
        loc: Loc = (),
        *,
        envelope_judged: bool = False,
    ) -> tuple[MetadataEntity | Opaque, tuple[ValidationProblem, ...]]:
        """One nested entity, read in this scope.

        The primitive the containing entities are built from: a `struct`
        data type reads its fields with it, a `sharding_indexed` codec its
        two pipelines. Returns the entity when its name is in scope, and
        the value untouched when it is not -- an unmodelled extension is
        left unjudged, which is what makes the format open.

        `loc` prefixes the problems, so they point at where in the
        containing configuration the entity sat.

        A metadata field is a metadata field wherever it appears, so the
        envelope gets the same structural judgment here that the model
        layer gives a top-level one -- an extra member, a `configuration`
        that is not an object, a `must_understand` that is not a boolean
        or is `false`. That last one is why the flag is passed: an
        extension point is something a reader must understand at every
        depth, not only at the document's top level.
        `envelope_judged` says that judgment has already happened, which
        it has for the fields of a document the model layer accepted.
        """
        problems: list[ValidationProblem] = []
        if not envelope_judged:
            problems.extend(
                ValidationProblem((*loc, *found.loc), found.message, found.kind)
                for found in validate_metadata_field_v3(value, allow_must_understand_false=False)
            )
        name, _, _ = named_configuration(value)
        if name is None:
            return Opaque(value, "invalid"), (
                *problems,
                ValidationProblem(loc, f"expected a metadata field, got {value!r}", "invalid_type"),
            )
        entity_type = self.resolve(field, name)
        if entity_type is None:
            return Opaque(value, "out_of_scope"), tuple(problems)
        entity, found = entity_type.coerce(value, self)
        problems.extend(
            ValidationProblem((*loc, *entry.loc), entry.message, entry.kind) for entry in found
        )
        if entity is None:
            return Opaque(value, "invalid"), tuple(problems)
        return entity, tuple(problems)


_CORE_CODECS: Final[dict[str, type[CodecEntity]]] = {
    BloscCodec.identifier: BloscCodec,
    BytesCodec.identifier: BytesCodec,
    Crc32cCodec.identifier: Crc32cCodec,
    GzipCodec.identifier: GzipCodec,
    ShardingIndexedCodec.identifier: ShardingIndexedCodec,
    TransposeCodec.identifier: TransposeCodec,
}
_EXTENSION_CODECS: Final[dict[str, type[CodecEntity]]] = {
    CastValueCodec.identifier: CastValueCodec,
    ScaleOffsetCodec.identifier: ScaleOffsetCodec,
    ZstdCodec.identifier: ZstdCodec,
}

_CORE_DATA_TYPES: Final[dict[str, type[DataTypeEntity]]] = {
    BoolDataType.identifier: BoolDataType,
    Int8DataType.identifier: Int8DataType,
    Int16DataType.identifier: Int16DataType,
    Int32DataType.identifier: Int32DataType,
    Int64DataType.identifier: Int64DataType,
    Uint8DataType.identifier: Uint8DataType,
    Uint16DataType.identifier: Uint16DataType,
    Uint32DataType.identifier: Uint32DataType,
    Uint64DataType.identifier: Uint64DataType,
    Float16DataType.identifier: Float16DataType,
    Float32DataType.identifier: Float32DataType,
    Float64DataType.identifier: Float64DataType,
    Complex64DataType.identifier: Complex64DataType,
    Complex128DataType.identifier: Complex128DataType,
    RawBytesDataType.identifier: RawBytesDataType,
}
_EXTENSION_DATA_TYPES: Final[dict[str, type[DataTypeEntity]]] = {
    BytesDataType.identifier: BytesDataType,
    StringDataType.identifier: StringDataType,
    NumpyDatetime64DataType.identifier: NumpyDatetime64DataType,
    NumpyTimedelta64DataType.identifier: NumpyTimedelta64DataType,
    StructDataType.identifier: StructDataType,
}

_CORE_CHUNK_GRIDS: Final[dict[str, type[ChunkGridEntity]]] = {
    RegularChunkGrid.identifier: RegularChunkGrid,
}
_EXTENSION_CHUNK_GRIDS: Final[dict[str, type[ChunkGridEntity]]] = {
    RectilinearChunkGrid.identifier: RectilinearChunkGrid,
}

_CORE_CHUNK_KEY_ENCODINGS: Final[dict[str, type[MetadataEntity]]] = {
    DefaultChunkKeyEncoding.identifier: DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding.identifier: V2ChunkKeyEncoding,
}


CORE: Final = Context(
    {
        CODECS: _CORE_CODECS,
        DATA_TYPE: _CORE_DATA_TYPES,
        CHUNK_GRID: _CORE_CHUNK_GRIDS,
        CHUNK_KEY_ENCODING: _CORE_CHUNK_KEY_ENCODINGS,
        STORAGE_TRANSFORMERS: {},
    }
)
"""Only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final = Context(
    {
        CODECS: {**_CORE_CODECS, **_EXTENSION_CODECS},
        DATA_TYPE: {**_CORE_DATA_TYPES, **_EXTENSION_DATA_TYPES},
        CHUNK_GRID: {**_CORE_CHUNK_GRIDS, **_EXTENSION_CHUNK_GRIDS},
        CHUNK_KEY_ENCODING: {**_CORE_CHUNK_KEY_ENCODINGS},
        STORAGE_TRANSFORMERS: {},
    }
)
"""What the specification defines, plus what `zarr-extensions` registers."""


__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "Context",
    "EntityTables",
    "PartialEntityTables",
]
