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

Identifiers are the `name` the metadata carries, with one exception. Every
`r<N>` spelling is one data-type family, so the family registers under an
invented identifier that no real name can collide with; `canonical_name`
folds a spelling onto it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from zarr_metadata.model._validation import (
    ValidationProblem,
    validate_metadata_field_v3,
)
from zarr_metadata.v3._entity import named_configuration
from zarr_metadata.v3._extension_points import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    canonical_name,
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

    from zarr_metadata.v3._entity import Loc, MetadataEntity
    from zarr_metadata.v3._extension_points import ExtensionPointField


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

    entities: Mapping[ExtensionPointField, Mapping[str, type[MetadataEntity]]]

    def resolve(self, field: ExtensionPointField, name: str) -> type[MetadataEntity] | None:
        """The entity `name` denotes at `field`, or None if out of scope.

        Out of scope is not an error: an unknown name may be an extension
        this reader does not model, and openness means leaving it unjudged.

        The entity has the last word, via `accepts`. Folding is what finds
        a candidate -- every `r<N>` spelling is tabled under one invented
        identifier -- and the candidate is what says whether the name is
        really one of its own. Otherwise the identifier itself would be a
        name a document could write.
        """
        entity = self.entities.get(field, {}).get(canonical_name(field, name))
        if entity is None or not entity.accepts(name):
            return None
        return entity

    def coerce(
        self,
        field: ExtensionPointField,
        value: object,
        loc: Loc = (),
        *,
        envelope_judged: bool = False,
    ) -> tuple[MetadataEntity | object, tuple[ValidationProblem, ...]]:
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
        that is not an object, a `must_understand` that is not a boolean.
        `envelope_judged` says that judgment has already happened, which
        it has for the fields of a document the model layer accepted.
        """
        problems: list[ValidationProblem] = []
        if not envelope_judged:
            problems.extend(
                ValidationProblem((*loc, *found.loc), found.message, found.kind)
                for found in validate_metadata_field_v3(value)
            )
        name, _, _ = named_configuration(value)
        if name is None:
            return value, (
                *problems,
                ValidationProblem(loc, f"expected a metadata field, got {value!r}", "invalid_type"),
            )
        entity_type = self.resolve(field, name)
        if entity_type is None:
            return value, tuple(problems)
        entity, found = entity_type.coerce(value, self)
        problems.extend(
            ValidationProblem((*loc, *entry.loc), entry.message, entry.kind) for entry in found
        )
        return (value if entity is None else entity), tuple(problems)


_CORE_CODECS: Final[dict[str, type[MetadataEntity]]] = {
    BloscCodec.identifier: BloscCodec,
    BytesCodec.identifier: BytesCodec,
    Crc32cCodec.identifier: Crc32cCodec,
    GzipCodec.identifier: GzipCodec,
    ShardingIndexedCodec.identifier: ShardingIndexedCodec,
    TransposeCodec.identifier: TransposeCodec,
}
_EXTENSION_CODECS: Final[dict[str, type[MetadataEntity]]] = {
    CastValueCodec.identifier: CastValueCodec,
    ScaleOffsetCodec.identifier: ScaleOffsetCodec,
    ZstdCodec.identifier: ZstdCodec,
}

_CORE_DATA_TYPES: Final[dict[str, type[MetadataEntity]]] = {
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
_EXTENSION_DATA_TYPES: Final[dict[str, type[MetadataEntity]]] = {
    BytesDataType.identifier: BytesDataType,
    StringDataType.identifier: StringDataType,
    NumpyDatetime64DataType.identifier: NumpyDatetime64DataType,
    NumpyTimedelta64DataType.identifier: NumpyTimedelta64DataType,
    StructDataType.identifier: StructDataType,
}

_CORE_CHUNK_GRIDS: Final[dict[str, type[MetadataEntity]]] = {
    RegularChunkGrid.identifier: RegularChunkGrid,
}
_EXTENSION_CHUNK_GRIDS: Final[dict[str, type[MetadataEntity]]] = {
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
    }
)
"""Only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final = Context(
    {
        CODECS: {**_CORE_CODECS, **_EXTENSION_CODECS},
        DATA_TYPE: {**_CORE_DATA_TYPES, **_EXTENSION_DATA_TYPES},
        CHUNK_GRID: {**_CORE_CHUNK_GRIDS, **_EXTENSION_CHUNK_GRIDS},
        CHUNK_KEY_ENCODING: {**_CORE_CHUNK_KEY_ENCODINGS},
    }
)
"""What the specification defines, plus what `zarr-extensions` registers."""


__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "Context",
]
