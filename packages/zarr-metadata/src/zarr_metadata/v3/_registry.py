"""Which definitions are in scope while a document is read.

The only registry the package needs. Reading a field means deciding which
definition its name denotes, and that decision is the *scope* the field
is read in, not a property of the definitions. A scope is a value, built
from definitions: each is filed under its kind, which is its type, and
its name, which it carries.

Two scopes, because "is this valid?" has two useful answers. `CORE` is
what the Zarr v3 specification itself defines, so a document read in it
uses nothing an implementation could refuse for being optional.
`CORE_AND_EXTENSIONS` adds what `zarr-extensions` registers and this
package defines. A name in neither is not refused -- that is what keeps
the format open -- it is read as `out_of_scope` and left unjudged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

from zarr_metadata.v3._definition import KINDS, Definition, as_kind, kind_of
from zarr_metadata.v3.chunk_grid.rectilinear import RECTILINEAR_CHUNK_GRID
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID
from zarr_metadata.v3.chunk_key_encoding.default import DEFAULT_CHUNK_KEY_ENCODING
from zarr_metadata.v3.chunk_key_encoding.v2 import V2_CHUNK_KEY_ENCODING
from zarr_metadata.v3.codec.blosc import BLOSC_CODEC
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.cast_value import CAST_VALUE_CODEC
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.codec.scale_offset import SCALE_OFFSET_CODEC
from zarr_metadata.v3.codec.sharding_indexed import SHARDING_INDEXED_CODEC
from zarr_metadata.v3.codec.transpose import TRANSPOSE_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE

if TYPE_CHECKING:
    from zarr_metadata.v3._definition import D

Tables = Mapping[type[Definition[Any]], Mapping[str, Definition[Any]]]
"""By kind, then by the name each definition is filed under."""


@dataclass(frozen=True, slots=True)
class Context:
    """The definitions in scope while metadata is read.

    A value with no reading of its own: `resolve` reads a field in it,
    and `claimant` is the one question it answers, which definition a
    name belongs to. Built from definitions with `Context.of`, extended
    with more by `extended_with`.
    """

    tables: Tables

    @classmethod
    def of(cls, *definitions: Definition[Any]) -> Context:
        """A scope of exactly these definitions; a later one takes a name over from an earlier.

        `TypeError` for a definition of no kind, which no position in a
        document could hold.
        """
        tables: dict[type[Definition[Any]], dict[str, Definition[Any]]] = {
            kind: {} for kind in KINDS
        }
        for definition in definitions:
            kind = kind_of(definition)
            if kind is None:
                msg = (
                    f"{definition.name!r} is a definition of no kind; build it as a "
                    "CodecDefinition, DataTypeDefinition, ChunkGridDefinition, "
                    "ChunkKeyEncodingDefinition or StorageTransformerDefinition"
                )
                raise TypeError(msg)
            tables[kind][definition.name] = definition
        return cls(
            MappingProxyType({kind: MappingProxyType(table) for kind, table in tables.items()})
        )

    def extended_with(self, *definitions: Definition[Any]) -> Context:
        """This scope, plus definitions of your own.

        A name already filed under the same kind, or claimed by a family,
        is taken over by what is passed here, which is how a reader
        substitutes its own reading of a codec the package already
        defines.
        """
        return Context.of(*self.definitions(), *definitions)

    def definitions(self) -> tuple[Definition[Any], ...]:
        """Every definition in scope, kind by kind."""
        return tuple(entry for table in self.tables.values() for entry in table.values())

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` in scope that claims `name`; None if none does.

        Asks each definition filed under the kind whether the name is its
        own -- a family claims every `r<N>` -- the one filed last first, so
        a definition a scope was extended with takes a name over from one
        before it, whether that one was filed under the name or claims it
        as a family.
        """
        table = self.tables.get(as_kind(kind), {})
        return cast(
            "D | None",
            next((entry for entry in reversed(tuple(table.values())) if entry.claims(name)), None),
        )


_CORE: Final[tuple[Definition[Any], ...]] = (
    BLOSC_CODEC,
    BYTES_CODEC,
    CRC32C_CODEC,
    GZIP_CODEC,
    SHARDING_INDEXED_CODEC,
    TRANSPOSE_CODEC,
    BOOL_DATA_TYPE,
    INT8_DATA_TYPE,
    INT16_DATA_TYPE,
    INT32_DATA_TYPE,
    INT64_DATA_TYPE,
    UINT8_DATA_TYPE,
    UINT16_DATA_TYPE,
    UINT32_DATA_TYPE,
    UINT64_DATA_TYPE,
    FLOAT16_DATA_TYPE,
    FLOAT32_DATA_TYPE,
    FLOAT64_DATA_TYPE,
    COMPLEX64_DATA_TYPE,
    COMPLEX128_DATA_TYPE,
    RAW_BYTES_DATA_TYPE,
    REGULAR_CHUNK_GRID,
    DEFAULT_CHUNK_KEY_ENCODING,
    V2_CHUNK_KEY_ENCODING,
)
"""What the Zarr v3 specification itself defines."""

_EXTENSIONS: Final[tuple[Definition[Any], ...]] = (
    CAST_VALUE_CODEC,
    SCALE_OFFSET_CODEC,
    ZSTD_CODEC,
    BYTES_DATA_TYPE,
    STRING_DATA_TYPE,
    NUMPY_DATETIME64_DATA_TYPE,
    NUMPY_TIMEDELTA64_DATA_TYPE,
    STRUCT_DATA_TYPE,
    RECTILINEAR_CHUNK_GRID,
)
"""What `zarr-extensions` registers and this package defines."""

CORE: Final = Context.of(*_CORE)
"""Only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final = Context.of(*_CORE, *_EXTENSIONS)
"""What the specification defines, plus what `zarr-extensions` registers."""

__all__ = ["CORE", "CORE_AND_EXTENSIONS", "Context", "Tables"]
