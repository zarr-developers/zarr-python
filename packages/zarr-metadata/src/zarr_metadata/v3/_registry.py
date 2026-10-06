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
the format open -- it is read as `Unclaimed` and left unjudged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

from zarr_metadata.v3._definition import KINDS, Definition, as_kind, kind_of, spelled
from zarr_metadata.v3._scope import Conflict, ScopeConflictError, disagreements_of
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
    from collections.abc import Callable

    from zarr_metadata.v3._definition import D
    from zarr_metadata.v3._scope import Claims, Disagreements

Tables = Mapping[type[Definition[Any]], Mapping[str, Definition[Any]]]
"""By kind, then by the name each definition is filed under."""


@dataclass(frozen=True, slots=True, eq=False)
class Context:
    """The definitions in scope while metadata is read.

    A value with no reading of its own: `resolve` reads a field in it,
    and `claimant` is the one question it answers, which definition a
    name belongs to. Built from definitions with `Context.of`, extended
    with more by `extended_with`; two scopes are equal when they file the
    same definitions, and equal scopes hash alike.
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

        A name already filed under the same kind is taken over by what is
        passed here, which is how a reader substitutes its own reading of a
        codec the package already defines -- or of raw bits, by defining
        `r*`.
        """
        return Context.of(*self.definitions(), *definitions)

    def definitions(self) -> tuple[Definition[Any], ...]:
        """Every definition in scope, kind by kind."""
        return tuple(entry for table in self.tables.values() for entry in table.values())

    def __repr__(self) -> str:
        # Short, as a default argument shows it: in full, a scope's repr is
        # every definition's, and `help` of a validator runs to pages.
        return f"Context(<{len(self.definitions())} definitions>)"

    def __reduce__(self) -> tuple[Callable[..., Context], tuple[Definition[Any], ...]]:
        # A scope is its definitions, so it pickles as them, and goes to
        # another process with the documents it is to read there.
        return (Context.of, self.definitions())

    def __copy__(self) -> Context:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Context:
        # A scope never changes, so a copy of it is itself.
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Context):
            return NotImplemented
        return self._filed() == other._filed()

    def __hash__(self) -> int:
        return hash(self._filed())

    def _filed(self) -> frozenset[tuple[type[Definition[Any]], str, Definition[Any]]]:
        """Every definition in scope with the kind and name it is filed under: what two scopes are compared by."""
        return frozenset(
            (kind, name, definition)
            for kind, table in self.tables.items()
            for name, definition in table.items()
        )

    def disagreements(self, claims: Claims) -> Disagreements:
        """Where this scope reads `claims`, a reading's, otherwise: what it would gain, and what it conflicts with, as `Disagreements` says.

        A claim is keyed by the name its definition is filed under -- raw
        bits under `r*` -- so it is looked up as filed, not as a document
        writes it.
        """
        return disagreements_of(lambda kind, name: self.tables.get(kind, {}).get(name), claims)

    @classmethod
    def joined(cls, *contexts: Context) -> Context:
        """The least scope that files everything each of `contexts` files: their join.

        `ScopeConflictError` when two of them file different definitions
        under one name of one kind; `extended_with` is for taking a name
        over on purpose.
        """
        filed: dict[tuple[type[Definition[Any]], str], Definition[Any]] = {}
        conflicts: list[Conflict] = []
        for context in contexts:
            for kind, table in context.tables.items():
                for name, definition in table.items():
                    held = filed.get((kind, name))
                    if held is not None and held != definition:
                        conflicts.append(Conflict((kind, name), held, definition))
                        continue
                    filed[kind, name] = definition
        if len(conflicts) != 0:
            raise ScopeConflictError(conflicts)
        return cls.of(*filed.values())

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` in scope that reads `name`, a name a document writes; None if none does.

        The one filed under the name `spelled` reads it as: itself, but
        for raw bits, `r16` read by the definition of `r*`.
        """
        asked = as_kind(kind)
        filed, _ = spelled(asked, name)
        if filed is None:
            return None
        return cast("D | None", self.tables.get(asked, {}).get(filed))


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
