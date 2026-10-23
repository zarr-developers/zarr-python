"""Which definitions are in scope while a document is read.

This is the only registry the package needs. Reading a field means
deciding which definition its name refers to. That decision depends on
the *scope* the field is read in, not on the definitions themselves. A
scope is a value built from definitions. Each definition is filed under
its kind (its type) and its name (an attribute of the definition).

The package provides two scopes, because "is this valid?" has two useful
answers. `CORE` contains what the Zarr v3 specification itself defines,
so a document read in it uses nothing that an implementation could reject
as optional. `CORE_AND_EXTENSIONS` adds the extensions that
`zarr-extensions` registers and this package defines. A name in neither
scope is not rejected, because that is what keeps the format open.
Instead it is read as an `UnclaimedField` and not checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, TypeAlias, TypeGuard, cast

from typing_extensions import TypeVar

from zarr_metadata.v3._definition import Definition, as_kind, filed_under, format_of, kind_of
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
"""Definitions keyed by kind, then by the name each definition is filed under."""

F = TypeVar("F", bound=Literal[2, 3], default=Any)
"""The Zarr format of the documents a scope reads: 2 or 3. It is `Any` when a scope is built from arbitrary definitions instead of from a format's own scope."""


@dataclass(frozen=True, slots=True, eq=False)
class Context(Generic[F]):
    """The definitions in scope while metadata of one Zarr format is read.

    A scope is a value and does no reading itself. `resolve` reads a field
    in a scope. The only question a scope answers is `claimant`: which
    definition a name belongs to. Build a scope from definitions with
    `Context.of`, and add more with `extended_with`. Two scopes are equal
    when they file the same definitions, and equal scopes have equal
    hashes. Every kind a scope files declares one format, and that is the
    scope's format. A v3 reader rejects a v2 scope, and the reverse. A
    scope that files nothing has no format, can be used by either reader,
    and claims nothing.
    """

    tables: Tables
    _format: Literal[2, 3] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # The constructor enforces what `of` guarantees: each key is a kind,
        # and every kind has the same format, which is the scope's format.
        formats = sorted({format_of(kind) for kind in self.tables})
        if len(formats) > 1:
            msg = (
                "a scope reads documents of one Zarr format; these definitions are of kinds of "
                f"formats {' and '.join(f'v{found}' for found in formats)}"
            )
            raise TypeError(msg)
        object.__setattr__(self, "_format", formats[0] if len(formats) == 1 else None)

    @property
    def format(self) -> Literal[2, 3] | None:
        """The Zarr format the definitions in scope read, 2 or 3; None for a scope that files nothing."""
        return self._format

    @classmethod
    def of(cls, *definitions: Definition[Any]) -> Context[Any]:
        """A scope of exactly these definitions. If two definitions share a kind and name, the later one replaces the earlier one.

        Raises `TypeError` for a definition of no kind, because no position
        in a document could hold it. Also raises `TypeError` for definitions
        whose kinds belong to two formats, because no document can contain
        both.
        """
        tables: dict[type[Definition[Any]], dict[str, Definition[Any]]] = {}
        for definition in definitions:
            kind = kind_of(definition)
            if kind is None:
                msg = (
                    f"{definition.name!r} is a definition of no kind; build it as a "
                    "CodecDefinition, DataTypeDefinition, ChunkGridDefinition, "
                    "ChunkKeyEncodingDefinition or StorageTransformerDefinition, or as a "
                    "kind of your own"
                )
                raise TypeError(msg)
            tables.setdefault(kind, {})[definition.name] = definition
        return cls(
            MappingProxyType({kind: MappingProxyType(table) for kind, table in tables.items()})
        )

    def extended_with(self, *definitions: Definition[Any]) -> Context[F]:
        """This scope, plus definitions of your own.

        A definition passed here replaces any definition already filed
        under the same kind and name. This is how a reader substitutes its
        own reading of a codec the package already defines, or of raw bits
        (by defining `r*`).
        """
        return Context.of(*self.definitions(), *definitions)

    def definitions(self) -> tuple[Definition[Any], ...]:
        """Every definition in scope, grouped by kind."""
        return tuple(entry for table in self.tables.values() for entry in table.values())

    def __repr__(self) -> str:
        # Keep this short, because it appears in signatures as a default
        # argument. A full repr would list every definition, and `help` on a
        # validator would run to pages.
        return f"Context(<{len(self.definitions())} definitions>)"

    def __reduce__(self) -> tuple[Callable[..., Context], tuple[Definition[Any], ...]]:
        # A scope is defined by its definitions, so pickle it as them. This
        # lets a scope go to another process along with the documents it
        # will read there.
        return (Context.of, self.definitions())

    def __copy__(self) -> Context:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Context:
        # A scope is immutable, so a copy can be the same object.
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Context):
            return NotImplemented
        return self._filed() == other._filed()

    def __hash__(self) -> int:
        return hash(self._filed())

    def _filed(self) -> frozenset[tuple[type[Definition[Any]], str, Definition[Any]]]:
        """Every definition in scope with the kind and name it is filed under. Scopes are compared by this set."""
        return frozenset(
            (kind, name, definition)
            for kind, table in self.tables.items()
            for name, definition in table.items()
        )

    def disagreements(self, claims: Claims) -> Disagreements:
        """The places where this scope reads `claims` (a reading's claims) differently: the names it would gain and the names it conflicts with, as described by `Disagreements`.

        A claim is keyed by the name its definition is filed under (raw
        bits under `r*`), so it is looked up by the filed name, not by the
        name a document uses.
        """
        return disagreements_of(lambda kind, name: self.tables.get(kind, {}).get(name), claims)

    @classmethod
    def joined(cls, *contexts: Context[F]) -> Context[F]:
        """The smallest scope that files everything each of `contexts` files (their join).

        Raises `ScopeConflictError` when two of them file different
        definitions under the same kind and name. To replace a definition
        on purpose, use `extended_with`.
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
        """The definition of `kind` in scope that reads `name` (a name as a document uses it), or None if there is none.

        This is the definition filed under the name that `filed_under`
        maps `name` to. That is `name` itself, except for raw bits, where
        `r16` is read by the definition of `r*`.
        """
        asked = as_kind(kind)
        filed, _ = filed_under(asked, name)
        if filed is None:
            return None
        return cast("D | None", self.tables.get(asked, {}).get(filed))


ZarrV3Context: TypeAlias = Context[Literal[3]]
"""A scope that reads Zarr v3 documents. Every v3 reader takes this type."""
ZarrV2Context: TypeAlias = Context[Literal[2]]
"""A scope that reads Zarr v2 documents. Every v2 reader takes this type."""


def is_scope(value: object) -> TypeGuard[Context[Any]]:
    """Whether `value` is a scope, of any format."""
    return isinstance(value, Context)


def scoped(context: Context[F] | None, default: Context[F]) -> Context[F]:
    """The scope for a reader of `default`'s format: `context`, or `default` when `context` is None. Raises `TypeError` for a scope of the other format."""
    if context is None:
        return default
    if context.format is not None and context.format != default.format:
        msg = (
            f"a v{default.format} document is read in a scope of that format or of none, "
            f"got a v{context.format} scope"
        )
        raise TypeError(msg)
    return context


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
"""The extensions that `zarr-extensions` registers and this package defines."""

CORE: Final[ZarrV3Context] = Context.of(*_CORE)
"""A scope with only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final[ZarrV3Context] = Context.of(*_CORE, *_EXTENSIONS)
"""A scope with what the specification defines, plus what `zarr-extensions` registers."""

__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "Context",
    "Tables",
    "ZarrV2Context",
    "ZarrV3Context",
    "is_scope",
    "scoped",
]
