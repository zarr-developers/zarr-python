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
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.crc32c import CRC32C_CODEC
from zarr_metadata.v3.codec.gzip import GZIP_CODEC

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

        A name already filed under the same kind is taken over by what is
        passed here, which is how a reader substitutes its own reading of
        a codec the package already defines.
        """
        return Context.of(*self.definitions(), *definitions)

    def definitions(self) -> tuple[Definition[Any], ...]:
        """Every definition in scope, kind by kind."""
        return tuple(entry for table in self.tables.values() for entry in table.values())

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` in scope that claims `name`; None if none does.

        Asks each definition filed under the kind whether the name is its
        own -- a family claims every `r<N>` -- rather than looking a key
        up, so the names a scope files under exist for `extended_with` to
        take one over.
        """
        table = self.tables.get(as_kind(kind), {})
        return cast(
            "D | None", next((entry for entry in table.values() if entry.claims(name)), None)
        )


_CORE: Final[tuple[Definition[Any], ...]] = (
    BYTES_CODEC,
    CRC32C_CODEC,
    GZIP_CODEC,
    REGULAR_CHUNK_GRID,
)
"""What the Zarr v3 specification itself defines."""

_EXTENSIONS: Final[tuple[Definition[Any], ...]] = ()
"""What `zarr-extensions` registers and this package defines."""

CORE: Final = Context.of(*_CORE)
"""Only what the Zarr v3 specification defines."""

CORE_AND_EXTENSIONS: Final = Context.of(*_CORE, *_EXTENSIONS)
"""What the specification defines, plus what `zarr-extensions` registers."""

__all__ = ["CORE", "CORE_AND_EXTENSIONS", "Context", "Tables"]
