"""The algebra of scopes: what a reading claims of each name, the order one reading refines another in, and where two scopes disagree.

A model is a document read in a scope. What the document means depends
only on what the scope said of the names it writes -- its *claims* --
not on everything the scope holds. Two readings of one document are
ordered by information: a name nothing claimed, read by a definition,
gains meaning and loses none; a name read by one definition and then
another is a conflict. `Context.disagreements` and `Context.joined` are
built on these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from zarr_metadata.v3._definition import (
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    Definition,
    StorageTransformerDefinition,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr_metadata._typed_json import Loc

ClaimKey: TypeAlias = tuple[type[Definition[Any]], str]
"""A kind and the name a definition is filed under: what a scope answers `claimant` for."""

Claims: TypeAlias = "Mapping[ClaimKey, Definition[Any] | None]"
"""What a reading claims of each name a document writes: the definition that read it, or None where nothing claimed it."""

_KIND_NAMES: dict[type[Definition[Any]], str] = {
    CodecDefinition: "codec",
    DataTypeDefinition: "data type",
    ChunkGridDefinition: "chunk grid",
    ChunkKeyEncodingDefinition: "chunk key encoding",
    StorageTransformerDefinition: "storage transformer",
}


@dataclass(frozen=True, slots=True)
class Conflict:
    """One place two readings of a name disagree: the key, what one claimed, what the other found, and where in a document when known."""

    key: ClaimKey
    claimed: Definition[Any] | None
    found: Definition[Any] | None
    loc: Loc | None = None

    def __str__(self) -> str:
        kind, name = self.key
        where = "" if self.loc is None else f" at {self.loc!r}"
        return (
            f"{_KIND_NAMES[kind]} {name!r}{where}: claimed {self.claimed!r}, found {self.found!r}"
        )


class ScopeConflictError(ValueError):
    """Raised where two scopes, or a scope and a reading, give one name two meanings, or one would lose a meaning the other has.

    Carries every conflict in `.conflicts`, as `MetadataValidationError`
    carries every problem.
    """

    def __init__(self, conflicts: Sequence[Conflict]) -> None:
        self.conflicts: tuple[Conflict, ...] = tuple(conflicts)
        super().__init__("; ".join(str(conflict) for conflict in self.conflicts))


__all__ = ["ClaimKey", "Claims", "Conflict", "ScopeConflictError"]
