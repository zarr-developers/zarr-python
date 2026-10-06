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
    Refused,
    StorageTransformerDefinition,
    Unclaimed,
    field_key,
    own_key,
    spelled,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v3._definition import Resolved

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


def claim_key(field: Resolved[Any]) -> ClaimKey | None:
    """The key `field` is claimed under: its kind and the name its definition is filed under, `r*` for `r16`; None for a field that names nothing."""
    if field.name is None:
        return None
    filed, _ = spelled(field.read_as, field.name)
    return None if filed is None else (field.read_as, filed)


def claims_of(
    fields: Iterable[tuple[Loc, Resolved[Any]]],
) -> dict[ClaimKey, Definition[Any] | None]:
    """What `fields`, each with where it sits, claim of each name: the definition that read it, None where nothing claimed it.

    `fields` are one reading's, as `fields_of` or a reading's `fields()`
    gives them. A name claimed two ways among them is a
    `ScopeConflictError`: no one scope read them.
    """
    claims: dict[ClaimKey, Definition[Any] | None] = {}
    conflicts: list[Conflict] = []
    for loc, field in fields:
        key = claim_key(field)
        if key is None:
            continue
        definition = field.definition
        if key in claims and claims[key] != definition:
            conflicts.append(Conflict(key, claims[key], definition, loc))
            continue
        claims[key] = definition
    if len(conflicts) != 0:
        raise ScopeConflictError(conflicts)
    return claims


def refines(field: Resolved[Any], other: Resolved[Any]) -> bool:
    """Whether `field` holds everything `other` holds: reads the same where both read, and reads what `other` left unclaimed.

    The order one reading of a document refines another in. A name nothing
    claimed, read by a definition, is a gain; the reverse is a loss; one
    name read by two definitions is a conflict; a refused field is in the
    order with nothing. Two fields that refine each other are equal.
    """
    if isinstance(field, Refused) or isinstance(other, Refused):
        return False
    if isinstance(other, Unclaimed):
        if isinstance(field, Unclaimed):
            return field_key(field) == field_key(other)
        return claim_key(field) == claim_key(other) and field.json == other.json
    if isinstance(field, Unclaimed):
        return False
    if field.definition != other.definition or own_key(field) != own_key(other):
        return False
    if set(field.nested) != set(other.nested):
        return False
    return all(refines(field.nested[loc], other.nested[loc]) for loc in field.nested)


@dataclass(frozen=True, slots=True)
class Disagreements:
    """Where a scope reads a reading's claims otherwise: the names it would gain a meaning for, and those it conflicts with, a lost meaning among them."""

    gains: tuple[ClaimKey, ...]
    conflicts: tuple[Conflict, ...]

    @property
    def agrees(self) -> bool:
        """Whether the scope reads every claim identically."""
        return len(self.gains) == 0 and len(self.conflicts) == 0


def disagreements_of(
    claimant: Callable[[type[Definition[Any]], str], Definition[Any] | None], claims: Claims
) -> Disagreements:
    """`Disagreements` between what `claimant`, asked by kind and filed name as a scope's tables answer, gives for each key and what `claims` records."""
    gains: list[ClaimKey] = []
    conflicts: list[Conflict] = []
    for key, claimed in claims.items():
        kind, name = key
        found = claimant(kind, name)
        if found == claimed:
            continue
        if claimed is None:
            gains.append(key)
        else:
            conflicts.append(Conflict(key, claimed, found))
    return Disagreements(tuple(gains), tuple(conflicts))


__all__ = [
    "ClaimKey",
    "Claims",
    "Conflict",
    "Disagreements",
    "ScopeConflictError",
    "claim_key",
    "claims_of",
    "disagreements_of",
    "refines",
]
