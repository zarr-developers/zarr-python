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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from zarr_metadata.v3._definition import (
    AcceptedField,
    Definition,
    RefusedField,
    UnclaimedField,
    as_kind,
    field_key,
    fields_of,
    format_of,
    own_key,
    resolve,
    spelled,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v3._definition import D, ResolvedField

ClaimKey: TypeAlias = tuple[type[Definition[Any]], str]
"""A kind and the name a definition is filed under: what a scope answers `claimant` for."""

Claims: TypeAlias = Mapping[ClaimKey, Definition[Any] | None]
"""What a reading claims of each name a document writes: the definition that read it, or None where nothing claimed it."""


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
        return f"{kind_name(kind)} {name!r}{where}: claimed {self.claimed!r}, found {self.found!r}"


def kind_name(kind: type[Definition[Any]]) -> str:
    """A kind of definition as a message names it: its label, "codec" for `CodecDefinition`."""
    return kind.label


class ScopeConflictError(ValueError):
    """Raised where two scopes, or a scope and a reading, give one name two meanings, or one would lose a meaning the other has.

    Carries every conflict in `.conflicts`, as `MetadataValidationError`
    carries every problem.
    """

    def __init__(self, conflicts: Sequence[Conflict]) -> None:
        self.conflicts: tuple[Conflict, ...] = tuple(conflicts)
        super().__init__("; ".join(str(conflict) for conflict in self.conflicts))

    def __reduce__(self) -> tuple[type[ScopeConflictError], tuple[tuple[Conflict, ...]]]:
        # Pickled and copied as it was raised: an exception's default
        # reduce calls the constructor with its message, not its conflicts.
        return type(self), (self.conflicts,)


def claim_key(field: ResolvedField[Any]) -> ClaimKey | None:
    """The key `field` is claimed under: its kind and the name its definition is filed under, `r*` for `r16`; None for a field that names nothing."""
    if field.name is None:
        return None
    filed, _ = spelled(field.read_as, field.name)
    return None if filed is None else (field.read_as, filed)


def claims_of(
    fields: Iterable[tuple[Loc, ResolvedField[Any]]],
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


def refines(field: ResolvedField[Any], other: ResolvedField[Any]) -> bool:
    """Whether `field` holds everything `other` holds: reads the same where both read, and reads what `other` left unclaimed.

    The order one reading of a document refines another in. A name nothing
    claimed, read by a definition, is a gain when what the unclaimed field
    wrote, read by that definition and those of the fields the read field
    holds, is the read field: two spellings of one configuration are one
    gain, as they are one field to `==`, so the order is transitive
    through equality. The reverse is a loss; one name read by two
    definitions is a conflict; a refused field refines itself alone. Two
    fields that refine each other are equal.
    """
    if isinstance(field, RefusedField) or isinstance(other, RefusedField):
        return field == other
    if isinstance(other, UnclaimedField):
        if isinstance(field, UnclaimedField):
            return field_key(field) == field_key(other)
        return claim_key(field) == claim_key(other) and _gained(field, other)
    if isinstance(field, UnclaimedField):
        return False
    if field.definition != other.definition or own_key(field) != own_key(other):
        return False
    if set(field.nested) != set(other.nested):
        return False
    return all(refines(field.nested[loc], other.nested[loc]) for loc in field.nested)


def _gained(field: AcceptedField[Any], other: UnclaimedField) -> bool:
    """Whether `other`, read in the scope `field`'s own claims make, is `field`: what a gain is."""
    again, _ = resolve(other.json, field.read_as, _Claimed.of(field))
    return again == field


@dataclass(frozen=True, slots=True)
class _Claimed:
    """The scope a field's own claims make: its definition and those of the fields it holds, by kind and filed name; what a gain re-reads in."""

    filed: Mapping[ClaimKey, Definition[Any]]
    format: Literal[2, 3] | None

    @classmethod
    def of(cls, field: AcceptedField[Any]) -> _Claimed:
        claimed = claims_of(fields_of(field))
        return cls(
            {key: definition for key, definition in claimed.items() if definition is not None},
            format_of(field.read_as),
        )

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` among the claims that reads `name`; None if none does, as `Context.claimant` answers."""
        asked = as_kind(kind)
        filed, _ = spelled(asked, name)
        if filed is None:
            return None
        return cast("D | None", self.filed.get((asked, filed)))


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
    "kind_name",
    "refines",
]
