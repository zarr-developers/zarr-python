"""Operations on scopes: which definition a reading records for each name, how one reading refines another, and where two scopes disagree.

A model is a document read in a scope. The meaning of the document
depends only on what the scope returned for the names the document uses
(its *claims*), not on everything the scope contains. Two readings of one
document are ordered by how much they know. If a name was unclaimed and a
definition later reads it, the name gains meaning and loses none. If one
definition reads a name and then a different definition reads it, that is
a conflict. `Context.disagreements` and `Context.joined` are built on
these operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from zarr_metadata.definition._definition import (
    AcceptedField,
    Definition,
    RefusedField,
    UnclaimedField,
    as_kind,
    field_key,
    fields_of,
    filed_under,
    format_of,
    own_key,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import D, ResolvedField

ClaimKey: TypeAlias = tuple[type[Definition[Any]], str]
"""A kind and the name a definition is filed under. A scope's `claimant` looks up definitions by this key."""

Claims: TypeAlias = Mapping[ClaimKey, Definition[Any] | None]
"""The claims a reading records for each name a document uses: the definition that read the name, or None if the name was unclaimed."""


@dataclass(frozen=True, slots=True)
class Conflict:
    """One place where two readings of a name disagree: the key, the definition one reading claimed, the definition the other found, and the location in the document when known."""

    key: ClaimKey
    claimed: Definition[Any] | None
    found: Definition[Any] | None
    loc: Loc | None = None
    written: str | None = None
    """The name the document uses at the conflict location, for example `r16` for a key of `r*`; None when no document is available."""

    def __str__(self) -> str:
        kind, filed = self.key
        name = filed if self.written is None else self.written
        where = "" if self.loc is None else f" at {self.loc!r}"
        return (
            f"{kind_name(kind)} {name!r}{where}: claimed {definition_said(self.claimed)}, "
            f"found {definition_said(self.found)}"
        )


def definition_said(definition: Definition[Any] | None) -> str:
    """Describe a definition for an error message. The description includes the configuration TypedDict, which distinguishes it from other definitions with the same name. Returns "no definition" for None."""
    if definition is None:
        return "no definition"
    return f"{definition!r} of {definition.configuration.__qualname__}"


def kind_name(kind: type[Definition[Any]]) -> str:
    """The name of a definition kind for an error message: its label, for example "codec" for `CodecDefinition`."""
    return kind.label


class ScopeConflictError(ValueError):
    """Raised when two scopes, or a scope and a reading, give one name two different meanings, or when one would lose a meaning the other has.

    Every conflict is stored in `.conflicts`, in the same way that
    `MetadataValidationError` stores every problem.
    """

    def __init__(self, conflicts: Sequence[Conflict]) -> None:
        self.conflicts: tuple[Conflict, ...] = tuple(conflicts)
        super().__init__("; ".join(str(conflict) for conflict in self.conflicts))

    def __reduce__(
        self,
    ) -> tuple[type[ScopeConflictError], tuple[tuple[Conflict, ...]], dict[str, object]]:
        # Pickle and copy the exception as it was raised, keeping its notes
        # and attributes. The default exception reduce calls the constructor
        # with the message, not the conflicts.
        return type(self), (self.conflicts,), dict(self.__dict__)


def claim_key(field: ResolvedField[Any]) -> ClaimKey | None:
    """The key `field` is claimed under: its kind and the name its definition is filed under (for example `r*` for `r16`). Returns None for a field with no name."""
    if field.name is None:
        return None
    filed, _ = filed_under(field.read_as, field.name)
    return None if filed is None else (field.read_as, filed)


def claims_of(
    fields: Iterable[tuple[Loc, ResolvedField[Any]]],
) -> dict[ClaimKey, Definition[Any] | None]:
    """The claims that `fields` (each paired with its location) record for each name: the definition that read the name, or None if the name was unclaimed.

    `fields` must come from one reading, as returned by `fields_of` or a
    reading's `fields()`. If two fields claim the same name with different
    definitions, this raises `ScopeConflictError`, because no single scope
    could have read them both.
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
    """Whether `field` contains everything `other` contains: it reads the same where both read, and it may also read what `other` left unclaimed.

    This is the order in which one reading of a document refines another.
    Suppose `other` is unclaimed and `field` is accepted. This counts as a
    gain when re-reading the unclaimed field's JSON, with `field`'s
    definition and the definitions of the fields nested in `field`,
    produces `field`. Two spellings of one configuration count as one gain,
    because `==` treats them as one field, so the order is transitive
    through equality. The reverse direction (accepted to unclaimed) is a
    loss. One name read by two different definitions is a conflict. A
    refused field refines only itself. Two fields that refine each other
    are equal.
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
    """Whether reading `other` in the scope formed by `field`'s own claims produces `field`. This is the test for a gain."""
    again, _ = resolve(other.json, field.read_as, _Claimed.of(field))
    return again == field


@dataclass(frozen=True, slots=True)
class _Claimed:
    """The scope formed by a field's own claims: its definition and the definitions of its nested fields, keyed by kind and filed name. `_gained` re-reads a field in this scope."""

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
        """The definition of `kind` among the claims that reads `name`, or None if there is none. This matches `Context.claimant`."""
        asked = as_kind(kind)
        filed, _ = filed_under(asked, name)
        if filed is None:
            return None
        return cast("D | None", self.filed.get((asked, filed)))


@dataclass(frozen=True, slots=True)
class Disagreements:
    """The places where a scope reads a reading's claims differently: the names that would gain a meaning, and the names that conflict (including names that would lose a meaning)."""

    gains: tuple[ClaimKey, ...]
    conflicts: tuple[Conflict, ...]

    @property
    def agrees(self) -> bool:
        """Whether the scope reads every claim identically."""
        return len(self.gains) == 0 and len(self.conflicts) == 0


def disagreements_of(
    claimant: Callable[[type[Definition[Any]], str], Definition[Any] | None], claims: Claims
) -> Disagreements:
    """The `Disagreements` between `claimant` and `claims`. For each key, compare the definition `claimant` returns (looked up by kind and filed name, like a scope's tables) with the definition `claims` records."""
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
