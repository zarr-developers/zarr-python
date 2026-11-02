"""Operations on scopes: which definition a reading records for each name, how one reading refines another, and where a scope disagrees with a reading.

A model is a document read in a scope. The meaning of the document
depends only on what the scope returned for the names the document uses
(its *claims*), not on everything the scope contains. Two readings of one
document are ordered by how much they know. If a name was unclaimed and a
definition later reads it, the name gains meaning and loses none. If one
definition reads a name and then a different definition reads it, that is
a conflict. The internal `disagreements` function in
`zarr_metadata.definition._context`, which a model uses to compare a
scope with its claims, is built on these operations. A scope's `union`
raises the `ScopeConflictError` defined here when two scopes file
different definitions under one name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from zarr_metadata.definition._definition import (
    AcceptedDeclaration,
    Definition,
    RefusedDeclaration,
    UnclaimedDeclaration,
    as_kind,
    declaration_key,
    declarations_of,
    filed_under,
    format_of,
    own_key,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import D, ResolvedDeclaration

ClaimKey: TypeAlias = tuple[type[Definition[Any]], str]
"""A kind and the name a definition is filed under. A scope's `claimant` looks up definitions by this key."""

Claims: TypeAlias = Mapping[ClaimKey, Definition[Any] | None]
"""The claims a reading records for each name a document uses: the definition that read the name, or None if the name was unclaimed."""


@dataclass(frozen=True, slots=True)
class Conflict:
    """One place where two readings of a name disagree: the key, the definition one reading claimed, the definition the other found, and the location in the document when known."""

    key: ClaimKey
    """The kind, and the name the two definitions are filed under."""
    claimed: Definition[Any] | None
    """The definition one reading or scope gives the name; None when it leaves the name unclaimed."""
    found: Definition[Any] | None
    """The definition the other gives the name; None when it gives none, so the name would lose its meaning."""
    loc: Loc | None = None
    """Where the name occurs in the document; None when no document is available."""
    written: str | None = None
    """The name the document uses at the conflict location, for example `r16` for a key of `r*`; None when no document is available."""

    def __str__(self) -> str:
        kind, filed = self.key
        name = filed if self.written is None else self.written
        where = "" if self.loc is None else f" at {self.loc!r}"
        claimed, found = definition_said(self.claimed), definition_said(self.found)
        if claimed == found:
            # Nothing a description shows tells them apart, such as their
            # rules, so say that they differ.
            return f"{kind_name(kind)} {name!r}{where}: two different definitions, each {claimed}"
        return f"{kind_name(kind)} {name!r}{where}: claimed {claimed}, found {found}"


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
        """Every conflict, in the order found; the message joins them."""
        super().__init__("; ".join(str(conflict) for conflict in self.conflicts))

    def __reduce__(
        self,
    ) -> tuple[type[ScopeConflictError], tuple[tuple[Conflict, ...]], dict[str, object]]:
        # Pickle and copy the exception as it was raised, keeping its notes
        # and attributes. The default exception reduce calls the constructor
        # with the message, not the conflicts.
        return type(self), (self.conflicts,), dict(self.__dict__)


def claim_key(declaration: ResolvedDeclaration[Any]) -> ClaimKey | None:
    """The key `declaration` is claimed under: its kind and the name its definition is filed under (for example `r*` for `r16`). Returns None for a declaration with no name."""
    if declaration.name is None:
        return None
    filed, _ = filed_under(declaration.read_as, declaration.name)
    return None if filed is None else (declaration.read_as, filed)


def claims_of(
    declarations: Iterable[tuple[Loc, ResolvedDeclaration[Any]]],
) -> dict[ClaimKey, Definition[Any] | None]:
    """The claims that `declarations` (each paired with its location) record for each name: the definition that read the name, or None if the name was unclaimed.

    `declarations` must come from one reading, as returned by `declarations_of` or a
    reading's `declarations()`. If two declarations claim the same name with different
    definitions, this raises `ScopeConflictError`, because no single scope
    could have read them both.
    """
    claims: dict[ClaimKey, Definition[Any] | None] = {}
    conflicts: list[Conflict] = []
    for loc, declaration in declarations:
        key = claim_key(declaration)
        if key is None:
            continue
        definition = declaration.definition
        if key in claims and claims[key] != definition:
            conflicts.append(Conflict(key, claims[key], definition, loc))
            continue
        claims[key] = definition
    if len(conflicts) != 0:
        raise ScopeConflictError(conflicts)
    return claims


def refines(declaration: ResolvedDeclaration[Any], other: ResolvedDeclaration[Any]) -> bool:
    """Whether `declaration` contains everything `other` contains: it reads the same where both read, and it may also read what `other` left unclaimed.

    This is the order in which one reading of a document refines another.
    Suppose `other` is unclaimed and `declaration` is accepted. This counts as a
    gain when re-reading the unclaimed declaration's JSON, with `declaration`'s
    definition and the definitions of the declarations nested in `declaration`,
    produces `declaration`. Two spellings of one configuration count as one gain,
    because `==` treats them as one declaration, so the order is transitive
    through equality. The reverse direction (accepted to unclaimed) is a
    loss. One name read by two different definitions is a conflict. A
    refused declaration refines only itself. Two declarations that refine each other
    are equal.
    """
    if isinstance(declaration, RefusedDeclaration) or isinstance(other, RefusedDeclaration):
        return declaration == other
    if isinstance(other, UnclaimedDeclaration):
        if isinstance(declaration, UnclaimedDeclaration):
            return declaration_key(declaration) == declaration_key(other)
        return claim_key(declaration) == claim_key(other) and _gained(declaration, other)
    if isinstance(declaration, UnclaimedDeclaration):
        return False
    if declaration.definition != other.definition or own_key(declaration) != own_key(other):
        return False
    if set(declaration.nested) != set(other.nested):
        return False
    return all(refines(declaration.nested[loc], other.nested[loc]) for loc in declaration.nested)


def _gained(declaration: AcceptedDeclaration[Any], other: UnclaimedDeclaration) -> bool:
    """Whether reading `other` in the scope formed by `declaration`'s own claims produces `declaration`. This is the test for a gain."""
    again, _ = resolve(other.json, declaration.read_as, _Claimed.of(declaration))
    return again == declaration


@dataclass(frozen=True, slots=True)
class _Claimed:
    """The scope formed by a declaration's own claims: its definition and the definitions of its nested declarations, keyed by kind and filed name. `_gained` re-reads a declaration in this scope."""

    filed: Mapping[ClaimKey, Definition[Any]]
    zarr_format: Literal[2, 3]

    @classmethod
    def of(cls, declaration: AcceptedDeclaration[Any]) -> _Claimed:
        claimed = claims_of(declarations_of(declaration))
        return cls(
            {key: definition for key, definition in claimed.items() if definition is not None},
            format_of(declaration.read_as),
        )

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` among the claims that reads `name`, or None if there is none. This matches a scope's `claimant`."""
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
