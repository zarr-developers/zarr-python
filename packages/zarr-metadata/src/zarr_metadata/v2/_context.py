"""The scope a Zarr v2 document is read in: one slot for each kind of v2 field."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from zarr_metadata.definition._context import Slots
from zarr_metadata.v2._definition import ZarrV2CodecDefinition, ZarrV2DataTypeDefinition

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from zarr_metadata.definition._definition import D, Definition
    from zarr_metadata.definition._scope import Claims, Disagreements


class ZarrV2Context:
    """A scope that reads Zarr v2 documents. Every v2 reader takes one.

    It implements `Context`, and has one slot for each kind of field a v2
    document holds: `data_types`, read from `dtype`, and `codecs`, read
    from `compressor` and `filters`. The constructor takes an iterable of
    definitions for each slot, and each slot is a read-only mapping from
    name to definition. A definition of another kind than its slot holds
    raises `TypeError`, and two different definitions under one name in a
    slot raise `ScopeConflictError`. To replace a definition, use
    `extended_with`. `ZarrV2Context()` is the scope that claims nothing.
    Two scopes are equal when they file the same definitions in each slot,
    and equal scopes have equal hashes.
    """

    __slots__ = ("_slots",)

    kinds: ClassVar[Mapping[str, type[Definition[Any]]]] = MappingProxyType(
        {
            "data_types": ZarrV2DataTypeDefinition,
            "codecs": ZarrV2CodecDefinition,
        }
    )
    """The kind of definition each slot holds, by slot."""

    _slots: Slots

    def __init__(
        self,
        *,
        data_types: Iterable[ZarrV2DataTypeDefinition[Any]] = (),
        codecs: Iterable[ZarrV2CodecDefinition[Any]] = (),
    ) -> None:
        self._slots = Slots.of(
            self.kinds,
            {
                "data_types": data_types,
                "codecs": codecs,
            },
            2,
        )

    @classmethod
    def _of(cls, slots: Slots) -> ZarrV2Context:
        """A scope that files `slots`, which were filed for this class's slots."""
        context = object.__new__(cls)
        context._slots = slots
        return context

    @classmethod
    def _restored(cls, contents: Mapping[str, Iterable[Definition[Any]]]) -> ZarrV2Context:
        """A scope rebuilt from the definitions in each slot, as `__reduce__` gives them."""
        return cls._of(Slots.of(cls.kinds, contents, 2))

    @property
    def zarr_format(self) -> Literal[2]:
        """The Zarr format of the documents this scope reads: 2."""
        return 2

    @property
    def data_types(self) -> Mapping[str, ZarrV2DataTypeDefinition[Any]]:
        """The data type definitions in scope, by the NumPy family each reads."""
        return cast("Mapping[str, ZarrV2DataTypeDefinition[Any]]", self._slots.tables["data_types"])

    @property
    def codecs(self) -> Mapping[str, ZarrV2CodecDefinition[Any]]:
        """The codec definitions in scope, by numcodecs id."""
        return cast("Mapping[str, ZarrV2CodecDefinition[Any]]", self._slots.tables["codecs"])

    def definitions_of(self, kind: type[D]) -> Mapping[str, D]:
        """The definitions of `kind` in scope, by the name each is filed under: the slot that holds `kind`. Empty for a kind that no slot holds, such as a kind of the other format."""
        return self._slots.definitions_of(kind)

    def claimant(self, kind: type[D], name: str) -> D | None:
        """The definition of `kind` in scope that reads `name` (a name as a document uses it), or None if there is none.

        This is the definition filed under the name that `filed_under`
        maps `name` to. That is `name` itself, except for raw bits, where
        `r16` is read by the definition of `r*`.
        """
        return self._slots.claimant(kind, name)

    def disagreements(self, claims: Claims) -> Disagreements:
        """The places where this scope reads `claims` (a reading's claims) differently: the names it would gain and the names it conflicts with, as described by `Disagreements`. A claim is keyed by the name its definition is filed under, so it is looked up by that name."""
        return self._slots.disagreements(claims)

    def extended_with(
        self,
        *,
        data_types: Iterable[ZarrV2DataTypeDefinition[Any]] = (),
        codecs: Iterable[ZarrV2CodecDefinition[Any]] = (),
    ) -> ZarrV2Context:
        """This scope, plus definitions of your own, each in its slot.

        A definition given here replaces any definition already filed in
        its slot under the same name. This is how a reader substitutes its
        own reading of a definition the package already defines.
        """
        return self._of(
            self._slots.extended(
                {
                    "data_types": data_types,
                    "codecs": codecs,
                },
                2,
            )
        )

    @classmethod
    def joined(cls, *contexts: ZarrV2Context) -> ZarrV2Context:
        """The smallest scope that files everything each of `contexts` files (their join).

        Raises `ScopeConflictError` when two of them file different
        definitions under one name in a slot. To replace a definition on
        purpose, use `extended_with`. Raises `TypeError` for a scope of
        another format.
        """
        return cls._of(Slots.joined(cls.kinds, [context._slots for context in contexts], 2))

    def __repr__(self) -> str:
        # Keep this short, because it appears in signatures as a default
        # argument. A full repr would list every definition, and `help` on a
        # validator would run to pages.
        return f"ZarrV2Context(<{len(self._slots)} definitions>)"

    def __reduce__(self) -> tuple[object, tuple[dict[str, tuple[Definition[Any], ...]]]]:
        # A scope is defined by the definitions in its slots, so pickle it as
        # them. This lets a scope go to another process along with the
        # documents it will read there.
        return (type(self)._restored, (self._slots.contents(),))

    def __copy__(self) -> ZarrV2Context:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> ZarrV2Context:
        # A scope is immutable, so a copy can be the same object.
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZarrV2Context):
            return NotImplemented
        return self._slots == other._slots

    def __hash__(self) -> int:
        return hash((ZarrV2Context, self._slots))


__all__ = ["ZarrV2Context"]
