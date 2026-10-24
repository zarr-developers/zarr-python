"""The scope a Zarr v2 document is read in: one slot for each kind of v2 field."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

from zarr_metadata.definition._context import Context
from zarr_metadata.v2._definition import ZarrV2CodecDefinition, ZarrV2DataTypeDefinition

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from zarr_metadata.definition._definition import Definition


class ZarrV2Context(Context):
    """A scope that reads Zarr v2 documents. Every v2 reader takes one.

    It has one slot for each kind of field a v2 document holds:
    `data_types`, read from `dtype`, and `codecs`, read from `compressor`
    and `filters`. The constructor takes an iterable of definitions for
    each slot, and each slot is a read-only mapping from name to
    definition. A definition of another kind than its slot holds raises
    `TypeError`, and two different definitions under one name in a slot
    raise `ScopeConflictError`. To replace a definition, use
    `extended_with`. `ZarrV2Context()` is the scope that claims nothing.
    """

    __slots__ = ()

    kinds: ClassVar[Mapping[str, type[Definition[Any]]]] = MappingProxyType(
        {"data_types": ZarrV2DataTypeDefinition, "codecs": ZarrV2CodecDefinition}
    )

    def __init__(
        self,
        *,
        data_types: Iterable[ZarrV2DataTypeDefinition[Any]] = (),
        codecs: Iterable[ZarrV2CodecDefinition[Any]] = (),
    ) -> None:
        super().__init__({"data_types": data_types, "codecs": codecs})

    @property
    def data_types(self) -> Mapping[str, ZarrV2DataTypeDefinition[Any]]:
        """The data type definitions in scope, by the NumPy family each reads."""
        return cast(
            "Mapping[str, ZarrV2DataTypeDefinition[Any]]", self._held(ZarrV2DataTypeDefinition)
        )

    @property
    def codecs(self) -> Mapping[str, ZarrV2CodecDefinition[Any]]:
        """The codec definitions in scope, by numcodecs id."""
        return cast("Mapping[str, ZarrV2CodecDefinition[Any]]", self._held(ZarrV2CodecDefinition))

    def extended_with(
        self,
        *,
        data_types: Iterable[ZarrV2DataTypeDefinition[Any]] = (),
        codecs: Iterable[ZarrV2CodecDefinition[Any]] = (),
    ) -> ZarrV2Context:
        """This scope, plus definitions of your own, each in its slot. A definition given here replaces any definition already filed in its slot under the same name."""
        return self._extended({"data_types": data_types, "codecs": codecs})


__all__ = ["ZarrV2Context"]
