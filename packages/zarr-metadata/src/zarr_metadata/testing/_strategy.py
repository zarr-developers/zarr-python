"""What draws one definition's configurations: a record for each kind of definition.

Each record holds what its kind's configurations depend on, the way
each kind of definition does: a data type's fill values depend on its
configuration, a chunk grid's on the shape of the array, a codec's on
the chunk it is handed. A chunk key encoding's depend on nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeAlias, cast

from zarr_metadata.v3._definition import (
    Chunk,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
)

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

    from zarr_metadata._common import JSONValue
    from zarr_metadata.definition._definition import Definition

Configurations: TypeAlias = "SearchStrategy[Mapping[str, JSONValue]]"
"""Configurations of one definition, each the mapping every configuration is."""

NOTHING_KNOWN: Final = Chunk()
"""A chunk nothing is known of, which every codec takes."""


class DefinitionStrategy(ABC):
    """How to draw the configurations of one kind of definition."""

    __slots__ = ()

    kind: ClassVar[type[Definition[Any]]]
    """The kind of definition whose configurations it draws."""

    @abstractmethod
    def unconstrained(self) -> Configurations:
        """Configurations when nothing is known of what they depend on: what the definition's rules accept."""


@dataclass(frozen=True, kw_only=True, slots=True)
class DataTypeStrategy(DefinitionStrategy):
    """How to draw a data type's configurations, and the fill values an array of one takes."""

    kind: ClassVar[type[Definition[Any]]] = DataTypeDefinition

    configurations: Configurations
    """Configurations its definition accepts."""
    fill_values: Callable[[Mapping[str, JSONValue]], SearchStrategy[JSONValue]]
    """Fill values of the data type a configuration describes."""

    def unconstrained(self) -> Configurations:
        return self.configurations


@dataclass(frozen=True, kw_only=True, slots=True)
class ChunkGridStrategy(DefinitionStrategy):
    """How to draw a chunk grid's configurations over an array of a shape."""

    kind: ClassVar[type[Definition[Any]]] = ChunkGridDefinition

    configurations: Callable[[tuple[int, ...] | None], Configurations]
    """Configurations of grids that fit an array of the shape.

    Handed None, no shape is known, and what is drawn is what the grid's
    rules accept, whatever shape it is put over.
    """

    def unconstrained(self) -> Configurations:
        return self.configurations(None)


@dataclass(frozen=True, kw_only=True, slots=True)
class ChunkKeyEncodingStrategy(DefinitionStrategy):
    """How to draw a chunk key encoding's configurations."""

    kind: ClassVar[type[Definition[Any]]] = ChunkKeyEncodingDefinition

    configurations: Configurations
    """Configurations its definition accepts."""

    def unconstrained(self) -> Configurations:
        return self.configurations


@dataclass(frozen=True, kw_only=True, slots=True)
class CodecStrategy(DefinitionStrategy):
    """How to draw a codec's configurations, handed a chunk.

    Every codec takes `Chunk()`, a chunk nothing is known of, so a
    strategy that draws nothing for it is refused when it is built.
    """

    kind: ClassVar[type[Definition[Any]]] = CodecDefinition

    configurations: Callable[[Chunk], Configurations | None]
    """Configurations of the codec handed the chunk; None when no configuration takes it.

    A `bytes` codec handed a `string` chunk is None, since the spec gives
    that data type a codec of its own.
    """

    def __post_init__(self) -> None:
        if self.configurations(NOTHING_KNOWN) is None:
            msg = "a codec strategy draws configurations for Chunk(), a chunk nothing is known of; got None"
            raise ValueError(msg)

    def unconstrained(self) -> Configurations:
        return cast("Configurations", self.configurations(NOTHING_KNOWN))


Strategies: TypeAlias = Mapping[Any, DefinitionStrategy]
"""How to draw each definition's configurations, by the definition.

The key is `Any`, not `Definition[Any]`: a mapping's key type is
invariant, so `{MY_CODEC: CodecStrategy(...)}`, a mapping keyed by a
`CodecDefinition`, would not be a `Mapping[Definition[Any], ...]`.
"""

__all__ = [
    "NOTHING_KNOWN",
    "ChunkGridStrategy",
    "ChunkKeyEncodingStrategy",
    "CodecStrategy",
    "Configurations",
    "DataTypeStrategy",
    "DefinitionStrategy",
    "Strategies",
]
