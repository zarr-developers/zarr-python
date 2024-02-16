from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Literal
from dataclasses import dataclass
from zarr.v3.abc.metadata import Metadata

from zarr.v3.common import (
    JSON,
    ChunkCoords,
    parse_named_configuration,
)

if TYPE_CHECKING:
    pass

SeparatorLiteral = Literal[".", "/"]


def parse_separator(data: JSON) -> SeparatorLiteral:
    if data not in (".", "/"):
        raise ValueError(f"Expected an '.' or '/' separator. Got {data} instead.")
    return data  # type: ignore


@dataclass(frozen=True)
class ChunkKeyEncoding(Metadata):
    name: str
    separator: SeparatorLiteral = "."

    def __init__(self, *, separator: SeparatorLiteral) -> None:
        separator_parsed = parse_separator(separator)

        object.__setattr__(self, "separator", separator_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> ChunkKeyEncoding:
        if isinstance(data, ChunkKeyEncoding):
            return data  # type: ignore

        name_parsed, configuration_parsed = parse_named_configuration(data)
        if name_parsed == "default":
            return DefaultChunkKeyEncoding(**configuration_parsed)  # type: ignore[arg-type]
        if name_parsed == "v2":
            return V2ChunkKeyEncoding(**configuration_parsed)  # type: ignore[arg-type]
        raise ValueError(f"Unknown chunk key encoding. Got {name_parsed}.")

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": self.name, "configuration": {"separator": self.separator}}

    @abstractmethod
    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        pass

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        pass


@dataclass(frozen=True)
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    name: Literal["default"] = "default"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.separator.join(map(str, ("c",) + chunk_coords))


@dataclass(frozen=True)
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    name: Literal["v2"] = "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier
