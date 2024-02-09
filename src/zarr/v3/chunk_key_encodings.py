from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Union
from dataclasses import asdict, dataclass, field
from zarr.v3.abc.metadata import Metadata

from zarr.v3.common import JSON, ChunkCoords, parse_name

if TYPE_CHECKING:
    from typing_extensions import Self

SeparatorLiteral = Literal[".", "/"]


def parse_separator(data: JSON) -> ChunkCoords:
    if data not in (".", "/"):
        raise ValueError(f"Expected an '.' or '/' separator. Got {data} instead.")
    return data


@dataclass(frozen=True)
class ChunkKeyEncoding(Metadata):
    name: str
    separator: SeparatorLiteral = "."

    def __init__(self, *, separator) -> None:
        separator_parsed = parse_separator(separator)

        object.__setattr__(self, "separator", separator_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        if isinstance(data, ChunkKeyEncoding):
            return data
        if data["name"] == "default":
            return DefaultChunkKeyEncoding(**data["configuration"])
        if data["name"] == "v2":
            return V2ChunkKeyEncoding(**data["configuration"])
        raise ValueError(f"Unknown chunk key encoding, got {data['name']}")

    def to_dict(self) -> Dict[str, JSON]:
        return {"name": self.name, "configuration": {"separator": self.separator}}


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
