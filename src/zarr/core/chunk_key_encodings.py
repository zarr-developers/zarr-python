from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, cast

if TYPE_CHECKING:
    from typing import NotRequired

from zarr.abc.metadata import Metadata
from zarr.core.common import (
    JSON,
    ChunkCoords,
    parse_named_configuration,
)

SeparatorLiteral = Literal[".", "/"]


def parse_separator(data: JSON) -> SeparatorLiteral:
    if data not in (".", "/"):
        raise ValueError(f"Expected an '.' or '/' separator. Got {data} instead.")
    return cast(SeparatorLiteral, data)


class ChunkKeyEncodingParams(TypedDict):
    name: Literal["v2", "default"]
    separator: NotRequired[SeparatorLiteral]


@dataclass(frozen=True)
class ChunkKeyEncoding(Metadata):
    name: str
    separator: SeparatorLiteral = "."

    def __init__(self, *, separator: SeparatorLiteral) -> None:
        separator_parsed = parse_separator(separator)

        object.__setattr__(self, "separator", separator_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON] | ChunkKeyEncodingLike) -> ChunkKeyEncoding:
        if isinstance(data, ChunkKeyEncoding):
            return data

        # handle ChunkKeyEncodingParams
        if "name" in data and "separator" in data:
            data = {"name": data["name"], "configuration": {"separator": data["separator"]}}

        # TODO: remove this cast when we are statically typing the JSON metadata completely.
        data = cast(dict[str, JSON], data)

        # configuration is optional for chunk key encodings
        name_parsed, config_parsed = parse_named_configuration(data, require_configuration=False)
        if name_parsed == "default":
            if config_parsed is None:
                # for default, normalize missing configuration to use the "/" separator.
                config_parsed = {"separator": "/"}
            return DefaultChunkKeyEncoding(**config_parsed)  # type: ignore[arg-type]
        if name_parsed == "v2":
            if config_parsed is None:
                # for v2, normalize missing configuration to use the "." separator.
                config_parsed = {"separator": "."}
            return V2ChunkKeyEncoding(**config_parsed)  # type: ignore[arg-type]
        msg = f"Unknown chunk key encoding. Got {name_parsed}, expected one of ('v2', 'default')."
        raise ValueError(msg)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": {"separator": self.separator}}

    @abstractmethod
    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        pass

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        pass


ChunkKeyEncodingLike: TypeAlias = ChunkKeyEncodingParams | ChunkKeyEncoding


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
