from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict, cast

if TYPE_CHECKING:
    from typing import NotRequired, Self

from zarr.abc.metadata import Metadata
from zarr.core.common import (
    JSON,
    NamedConfig,
    parse_named_configuration,
)
from zarr.core.json_parse import parse_field
from zarr.registry import get_chunk_key_encoding_class, register_chunk_key_encoding

SeparatorLiteral = Literal[".", "/"]


def parse_separator(data: JSON) -> SeparatorLiteral:
    return cast("SeparatorLiteral", parse_field(data, Literal[".", "/"], "separator"))


class ChunkKeyEncodingParams(TypedDict):
    name: Literal["v2", "default", "offset"]
    separator: NotRequired[SeparatorLiteral]
    offset: NotRequired[tuple[int, ...]]


@dataclass(frozen=True)
class ChunkKeyEncoding(ABC, Metadata):
    """
    Defines how chunk coordinates are mapped to store keys.

    Subclasses must define a class variable `name` and implement `encode_chunk_key`.
    """

    name: ClassVar[str]

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, config_parsed = parse_named_configuration(data, require_configuration=False)
        return cls(**config_parsed or {})

    def to_dict(self) -> dict[str, JSON]:
        return {"name": self.name, "configuration": super().to_dict()}

    def decode_chunk_key(self, chunk_key: str) -> tuple[int, ...]:
        """
        Optional: decode a chunk key string into chunk coordinates.
        Not required for normal operation; override if needed for testing or debugging.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement decode_chunk_key.")

    @abstractmethod
    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        """
        Encode chunk coordinates into a chunk key string.
        Must be implemented by subclasses.
        """


type ChunkKeyEncodingLike = (
    dict[str, JSON] | ChunkKeyEncodingParams | ChunkKeyEncoding | NamedConfig[str, Any]
)


@dataclass(frozen=True)
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    name: ClassVar[Literal["default"]] = "default"
    separator: SeparatorLiteral = "/"

    def __post_init__(self) -> None:
        separator_parsed = parse_separator(self.separator)
        object.__setattr__(self, "separator", separator_parsed)

    def decode_chunk_key(self, chunk_key: str) -> tuple[int, ...]:
        if chunk_key == "c":
            return ()
        # Strip the "c<sep>" prefix (e.g. "c/" or "c.") before splitting.
        prefix = "c" + self.separator
        if chunk_key.startswith(prefix):
            return tuple(map(int, chunk_key[len(prefix) :].split(self.separator)))
        raise ValueError(f"Invalid chunk key for default encoding: {chunk_key!r}")

    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        return self.separator.join(map(str, ("c",) + chunk_coords))


@dataclass(frozen=True)
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    name: ClassVar[Literal["v2"]] = "v2"
    separator: SeparatorLiteral = "."

    def __post_init__(self) -> None:
        separator_parsed = parse_separator(self.separator)
        object.__setattr__(self, "separator", separator_parsed)

    def decode_chunk_key(self, chunk_key: str) -> tuple[int, ...]:
        return tuple(map(int, chunk_key.split(self.separator)))

    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        chunk_identifier = self.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


def parse_offset(data: object) -> tuple[int, ...]:
    if isinstance(data, (str, bytes)) or not isinstance(data, Iterable):
        raise TypeError(f"offset must be a sequence of integers, got {data!r}")
    offset = tuple(data)
    if any(isinstance(o, bool) or not isinstance(o, int) for o in offset):
        raise TypeError(f"offset must be a sequence of integers, got {data!r}")
    return offset


@dataclass(frozen=True)
class OffsetChunkKeyEncoding(ChunkKeyEncoding):
    """
    The ``default`` encoding applied to ``chunk_coords + offset``, one integer per dimension.

    https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/offset
    """

    name: ClassVar[Literal["offset"]] = "offset"
    offset: tuple[int, ...]
    separator: SeparatorLiteral = "/"

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", parse_offset(self.offset))
        object.__setattr__(self, "separator", parse_separator(self.separator))

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, config_parsed = parse_named_configuration(data, require_configuration=False)
        if not config_parsed or "offset" not in config_parsed:
            raise ValueError(
                "The offset chunk key encoding requires an offset in its configuration."
            )
        return cls(**cast("dict[str, Any]", config_parsed))

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": self.name,
            "configuration": {"separator": self.separator, "offset": list(self.offset)},
        }

    def _check_rank(self, chunk_coords: tuple[int, ...]) -> None:
        if len(chunk_coords) != len(self.offset):
            raise ValueError(
                f"The offset has {len(self.offset)} dimension(s) but the chunk coordinates "
                f"{chunk_coords!r} have {len(chunk_coords)}."
            )

    def decode_chunk_key(self, chunk_key: str) -> tuple[int, ...]:
        shifted = DefaultChunkKeyEncoding(separator=self.separator).decode_chunk_key(chunk_key)
        self._check_rank(shifted)
        return tuple(c - o for c, o in zip(shifted, self.offset, strict=True))

    def encode_chunk_key(self, chunk_coords: tuple[int, ...]) -> str:
        self._check_rank(chunk_coords)
        shifted = tuple(c + o for c, o in zip(chunk_coords, self.offset, strict=True))
        return self.separator.join(map(str, ("c",) + shifted))


def parse_chunk_key_encoding(data: ChunkKeyEncodingLike) -> ChunkKeyEncoding:
    """
    Take an implicit specification of a chunk key encoding and parse it into a ChunkKeyEncoding object.
    """
    if isinstance(data, ChunkKeyEncoding):
        return data

    # handle ChunkKeyEncodingParams
    if "name" in data and "configuration" not in data and ("separator" in data or "offset" in data):
        params = cast("dict[str, JSON]", data)
        configuration = {key: params[key] for key in ("separator", "offset") if key in params}
        data = cast("dict[str, JSON]", {"name": params["name"], "configuration": configuration})

    # Now must be a named config
    data = cast("dict[str, JSON]", data)

    name_parsed, _ = parse_named_configuration(data, require_configuration=False)
    try:
        chunk_key_encoding = get_chunk_key_encoding_class(name_parsed).from_dict(data)
    except KeyError as e:
        raise ValueError(f"Unknown chunk key encoding: {e.args[0]!r}") from e

    return chunk_key_encoding


register_chunk_key_encoding("default", DefaultChunkKeyEncoding)
register_chunk_key_encoding("v2", V2ChunkKeyEncoding)
register_chunk_key_encoding("offset", OffsetChunkKeyEncoding)
