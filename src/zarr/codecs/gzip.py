from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, TypeGuard, overload

from numcodecs.gzip import GZip
from typing_extensions import ReadOnly

from zarr.abc.codec import BytesBytesCodec, CodecJSON
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import (
    JSON,
    NamedRequiredConfig,
    ZarrFormat,
)

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


def parse_gzip_level(data: JSON) -> int:
    if not isinstance(data, (int)):
        raise TypeError(f"Expected int, got {type(data)}")
    if data not in range(10):
        raise ValueError(
            f"Expected an integer from the inclusive range (0, 9). Got {data} instead."
        )
    return data


class GZipConfig(TypedDict):
    level: int


class GZipJSON_V2(GZipConfig):
    """
    The JSON form of the GZip codec in Zarr V2.
    """

    id: ReadOnly[Literal["gzip"]]


class GZipJSON_V3(NamedRequiredConfig[Literal["gzip"], GZipConfig]):
    """
    The JSON form of the GZip codec in Zarr V3.
    """


@dataclass(frozen=True)
class GzipCodec(BytesBytesCodec):
    is_fixed_size = False

    level: int = 5

    def __init__(self, *, level: int = 5) -> None:
        level_parsed = parse_gzip_level(level)

        object.__setattr__(self, "level", level_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data, zarr_format=3)

    def to_dict(self) -> dict[str, JSON]:
        return self.to_json(zarr_format=3)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> GZipJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> GZipJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> GZipJSON_V2 | GZipJSON_V3:
        if zarr_format == 2:
            return {"id": "gzip", "level": self.level}
        elif zarr_format == 3:
            return {"name": "gzip", "configuration": {"level": self.level}}
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    @classmethod
    def _check_json_v2(cls, data: CodecJSON) -> TypeGuard[GZipJSON_V2]:
        return (
            isinstance(data, Mapping)
            and set(data.keys()) == {"id", "level"}
            and data["id"] == "gzip"
            and isinstance(data["level"], int)
        )

    @classmethod
    def _check_json_v3(cls, data: CodecJSON) -> TypeGuard[GZipJSON_V3]:
        return (
            isinstance(data, Mapping)
            and set(data.keys()) == {"name", "configuration"}
            and data["name"] == "gzip"
            and isinstance(data["configuration"], dict)
            and "level" in data["configuration"]
            and isinstance(data["configuration"]["level"], int)
        )

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if cls._check_json_v2(data):
            return cls(level=data["level"])
        raise ValueError(f"Invalid GZip JSON data for Zarr format 2: {data!r}")

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if cls._check_json_v3(data):
            return cls(level=data["configuration"]["level"])
        raise ValueError(f"Invalid GZip JSON data for Zarr format 3: {data!r}")

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, GZip(self.level).decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, GZip(self.level).encode, chunk_bytes, chunk_spec.prototype
        )

    def compute_encoded_size(
        self,
        _input_byte_length: int,
        _chunk_spec: ArraySpec,
    ) -> int:
        raise NotImplementedError
