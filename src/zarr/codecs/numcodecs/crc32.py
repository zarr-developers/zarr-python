from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsChecksumCodec,
    _warn_unstable_specification,
)
from zarr.core.common import CodecJSON, CodecJSON_V2, CodecJSON_V3, NamedConfig, ZarrFormat, _check_codecjson_v2, _check_codecjson_v3

if TYPE_CHECKING:
    pass


class Crc32Config(TypedDict):
    """Configuration parameters for CRC32 codec."""

    location: Literal["start", "end"]


class Crc32JSON_V2(Crc32Config):
    """JSON representation of CRC32 codec for Zarr V2."""

    id: ReadOnly[Literal["crc32"]]

class Crc32JSON_V3_Legacy(NamedConfig[Literal["numcodecs.crc32"], Crc32Config]):
    """Legacy JSON representation of CRC32 codec for Zarr V3."""


class Crc32JSON_V3(NamedConfig[Literal["crc32"], Crc32Config]):
    """JSON representation of CRC32 codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[Crc32JSON_V2]:
    """
    A type guard for the Zarr V2 form of the CRC32 codec JSON
    """
    return (
        _check_codecjson_v2(data)
        and data["id"] == "crc32"
        and ("location" not in data or data["location"] in ("start", "end"))
    )


def check_json_v3(data: object) -> TypeGuard[Crc32JSON_V3 | Crc32JSON_V3_Legacy]:
    """
    A type guard for the Zarr V3 form of the CRC32 codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] in ("crc32", "numcodecs.crc32")
        and (
            "configuration" not in data
            or (
                "location" not in data["configuration"]
                or data["configuration"]["location"] in ("start", "end")
            )
        )
    )


class CRC32(_NumcodecsChecksumCodec):
    """
    A wrapper around the numcodecs.CRC32 codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.crc32"
    _codec_id = "crc32"
    codec_config: Crc32Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Crc32JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Crc32JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Crc32JSON_V2 | Crc32JSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]
    
    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        if check_json_v3(data):
            config = data["configuration"]
            return cls(**config)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if _check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)