from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsChecksumCodec,
    _warn_unstable_specification,
)
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedConfig,
    ZarrFormat,
    check_codecjson_v2,
    check_named_required_config,
)


class Crc32Config(TypedDict, total=False):
    """Configuration parameters for CRC32 codec."""

    location: Literal["start", "end"]


class Crc32JSON_V2(Crc32Config):
    """JSON representation of CRC32 codec for Zarr V2."""

    id: ReadOnly[Literal["crc32"]]


class Crc32JSON_V3(NamedConfig[Literal["crc32"], Crc32Config]):
    """JSON representation of CRC32 codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[Crc32JSON_V2]:
    """
    A type guard for the Zarr V2 form of the CRC32 codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "crc32"
        and ("location" not in data or data["location"] in ("start", "end"))  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[Crc32JSON_V3]:
    """
    A type guard for the Zarr V3 form of the CRC32 codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "crc32"
        and (
            "location" not in data["configuration"]
            or data["configuration"]["location"] in ("start", "end")
        )
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if data.get("name") == "numcodecs.crc32":
            data_copy = data_copy | {"name": "crc32"}
        return data_copy  # type: ignore[return-value]
    return data


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
        data = _handle_json_alias_v3(data)
        if check_json_v3(data):
            config = data.get("configuration", {})
            return cls(**config)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
