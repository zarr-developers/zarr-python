from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsBytesBytesCodec,
    _warn_unstable_specification,
)
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedRequiredConfig,
    ZarrFormat,
    _check_codecjson_v2,
    _check_codecjson_v3,
)

if TYPE_CHECKING:
    pass


class LZ4Config(TypedDict):
    acceleration: int


class LZ4JSON_V2(LZ4Config):
    """JSON representation of LZ4 codec for Zarr V2."""

    id: ReadOnly[Literal["lz4"]]


class LZ4JSON_V3_Legacy(NamedRequiredConfig[Literal["numcodecs.lz4"], LZ4Config]):
    """Legacy JSON representation of LZ4 codec for Zarr V3."""

class LZ4JSON_V3(NamedRequiredConfig[Literal["lz4"], LZ4Config]):
    """JSON representation of LZ4 codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[LZ4JSON_V2]:
    """
    A type guard for the Zarr V2 form of the LZ4 codec JSON
    """
    return (
        _check_codecjson_v2(data)
        and data["id"] == "lz4"
        and "acceleration" in data
        and isinstance(data["acceleration"], int)
        and data["acceleration"] >= 1
    )


def check_json_v3(data: object) -> TypeGuard[LZ4JSON_V3 | LZ4JSON_V3_Legacy]:
    """
    A type guard for the Zarr V3 form of the LZ4 codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] in ("lz4", "numcodecs.lz4")
        and "configuration" in data
        and "acceleration" in data["configuration"]
        and isinstance(data["configuration"]["acceleration"], int)
        and data["configuration"]["acceleration"] >= 1
    )


class LZ4(_NumcodecsBytesBytesCodec):
    """
    A wrapper around the numcodecs.LZ4 codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.lz4"
    _codec_id = "lz4"
    codec_config: LZ4Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> LZ4JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> LZ4JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> LZ4JSON_V2 | LZ4JSON_V3:
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