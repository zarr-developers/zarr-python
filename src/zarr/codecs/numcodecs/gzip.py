from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, overload

from zarr.codecs.gzip import GZipConfig
from zarr.codecs.numcodecs._codecs import _NumcodecsBytesBytesCodec
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
    from zarr.codecs.gzip import GZipJSON_V2, GZipJSON_V3


def check_json_v2(data: object) -> TypeGuard[GZipJSON_V2]:
    """
    A type guard for the Zarr V2 form of the GZip codec JSON
    """
    return (
        _check_codecjson_v2(data)
        and data["id"] == "gzip"
        and "level" in data
        and isinstance(data["level"], int)
        and 0 <= data["level"] <= 9
    )


class GZipJSON_V3_Legacy(NamedRequiredConfig[Literal["numcodecs.gzip"], GZipConfig]):
    """
    The JSON form of the GZip codec in Zarr V3.
    """


def check_json_v3(data: object) -> TypeGuard[GZipJSON_V3 | GZipJSON_V3_Legacy]:
    """
    A type guard for the Zarr V3 form of the GZip codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] in ("numcodecs.gzip", "gzip")
        and "configuration" in data
        and "level" in data["configuration"]
        and isinstance(data["configuration"]["level"], int)
        and 0 <= data["configuration"]["level"] <= 9
    )


class GZip(_NumcodecsBytesBytesCodec):
    """
    A legacy wrapper used to provide a Zarr V3 API for the numcodecs gzip codec.

    Use `zarr.codecs.gzip.GzipCodec` instead.
    """

    codec_name = "numcodecs.gzip"
    _codec_id = "gzip"
    codec_config: GZipConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> GZipJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> GZipJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> GZipJSON_V2 | GZipJSON_V3:
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
