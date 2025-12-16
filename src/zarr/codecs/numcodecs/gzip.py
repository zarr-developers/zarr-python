from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, overload

from typing_extensions import deprecated

from zarr.codecs.numcodecs._codecs import _NumcodecsBytesBytesCodec
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    ZarrFormat,
    check_codecjson_v2,
    check_named_required_config,
)
from zarr.errors import ZarrDeprecationWarning

if TYPE_CHECKING:
    from zarr.codecs.gzip import GZipConfig, GZipJSON_V2, GZipJSON_V3


def check_json_v2(data: object) -> TypeGuard[GZipJSON_V2]:
    """
    A type guard for the Zarr V2 form of the GZip codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "gzip"
        and "level" in data
        and isinstance(data["level"], int)  # type: ignore[typeddict-item]
        and 0 <= data["level"] <= 9  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[GZipJSON_V3]:
    """
    A type guard for the Zarr V3 form of the GZip codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "gzip"
        and "level" in data["configuration"]
        and isinstance(data["configuration"]["level"], int)
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if "configuration" in data and data["configuration"] == {}:
            data_copy = data_copy | {"configuration": {"level": 1}}
        if data.get("name") == "numcodecs.gzip":
            data_copy = data_copy | {"name": "gzip"}
        return data_copy  # type: ignore[return-value]
    return data


@deprecated("Use `zarr.codecs.GzipCodec` instead.", category=ZarrDeprecationWarning)
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
        data = _handle_json_alias_v3(data)
        if check_json_v3(data):
            config = data["configuration"]
            return cls(**config)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
