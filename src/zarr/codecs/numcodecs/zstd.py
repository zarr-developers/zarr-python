from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, overload

from zarr.codecs.numcodecs._codecs import _NumcodecsBytesBytesCodec
from zarr.codecs.zstd import ZstdConfig_V3
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedRequiredConfig,
    _check_codecjson_v2,
    _check_codecjson_v3,
)

if TYPE_CHECKING:
    from zarr.codecs.zstd import ZstdJSON_V2, ZstdJSON_V3
    from zarr.core.common import ZarrFormat


def check_json_v2(data: object) -> TypeGuard[ZstdJSON_V2]:
    """
    A type guard for the Zarr V2 form of the Zstd codec JSON
    """
    return (
        _check_codecjson_v2(data)
        and data["id"] == "zstd"
        and "level" in data
        and isinstance(data["level"], int)
        and isinstance(data.get("checksum", False), bool)
    )


class ZstdJSON_V3_Legacy(NamedRequiredConfig[Literal["numcodecs.zstd"], ZstdConfig_V3]):
    """
    The JSON form of the ZStandard codec in Zarr v3.
    """


def check_json_v3(data: object) -> TypeGuard[ZstdJSON_V3 | ZstdJSON_V3_Legacy]:
    """
    A type guard for the Zarr V3 form of the Zstd codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] in ("zstd", "numcodecs.zstd")
        and "configuration" in data
        and "level" in data["configuration"]
        and "checksum" in data["configuration"]
        and isinstance(data["configuration"]["level"], int)
        and isinstance(data["configuration"]["checksum"], bool)
    )


class Zstd(_NumcodecsBytesBytesCodec):
    """
    A legacy wrapper used to provide a Zarr V3 API for the numcodecs zstd codec.

    Use `zarr.codecs.zstd.ZStdCodec` instead.
    """

    codec_name = "numcodecs.zstd"
    _codec_id = "zstd"
    codec_config: ZstdConfig_V3

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ZstdJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> ZstdJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> ZstdJSON_V2 | ZstdJSON_V3:
        res = super().to_json(zarr_format)
        if zarr_format == 2 and not res.get("checksum", False):  # type: ignore[union-attr]
            # https://github.com/zarr-developers/zarr-python/pull/2655
            res.pop("checksum")  # type: ignore[union-attr, typeddict-item]
        return res  # type: ignore[return-value]

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
