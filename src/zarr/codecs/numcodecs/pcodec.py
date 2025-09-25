from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NotRequired, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsArrayBytesCodec,
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


class PCodecConfig(TypedDict):
    """Configuration parameters for PCodec codec."""

    level: NotRequired[int]
    delta_encoding_order: NotRequired[int]


class PCodecJSON_V2(PCodecConfig):
    """JSON representation of PCodec codec for Zarr V2."""

    id: ReadOnly[Literal["pcodec"]]


class PCodecJSON_V3(NamedRequiredConfig[Literal["pcodec"], PCodecConfig]):
    """JSON representation of PCodec codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[PCodecJSON_V2]:
    """
    A type guard for the Zarr V2 form of the PCodec codec JSON
    """
    return (
        _check_codecjson_v2(data)
        and data["id"] == "pcodec"
        and ("level" not in data or isinstance(data["level"], int))
        and ("delta_encoding_order" not in data or isinstance(data["delta_encoding_order"], int))
    )


def check_json_v3(data: object) -> TypeGuard[PCodecJSON_V3]:
    """
    A type guard for the Zarr V3 form of the PCodec codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] == "pcodec"
        and (
            "configuration" not in data
            or (
                (
                    "level" not in data["configuration"]
                    or isinstance(data["configuration"]["level"], int)
                )
                and (
                    "delta_encoding_order" not in data["configuration"]
                    or isinstance(data["configuration"]["delta_encoding_order"], int)
                )
            )
        )
    )


class PCodec(_NumcodecsArrayBytesCodec):
    """
    A wrapper around the numcodecs.PCodec codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.pcodec"
    _codec_id = "pcodec"
    codec_config: PCodecConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> PCodecJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> PCodecJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> PCodecJSON_V2 | PCodecJSON_V3:
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
