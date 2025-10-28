from __future__ import annotations

from typing import Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsArrayArrayCodec,
    _warn_unstable_specification,
)
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedRequiredConfig,
    ZarrFormat,
    check_codecjson_v2,
    check_named_required_config,
)


class BitRoundConfig(TypedDict):
    keepbits: int


class BitRoundJSON_V2(BitRoundConfig):
    """JSON representation of BitRound codec for Zarr V2."""

    id: ReadOnly[Literal["bitround"]]


class BitRoundJSON_V3(NamedRequiredConfig[Literal["bitround"], BitRoundConfig]):
    """JSON representation of BitRound codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[BitRoundJSON_V2]:
    """
    A type guard for the Zarr V2 form of the BitRound codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "bitround"
        and "keepbits" in data
        and isinstance(data["keepbits"], int)  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[BitRoundJSON_V3]:
    """
    A type guard for the Zarr V3 form of the BitRound codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "bitround"
        and "keepbits" in data["configuration"]
        and isinstance(data["configuration"]["keepbits"], int)
    )


class BitRound(_NumcodecsArrayArrayCodec):
    """
    A wrapper around the numcodecs.BitRound codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.bitround"
    _codec_id = "bitround"
    codec_config: BitRoundConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BitRoundJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BitRoundJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> BitRoundJSON_V2 | BitRoundJSON_V3:
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
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
