from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, TypedDict, TypeGuard, overload

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
    check_codecjson_v2,
    check_named_required_config,
)


class CompressionKwargs(TypedDict):
    tolerance: int


class ZFPYConfig(TypedDict):
    """Configuration parameters for ZFPY codec."""

    mode: int
    compression_kwargs: CompressionKwargs
    tolerance: int
    rate: int
    precision: int


class ZFPYJSON_V2(ZFPYConfig):
    """JSON representation of ZFPY codec for Zarr V2."""

    id: ReadOnly[Literal["zfpy"]]


class ZFPYJSON_V3(NamedRequiredConfig[Literal["zfpy"], ZFPYConfig]):
    """JSON representation of ZFPY codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[ZFPYJSON_V2]:
    """
    A type guard for the Zarr V2 form of the ZFPY codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "zfpy"
        and ("mode" not in data or isinstance(data["mode"], int))  # type: ignore[typeddict-item]
        and ("rate" not in data or isinstance(data["rate"], (int, float)))  # type: ignore[typeddict-item]
        and ("precision" not in data or isinstance(data["precision"], int))  # type: ignore[typeddict-item]
        and ("tolerance" not in data or isinstance(data["tolerance"], (int, float)))  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[ZFPYJSON_V3]:
    """
    A type guard for the Zarr V3 form of the ZFPY codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "zfpy"
        and set(data["configuration"].keys())
        == {"mode", "compression_kwargs", "tolerance", "rate", "precision"}
        and isinstance(data["configuration"]["mode"], int)
        and isinstance(data["configuration"]["compression_kwargs"], Mapping)
        and isinstance(data["configuration"]["tolerance"], int)
        and isinstance(data["configuration"]["rate"], int)
        and isinstance(data["configuration"]["precision"], int)
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if "configuration" in data and data["configuration"] == {}:
            data_copy = data_copy | {
                "configuration": {
                    "mode": 4,
                    "tolerance": -1,
                    "rate": -1,
                    "precision": -1,
                    "compression_kwargs": {"tolerance": -1},
                }
            }
        if data.get("name") == "numcodecs.zfpy":
            data_copy = data_copy | {"name": "zfpy"}
        return data_copy  # type: ignore[return-value]
    return data


class ZFPY(_NumcodecsArrayBytesCodec):
    """
    A wrapper around the numcodecs.ZFPY codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.zfpy"
    _codec_id = "zfpy"
    codec_config: ZFPYConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ZFPYJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> ZFPYJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> ZFPYJSON_V2 | ZFPYJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        data = _handle_json_alias_v3(data)
        if check_json_v3(data):
            config = data["configuration"]
            return cls(**config)  # type: ignore[arg-type]
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
