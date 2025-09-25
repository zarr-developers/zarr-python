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


class ZFPYConfig(TypedDict):
    """Configuration parameters for ZFPY codec."""

    mode: NotRequired[int]
    rate: NotRequired[float]
    precision: NotRequired[int]
    tolerance: NotRequired[float]


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
        _check_codecjson_v2(data)
        and data["id"] == "zfpy"
        and ("mode" not in data or isinstance(data["mode"], int))
        and ("rate" not in data or isinstance(data["rate"], (int, float)))
        and ("precision" not in data or isinstance(data["precision"], int))
        and ("tolerance" not in data or isinstance(data["tolerance"], (int, float)))
    )


def check_json_v3(data: object) -> TypeGuard[ZFPYJSON_V3]:
    """
    A type guard for the Zarr V3 form of the ZFPY codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] == "zfpy"
        and (
            "configuration" not in data
            or (
                (
                    "mode" not in data["configuration"]
                    or isinstance(data["configuration"]["mode"], int)
                )
                and (
                    "rate" not in data["configuration"]
                    or isinstance(data["configuration"]["rate"], (int, float))
                )
                and (
                    "precision" not in data["configuration"]
                    or isinstance(data["configuration"]["precision"], int)
                )
                and (
                    "tolerance" not in data["configuration"]
                    or isinstance(data["configuration"]["tolerance"], (int, float))
                )
            )
        )
    )


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
        if check_json_v3(data):
            config = data["configuration"]
            return cls(**config)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if _check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
