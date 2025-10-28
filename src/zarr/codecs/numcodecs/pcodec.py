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


class PCodecConfig(TypedDict):
    """Configuration parameters for PCodec codec."""

    level: int
    mode_spec: Literal["auto", "classic"]
    delta_spec: Literal["auto", "none", "try_consecutive", "try_lookback"]
    paging_spec: Literal["equal_pages_up_to"]
    delta_encoding_order: int | None
    equal_pages_up_to: int


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
        check_codecjson_v2(data)
        and data["id"] == "pcodec"
        and set(data.keys())
        == {
            "id",
            "level",
            "mode_spec",
            "delta_spec",
            "paging_spec",
            "equal_pages_up_to",
            "delta_encoding_order",
        }
        and isinstance(data["level"], int)  # type: ignore[typeddict-item]
        and isinstance(data["delta_encoding_order"], int)  # type: ignore[typeddict-item]
        and isinstance(data["mode_spec"], str)  # type: ignore[typeddict-item]
        and isinstance(data["delta_spec"], str)  # type: ignore[typeddict-item]
        and isinstance(data["paging_spec"], str)  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[PCodecJSON_V3]:
    """
    A type guard for the Zarr V3 form of the PCodec codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "pcodec"
        and isinstance(data["configuration"].get("level"), int)
        and isinstance(data["configuration"].get("delta_encoding_order"), int)
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
                    "level": 8,
                    "mode_spec": "auto",
                    "delta_spec": "auto",
                    "paging_spec": "equal_pages_up_to",
                    "delta_encoding_order": None,
                    "equal_pages_up_to": 262144,
                }
            }
        if data.get("name") == "numcodecs.pcodec":
            data_copy = data_copy | {"name": "pcodec"}
        return data_copy  # type: ignore[return-value]
    return data


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
