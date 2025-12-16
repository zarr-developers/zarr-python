from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self, TypedDict, TypeGuard, overload

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
    check_codecjson_v2,
    check_named_required_config,
)


class LZMAConfig(TypedDict):
    format: int
    check: int
    preset: int | None
    filters: list[dict[str, Any]] | None


class LZMAJSON_V2(LZMAConfig):
    """JSON representation of LZMA codec for Zarr V2."""

    id: ReadOnly[Literal["lzma"]]


class LZMAJSON_V3(NamedRequiredConfig[Literal["lzma"], LZMAConfig]):
    """JSON representation of LZMA codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[LZMAJSON_V2]:
    """
    A type guard for the Zarr V2 form of the LZMA codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "lzma"
        and "format" in data
        and "check" in data
        and isinstance(data["format"], int)  # type: ignore[typeddict-item]
        and isinstance(data["check"], int)  # type: ignore[typeddict-item]
        and ("preset" not in data or data["preset"] is None or isinstance(data["preset"], int))  # type: ignore[typeddict-item]
        and ("filters" not in data or data["filters"] is None or isinstance(data["filters"], list))  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[LZMAJSON_V3]:
    """
    A type guard for the Zarr V3 form of the LZMA codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "lzma"
        and "format" in data["configuration"]
        and "check" in data["configuration"]
        and isinstance(data["configuration"]["format"], int)
        and isinstance(data["configuration"]["check"], int)
        and (
            "preset" not in data["configuration"]
            or data["configuration"]["preset"] is None
            or isinstance(data["configuration"]["preset"], int)
        )
        and (
            "filters" not in data["configuration"]
            or data["configuration"]["filters"] is None
            or isinstance(data["configuration"]["filters"], list)
        )
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if "configuration" in data and data["configuration"] == {}:
            data_copy = data_copy | {
                "configuration": {"format": 1, "check": -1, "preset": None, "filters": None}
            }
        if data.get("name") == "numcodecs.lzma":
            data_copy = data_copy | {"name": "lzma"}
        return data_copy  # type: ignore[return-value]
    return data


class LZMA(_NumcodecsBytesBytesCodec):
    """
    A wrapper around the numcodecs.LZMA codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.lzma"
    _codec_id = "lzma"
    codec_config: LZMAConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> LZMAJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> LZMAJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> LZMAJSON_V2 | LZMAJSON_V3:
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
