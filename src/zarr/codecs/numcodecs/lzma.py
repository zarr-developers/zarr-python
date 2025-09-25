from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, TypeGuard, overload

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


class LZMAConfig(TypedDict):
    format: int
    check: int
    preset: int | None
    filters: list[dict[str, Any]] | None


class LZMAJSON_V2(LZMAConfig):
    """JSON representation of LZMA codec for Zarr V2."""

    id: ReadOnly[Literal["lzma"]]


class LZMAJSON_V3_Legacy(NamedRequiredConfig[Literal["lzma"], LZMAConfig]):
    """Legacy JSON representation of LZMA codec for Zarr V3."""

class LZMAJSON_V3(NamedRequiredConfig[Literal["lzma"], LZMAConfig]):
    """JSON representation of LZMA codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[LZMAJSON_V2]:
    """
    A type guard for the Zarr V2 form of the LZMA codec JSON
    """
    return (
        _check_codecjson_v2(data)
        and data["id"] == "lzma"
        and "format" in data
        and "check" in data
        and isinstance(data["format"], int)
        and isinstance(data["check"], int)
        and ("preset" not in data or data["preset"] is None or isinstance(data["preset"], int))
        and ("filters" not in data or data["filters"] is None or isinstance(data["filters"], list))
    )


def check_json_v3(data: object) -> TypeGuard[LZMAJSON_V3 | LZMAJSON_V3_Legacy]:
    """
    A type guard for the Zarr V3 form of the LZMA codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] in ("lzma", "numcodecs.lzma")
        and "configuration" in data
        and "format" in data["configuration"]
        and "check" in data["configuration"]
        and isinstance(data["configuration"]["format"], int)
        and isinstance(data["configuration"]["check"], int)
        and ("preset" not in data["configuration"] or data["configuration"]["preset"] is None or isinstance(data["configuration"]["preset"], int))
        and ("filters" not in data["configuration"] or data["configuration"]["filters"] is None or isinstance(data["configuration"]["filters"], list))
    )


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
        if check_json_v3(data):
            config = data["configuration"]
            return cls(**config)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if _check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
