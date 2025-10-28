from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, TypedDict, TypeGuard, overload

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


class BZ2Config(TypedDict):
    level: int


class BZ2JSON_V2(BZ2Config):
    """JSON representation of BZ2 codec for Zarr V2."""

    id: ReadOnly[Literal["bz2"]]


class BZ2JSON_V3(NamedRequiredConfig[Literal["bz2"], BZ2Config]):
    """JSON representation of BZ2 codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[BZ2JSON_V2]:
    """
    A type guard for the Zarr V2 form of the BZ2 codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "bz2"
        and "level" in data
        and isinstance(data["level"], int)  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[BZ2JSON_V3]:
    """
    A type guard for the Zarr V3 form of the BZ2 codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "bz2"
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
        if data.get("name") == "numcodecs.bz2":
            data_copy = data_copy | {"name": "bz2"}
        return data_copy  # type: ignore[return-value]
    return data


class BZ2(_NumcodecsBytesBytesCodec):
    """
    A wrapper around the numcodecs.BZ2 codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.bz2"
    _codec_id = "bz2"
    codec_config: BZ2Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BZ2JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BZ2JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> BZ2JSON_V2 | BZ2JSON_V3:
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
