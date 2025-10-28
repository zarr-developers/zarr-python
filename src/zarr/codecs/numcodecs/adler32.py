from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsChecksumCodec,
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

if TYPE_CHECKING:
    ...


class Adler32Config(TypedDict, total=False):
    """Configuration parameters for Adler32 codec."""

    location: Literal["start", "end"]


class Adler32JSON_V2(Adler32Config):
    """JSON representation of Adler32 codec for Zarr V2."""

    id: ReadOnly[Literal["adler32"]]


class Adler32JSON_V3(NamedRequiredConfig[Literal["adler32"], Adler32Config]):
    """JSON representation of Adler32 codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[Adler32JSON_V2]:
    """
    A type guard for the Zarr V2 form of the Adler32 codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "adler32"
        and ("location" not in data or data["location"] in ("start", "end"))  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[Adler32JSON_V3]:
    """
    A type guard for the Zarr V3 form of the Adler32 codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "adler32"
        and (
            "location" not in data["configuration"]
            or data["configuration"]["location"] in ("start", "end")
        )
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if data.get("name") == "numcodecs.adler32":
            data_copy = data_copy | {"name": "adler32"}
        return data_copy  # type: ignore[return-value]
    return data


class Adler32(_NumcodecsChecksumCodec):
    """
    A wrapper around the numcodecs.Adler32 codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.adler32"
    _codec_id = "adler32"
    codec_config: Adler32Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Adler32JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Adler32JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Adler32JSON_V2 | Adler32JSON_V3:
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
