from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, TypedDict, TypeGuard, overload

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
    _check_codecjson_v2,
    _check_codecjson_v3,
)


class Fletcher32Config(TypedDict):
    """Configuration parameters for Fletcher32 codec."""


class Fletcher32JSON_V2(Fletcher32Config):
    """JSON representation of Fletcher32 codec for Zarr V2."""

    id: ReadOnly[Literal["fletcher32"]]


class Fletcher32JSON_V3_Legacy(
    NamedRequiredConfig[Literal["numcodecs.fletcher32"], Fletcher32Config]
):
    """Legacy JSON representation of Fletcher32 codec for Zarr V3."""


class Fletcher32JSON_V3(NamedRequiredConfig[Literal["fletcher32"], Fletcher32Config]):
    """JSON representation of Fletcher32 codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[Fletcher32JSON_V2]:
    """
    A type guard for the Zarr V2 form of the Fletcher32 codec JSON
    """
    return _check_codecjson_v2(data) and data["id"] == "fletcher32"


def check_json_v3(data: object) -> TypeGuard[Fletcher32JSON_V3 | Fletcher32JSON_V3_Legacy]:
    """
    A type guard for the Zarr V3 form of the Fletcher32 codec JSON
    """
    return (
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] in ("fletcher32", "numcodecs.fletcher32")
        and ("configuration" not in data or data["configuration"] == {})
    )


class Fletcher32(_NumcodecsChecksumCodec):
    """
    A wrapper around the numcodecs.Fletcher32 codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.fletcher32"
    _codec_id = "fletcher32"
    codec_config: Fletcher32Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Fletcher32JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Fletcher32JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Fletcher32JSON_V2 | Fletcher32JSON_V3:
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
