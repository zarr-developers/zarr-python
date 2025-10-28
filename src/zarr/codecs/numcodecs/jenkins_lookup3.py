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
    check_codecjson_v2,
    check_named_required_config,
)


class JenkinsLookup3Config(TypedDict):
    """Configuration parameters for JenkinsLookup3 codec."""

    initval: int
    prefix: bytes | None


class JenkinsLookup3JSON_V2(JenkinsLookup3Config):
    """JSON representation of JenkinsLookup3 codec for Zarr V2."""

    id: ReadOnly[Literal["jenkins_lookup3"]]


class JenkinsLookup3JSON_V3(NamedRequiredConfig[Literal["jenkins_lookup3"], JenkinsLookup3Config]):
    """JSON representation of JenkinsLookup3 codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[JenkinsLookup3JSON_V2]:
    """
    A type guard for the Zarr V2 form of the JenkinsLookup3 codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "jenkins_lookup3"
        and "initval" in data
        and "prefix" in data
        and isinstance(data["initval"], int)  # type: ignore[typeddict-item]
        and isinstance(data["prefix"], bytes | None)  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[JenkinsLookup3JSON_V3]:
    """
    A type guard for the Zarr V3 form of the JenkinsLookup3 codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "jenkins_lookup3"
        and "initval" in data["configuration"]
        and "prefix" in data["configuration"]
        and isinstance(data["configuration"]["initval"], int)
        and isinstance(data["configuration"]["prefix"], bytes | None)
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if "configuration" in data and data["configuration"] == {}:
            data_copy = data_copy | {"configuration": {"initval": 0, "prefix": None}}
        if data.get("name") == "numcodecs.jenkins_lookup3":
            data_copy = data_copy | {"name": "jenkins_lookup3"}
        return data_copy  # type: ignore[return-value]
    return data


class JenkinsLookup3(_NumcodecsChecksumCodec):
    """
    A wrapper around the numcodecs.JenkinsLookup3 codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.jenkins_lookup3"
    _codec_id = "jenkins_lookup3"
    codec_config: JenkinsLookup3Config

    @overload
    def to_json(self, zarr_format: Literal[2]) -> JenkinsLookup3JSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> JenkinsLookup3JSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> JenkinsLookup3JSON_V2 | JenkinsLookup3JSON_V3:
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
