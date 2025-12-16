from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, overload

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

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec


class ShuffleConfig(TypedDict):
    elementsize: int


class ShuffleJSON_V2(ShuffleConfig):
    """JSON representation of Shuffle codec for Zarr V2."""

    id: ReadOnly[Literal["shuffle"]]


class ShuffleJSON_V3(NamedRequiredConfig[Literal["shuffle"], ShuffleConfig]):
    """JSON representation of Shuffle codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[ShuffleJSON_V2]:
    """
    A type guard for the Zarr V2 form of the Shuffle codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "shuffle"
        and "elementsize" in data
        and isinstance(data["elementsize"], int)  # type: ignore[typeddict-item]
        and data["elementsize"] > 0  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[ShuffleJSON_V3]:
    """
    A type guard for the Zarr V3 form of the Shuffle codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "shuffle"
        and "elementsize" in data["configuration"]
        and isinstance(data["configuration"]["elementsize"], int)
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if "configuration" in data and data["configuration"] == {}:
            data_copy = data_copy | {"configuration": {"elementsize": 4}}
        if data.get("name") == "numcodecs.shuffle":
            data_copy = data_copy | {"name": "shuffle"}
        return data_copy  # type: ignore[return-value]
    return data


class Shuffle(_NumcodecsBytesBytesCodec):
    """
    A wrapper around the numcodecs.Shuffle codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.shuffle"
    _codec_id = "shuffle"
    codec_config: ShuffleConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ShuffleJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> ShuffleJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> ShuffleJSON_V2 | ShuffleJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if self.codec_config.get("elementsize") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return type(self)(**{**self.codec_config, "elementsize": dtype.itemsize})
        return self  # pragma: no cover

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
