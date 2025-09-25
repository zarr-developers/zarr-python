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
    _check_codecjson_v2,
    _check_codecjson_v3,
)

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec


class ShuffleConfig(TypedDict):
    elementsize: int


class ShuffleJSON_V2(ShuffleConfig):
    """JSON representation of Shuffle codec for Zarr V2."""

    id: ReadOnly[Literal["shuffle"]]


class ShuffleJSON_V3_Legacy(NamedRequiredConfig[Literal["numcodecs.shuffle"], ShuffleConfig]):
    """JSON representation of Shuffle codec for Zarr V3."""


class ShuffleJSON_V3(NamedRequiredConfig[Literal["shuffle"], ShuffleConfig]):
    """JSON representation of Shuffle codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[ShuffleJSON_V2]:
    """
    A type guard for the Zarr V2 form of the Shuffle codec JSON
    """
    return (
        _check_codecjson_v2(data)
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
        _check_codecjson_v3(data)
        and isinstance(data, Mapping)
        and data["name"] in ("numcodecs.shuffle", "shuffle")
        and "configuration" in data
        and "elementsize" in data["configuration"]
        and isinstance(data["configuration"]["elementsize"], int)
        and data["configuration"]["elementsize"] > 0
    )


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
        if check_json_v3(data):
            config = data["configuration"]
            return cls(**config)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if _check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
