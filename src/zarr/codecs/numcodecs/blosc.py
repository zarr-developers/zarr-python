from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, overload

from zarr.codecs.numcodecs._codecs import _NumcodecsBytesBytesCodec
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedRequiredConfig,
    ZarrFormat,
    _check_codecjson_v2,
)

if TYPE_CHECKING:
    from zarr.codecs.blosc import BloscCname_Lit, BloscJSON_V2


# This is different from zarr.codecs.blosc
class BloscConfigV3_Legacy(TypedDict):
    cname: BloscCname_Lit
    clevel: int
    shuffle: int
    blocksize: int


class BloscJSON_V3_Legacy(NamedRequiredConfig[Literal["numcodecs.blosc"], BloscConfigV3_Legacy]):
    """
    Legacy JSON form of the Blosc codec in Zarr V3.
    """


def check_json_v3(data: object) -> TypeGuard[BloscJSON_V3_Legacy]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) == {"name", "configuration"}
        and data["name"] in ("blosc", "numcodecs.blosc")
        and isinstance(data["configuration"], Mapping)
        and set(data["configuration"].keys()) == {"cname", "clevel", "shuffle", "blocksize"}
    )


class Blosc(_NumcodecsBytesBytesCodec):
    """
    A legacy wrapper used to provide a Zarr V3 API for the numcodecs blosc codec.

    Use `zarr.codecs.blosc.BloscCodec` instead.
    """

    codec_name = "numcodecs.blosc"
    _codec_id = "blosc"

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BloscJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BloscJSON_V3_Legacy: ...
    def to_json(self, zarr_format: ZarrFormat) -> BloscJSON_V2 | BloscJSON_V3_Legacy:
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
