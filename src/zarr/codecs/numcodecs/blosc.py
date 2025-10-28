from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, overload

from typing_extensions import deprecated

from zarr.codecs.numcodecs._codecs import _NumcodecsBytesBytesCodec
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    ZarrFormat,
    check_codecjson_v2,
    check_named_required_config,
)
from zarr.errors import ZarrDeprecationWarning

if TYPE_CHECKING:
    from zarr.codecs.blosc import BloscJSON_V2, BloscJSON_V3


def check_json_v3(data: object) -> TypeGuard[BloscJSON_V3]:
    return (
        check_named_required_config(data)
        and data["name"] == "blosc"
        and set(data["configuration"].keys()).issuperset(
            {"cname", "clevel", "shuffle", "blocksize"}
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
                "configuration": {"cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}
            }
        if data.get("name") == "numcodecs.blosc":
            data_copy = data_copy | {"name": "blosc"}

        return data_copy  # type: ignore[return-value]
    return data


@deprecated("Use `zarr.codecs.BloscCodec` instead.", category=ZarrDeprecationWarning)
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
    def to_json(self, zarr_format: Literal[3]) -> BloscJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> BloscJSON_V2 | BloscJSON_V3:
        if zarr_format == 3:
            # Add typesize for v3 format (required by official blosc codec)
            config = dict(self.codec_config)
            config.pop("id", None)
            config["typesize"] = 1  # Default typesize for numcodecs blosc
            return {"name": "blosc", "configuration": config}  # type: ignore[typeddict-item]
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
