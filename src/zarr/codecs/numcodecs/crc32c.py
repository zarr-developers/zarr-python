from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Self, TypeGuard, overload

from typing_extensions import deprecated

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsChecksumCodec,
    _warn_unstable_specification,
)
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    check_codecjson_v2,
    check_named_config,
)
from zarr.errors import ZarrDeprecationWarning

if TYPE_CHECKING:
    from zarr.codecs.crc32c_ import Crc32cConfig_V2, Crc32cJSON_V2, Crc32cJSON_V3
    from zarr.core.common import ZarrFormat


def check_json_v2(data: object) -> TypeGuard[Crc32cJSON_V2]:
    """
    A type guard for the Zarr V2 form of the CRC32C codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "crc32c"
        and ("location" not in data or data["location"] in ("start", "end"))  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[Crc32cJSON_V3]:
    """
    A type guard for the Zarr V3 form of the CRC32C codec JSON
    """
    if data == "crc32c":
        return True
    return (
        check_named_config(data)
        and data["name"] == "crc32c"
        and data.get("configuration") in ({}, None)
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle JSON representation of the codec produced by numcodecs.zarr3 codec.
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if data.get("name") == "numcodecs.crc32c":
            data_copy = data_copy | {"name": "crc32c"}
        return data_copy  # type: ignore[return-value]
    return data


@deprecated("Use `zarr.codecs.CRC32CCodec` instead.", category=ZarrDeprecationWarning)
class CRC32C(_NumcodecsChecksumCodec):
    """
    A wrapper around the numcodecs.CRC32C codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.crc32c"
    _codec_id = "crc32c"
    codec_config: Crc32cConfig_V2

    @overload
    def to_json(self, zarr_format: Literal[2]) -> Crc32cJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> Crc32cJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> Crc32cJSON_V2 | Crc32cJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        data = _handle_json_alias_v3(data)
        if check_json_v3(data):
            if isinstance(data, str):
                return cls()
            config = data.get("configuration", {})
            return cls(**config)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
