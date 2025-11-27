from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsArrayArrayCodec,
    _warn_unstable_specification,
)
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedConfig,
    ZarrFormat,
    check_codecjson_v2,
    product,
)
from zarr.core.dtype.npy.bool import Bool
from zarr.dtype import UInt8, ZDType

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec


class PackBitsConfig(TypedDict):
    pass  # PackBits has no configuration parameters


class PackBitsJSON_V2(PackBitsConfig):
    """JSON representation of PackBits codec for Zarr V2."""

    id: ReadOnly[Literal["packbits"]]


PackBitsJSON_V3 = Literal["packbits"]
"""Serialized JSON representation of PackBits codec for Zarr V3."""

PackBitsJSON_V3_Read = NamedConfig[Literal["packbits"], PackBitsConfig] | Literal["packbits"]
"""Readable JSON representation of PackBits codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[PackBitsJSON_V2]:
    """
    A type guard for the Zarr V2 form of the PackBits codec JSON
    """
    return check_codecjson_v2(data) and data["id"] == "packbits"


def check_json_v3(data: object) -> TypeGuard[PackBitsJSON_V3_Read]:
    """
    A type guard for the readable Zarr V3 form of the PackBits codec JSON
    """
    return data in ("packbits", {"name": "packbits"}, {"name": "packbits", "configuration": {}})


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if data.get("name") == "numcodecs.packbits":
            data_copy = data_copy | {"name": "packbits"}
        return data_copy  # type: ignore[return-value]
    return data


class PackBits(_NumcodecsArrayArrayCodec):
    """
    A wrapper around the numcodecs.PackBits codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.packbits"
    _codec_id = "packbits"
    codec_config: PackBitsConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> PackBitsJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> PackBitsJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> PackBitsJSON_V2 | PackBitsJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return replace(
            chunk_spec,
            shape=(1 + math.ceil(product(chunk_spec.shape) / 8),),
            dtype=UInt8(),
        )

    # todo: remove this type: ignore when this class can be defined w.r.t.
    # a single zarr dtype API
    def validate(self, *, dtype: ZDType[Any, Any], **_kwargs: Any) -> None:
        if not isinstance(dtype, Bool):
            raise ValueError(f"Packbits filter requires bool dtype. Got {dtype}.")  # noqa: TRY004

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        data = _handle_json_alias_v3(data)
        if check_json_v3(data):
            return cls()
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
