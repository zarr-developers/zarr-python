from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import (
    _NumcodecsArrayArrayCodec,
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
from zarr.core.dtype.common import check_dtype_name_v2, check_dtype_spec_v3
from zarr.dtype import parse_dtype

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.dtype.common import DTypeName_V2, DTypeSpec_V3


class DeltaConfig_V2(TypedDict):
    dtype: DTypeName_V2
    astype: DTypeName_V2


class DeltaConfig_V3(TypedDict):
    dtype: DTypeSpec_V3
    astype: DTypeSpec_V3


class DeltaJSON_V2(DeltaConfig_V2):
    """JSON representation of Delta codec for Zarr V2."""

    id: ReadOnly[Literal["delta"]]


class DeltaJSON_V3(NamedRequiredConfig[Literal["delta"], DeltaConfig_V3]):
    """JSON representation of Delta codec for Zarr V3."""


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle JSON representation of the codec produced by legacy numcodecs.zarr3 codecs.
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if data.get("name") == "numcodecs.delta":
            data_copy = data_copy | {"name": "delta"}
        return data_copy  # type: ignore[return-value]
    return data


def check_json_v2(data: object) -> TypeGuard[DeltaJSON_V2]:
    """
    A type guard for the Zarr V2 form of the Delta codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "delta"
        and "astype" in data
        and "dtype" in data
        and check_dtype_name_v2(data["dtype"])  # type: ignore[typeddict-item]
        and check_dtype_name_v2(data["astype"])  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[DeltaJSON_V3]:
    """
    A type guard for the Zarr V3 form of the Delta codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "delta"
        and "astype" in data["configuration"]
        and "dtype" in data["configuration"]
        and check_dtype_spec_v3(data["configuration"]["dtype"])
        and check_dtype_spec_v3(data["configuration"]["astype"])
    )


class Delta(_NumcodecsArrayArrayCodec):
    """
    A wrapper around the numcodecs.Delta codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.delta"
    _codec_id = "delta"
    codec_config: DeltaJSON_V2

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        data = _handle_json_alias_v3(data)
        if check_json_v3(data):
            config = data["configuration"]
            astype = parse_dtype(config["astype"], zarr_format=3).to_json(zarr_format=2)["name"]
            dtype = parse_dtype(config["dtype"], zarr_format=3).to_json(zarr_format=2)["name"]

            return cls(astype=astype, dtype=dtype)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> DeltaJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> DeltaJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> DeltaJSON_V2 | DeltaJSON_V3:
        _warn_unstable_specification(self)
        if zarr_format == 2:
            return self.codec_config
        conf = self.codec_config
        astype_v3 = parse_dtype(conf["astype"], zarr_format=2).to_json(zarr_format=3)
        dtype_v3 = parse_dtype(conf["dtype"], zarr_format=2).to_json(zarr_format=3)
        return {
            "name": "delta",
            "configuration": {"astype": astype_v3, "dtype": dtype_v3},
        }

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            dtype = parse_dtype(astype, zarr_format=3)
            return replace(chunk_spec, dtype=dtype)
        return chunk_spec
