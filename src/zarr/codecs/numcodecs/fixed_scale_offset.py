from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import _NumcodecsArrayArrayCodec, _warn_unstable_specification
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedRequiredConfig,
    check_codecjson_v2,
    check_named_required_config,
)
from zarr.core.dtype import parse_dtype
from zarr.core.dtype.common import check_dtype_name_v2, check_dtype_spec_v3

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.common import DTypeName_V2, DTypeSpec_V3


class FixedScaleOffsetConfig_V2(TypedDict):
    dtype: DTypeName_V2
    astype: DTypeName_V2
    scale: float
    offset: float


class FixedScaleOffsetConfig_V3(TypedDict):
    dtype: DTypeSpec_V3
    astype: DTypeSpec_V3
    scale: float
    offset: float


class FixedScaleOffsetJSON_V2(FixedScaleOffsetConfig_V2):
    """JSON representation of FixedScaleOffset codec for Zarr V2."""

    id: ReadOnly[Literal["fixedscaleoffset"]]


class FixedScaleOffsetJSON_V3(
    NamedRequiredConfig[Literal["fixedscaleoffset"], FixedScaleOffsetConfig_V3]
):
    """JSON representation of FixedScaleOffset codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[FixedScaleOffsetJSON_V2]:
    """
    A type guard for the Zarr V2 form of the Delta codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "fixedscaleoffset"
        and "scale" in data
        and "offset" in data
        and "dtype" in data
        and "astype" in data
        and check_dtype_name_v2(data["dtype"])  # type: ignore[typeddict-item]
        and check_dtype_name_v2(data["astype"])  # type: ignore[typeddict-item]
    )


def check_json_v3(
    data: object,
) -> TypeGuard[FixedScaleOffsetJSON_V3]:
    """
    A type guard for the Zarr V3 form of the Delta codec JSON

    We accept the Zarr V2 data type spec for backwards compatibility.
    """
    return (
        check_named_required_config(data)
        and data["name"] == "fixedscaleoffset"
        and set(data["configuration"].keys()) == {"astype", "dtype", "scale", "offset"}
        and check_dtype_spec_v3(data["configuration"]["dtype"])
        and check_dtype_spec_v3(data["configuration"]["astype"])
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle JSON representation of the codec produced by legacy numcodecs.zarr3 codecs.
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if data.get("name") == "numcodecs.fixedscaleoffset":
            data_copy = data_copy | {"name": "fixedscaleoffset"}
            if isinstance(data_copy.get("configuration"), Mapping):
                config_copy: dict[str, object] = dict(data_copy["configuration"])  # type: ignore[call-overload]
                if "dtype" in config_copy:
                    dtype_v2 = config_copy["dtype"]
                    if check_dtype_name_v2(dtype_v2):
                        config_copy["dtype"] = parse_dtype(dtype_v2, zarr_format=2).to_json(
                            zarr_format=3
                        )
                if "astype" in config_copy:
                    astype_v2 = config_copy["astype"]
                    if check_dtype_name_v2(astype_v2):
                        config_copy["astype"] = parse_dtype(astype_v2, zarr_format=2).to_json(
                            zarr_format=3
                        )
                data_copy["configuration"] = config_copy
        return data_copy  # type: ignore[return-value]
    return data


class FixedScaleOffset(_NumcodecsArrayArrayCodec):
    """
    A wrapper around the numcodecs.FixedScaleOffset codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.fixedscaleoffset"
    _codec_id = "fixedscaleoffset"
    codec_config: FixedScaleOffsetConfig_V2

    @overload
    def to_json(self, zarr_format: Literal[2]) -> FixedScaleOffsetJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> FixedScaleOffsetJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> FixedScaleOffsetJSON_V2 | FixedScaleOffsetJSON_V3:
        _warn_unstable_specification(self)
        if zarr_format == 2:
            return super().to_json(zarr_format)  # type: ignore[return-value]
        return {
            "name": "fixedscaleoffset",
            "configuration": {
                "astype": parse_dtype(self.codec_config["astype"], zarr_format=2).to_json(
                    zarr_format=3
                ),
                "dtype": parse_dtype(self.codec_config["dtype"], zarr_format=2).to_json(
                    zarr_format=3
                ),
                "scale": self.codec_config["scale"],
                "offset": self.codec_config["offset"],
            },
        }

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
            return cls(astype=astype, dtype=dtype, scale=config["scale"], offset=config["offset"])
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        if astype := self.codec_config.get("astype"):
            dtype = parse_dtype(astype, zarr_format=3)
            return replace(chunk_spec, dtype=dtype)
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        dtype = array_spec.dtype.to_json(zarr_format=2)["name"]
        return type(self)(**{**self.codec_config, "dtype": dtype})
