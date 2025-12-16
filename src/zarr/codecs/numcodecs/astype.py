from collections.abc import Mapping
from dataclasses import replace
from typing import Literal, Self, TypedDict, TypeGuard, overload

from typing_extensions import ReadOnly

from zarr.codecs.numcodecs._codecs import _NumcodecsArrayArrayCodec, _warn_unstable_specification
from zarr.core.array_spec import ArraySpec
from zarr.core.common import (
    CodecJSON,
    CodecJSON_V2,
    CodecJSON_V3,
    NamedRequiredConfig,
    ZarrFormat,
    check_codecjson_v2,
    check_named_required_config,
)
from zarr.core.dtype import parse_dtype
from zarr.core.dtype.common import (
    DTypeName_V2,
    DTypeSpec_V3,
    check_dtype_name_v2,
    check_dtype_spec_v3,
)


class AsTypeConfig_V2(TypedDict):
    encode_dtype: DTypeName_V2
    decode_dtype: DTypeName_V2


class AsTypeConfig_V3(TypedDict):
    encode_dtype: DTypeSpec_V3
    decode_dtype: DTypeSpec_V3


class AsTypeJSON_V2(AsTypeConfig_V2):
    """JSON representation of AsType codec for Zarr V2."""

    id: ReadOnly[Literal["astype"]]


class AsTypeJSON_V3(NamedRequiredConfig[Literal["astype"], AsTypeConfig_V3]):
    """JSON representation of AsType codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[AsTypeJSON_V2]:
    """
    A type guard for the Zarr V2 form of the Astype codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "astype"
        and "encode_dtype" in data
        and "decode_dtype" in data
        and check_dtype_name_v2(data["encode_dtype"])  # type: ignore[typeddict-item]
        and check_dtype_name_v2(data["decode_dtype"])  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[AsTypeJSON_V3]:
    """
    A type guard for the Zarr V3 form of the Astype codec JSON.
    """
    return (
        check_named_required_config(data)
        and data["name"] == "astype"
        and "encode_dtype" in data["configuration"]
        and "decode_dtype" in data["configuration"]
        and check_dtype_spec_v3(data["configuration"]["decode_dtype"])
        and check_dtype_spec_v3(data["configuration"]["encode_dtype"])
    )


def _handle_json_alias_v3(data: CodecJSON_V3) -> CodecJSON_V3:
    """
    Handle underspecified JSON representation of the codec produced by legacy code
    """
    if isinstance(data, Mapping):
        data_copy = dict(data)
        if data.get("name") == "numcodecs.astype":
            data_copy = data_copy | {"name": "astype"}
        return data_copy  # type: ignore[return-value]
    return data


class AsType(_NumcodecsArrayArrayCodec):
    """
    A wrapper around the numcodecs.Astype codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.astype"
    _codec_id = "astype"
    codec_config: AsTypeConfig_V2

    @overload
    def to_json(self, zarr_format: Literal[2]) -> AsTypeJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> AsTypeJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> AsTypeJSON_V2 | AsTypeJSON_V3:
        _warn_unstable_specification(self)
        if zarr_format == 2:
            return super().to_json(zarr_format)  # type: ignore[return-value]
        # For v3, we need to convert dtype format
        conf = self.codec_config
        encode_dtype_v3 = parse_dtype(conf["encode_dtype"], zarr_format=2).to_json(zarr_format=3)
        decode_dtype_v3 = parse_dtype(conf["decode_dtype"], zarr_format=2).to_json(zarr_format=3)
        return {
            "name": "astype",
            "configuration": {"encode_dtype": encode_dtype_v3, "decode_dtype": decode_dtype_v3},
        }

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        dtype = parse_dtype(self.codec_config["encode_dtype"], zarr_format=3)
        return replace(chunk_spec, dtype=dtype)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        dtype = array_spec.dtype.to_native_dtype()  # pragma: no cover
        return type(self)(**{**self.codec_config, "decode_dtype": str(dtype)})  # pragma: no cover

    @classmethod
    def _from_json_v2(cls, data: CodecJSON_V2) -> Self:
        return cls(**data)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON_V3) -> Self:
        data = _handle_json_alias_v3(data)
        if check_json_v3(data):
            config = data["configuration"]
            encode_dtype = parse_dtype(config["encode_dtype"], zarr_format=3).to_json(
                zarr_format=2
            )["name"]
            decode_dtype = parse_dtype(config["decode_dtype"], zarr_format=3).to_json(
                zarr_format=2
            )["name"]

            return cls(encode_dtype=encode_dtype, decode_dtype=decode_dtype)
        raise TypeError(f"Invalid JSON: {data}")

    @classmethod
    def from_json(cls, data: CodecJSON) -> Self:
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
