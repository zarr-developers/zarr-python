from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, Self, TypedDict, TypeGuard, overload

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
from zarr.core.dtype.common import check_dtype_spec_v3

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec


class QuantizeConfig(TypedDict):
    digits: int
    dtype: NotRequired[str]


class QuantizeJSON_V2(QuantizeConfig):
    """JSON representation of Quantize codec for Zarr V2."""

    id: ReadOnly[Literal["quantize"]]


class QuantizeJSON_V3(NamedRequiredConfig[Literal["quantize"], QuantizeConfig]):
    """JSON representation of Quantize codec for Zarr V3."""


def check_json_v2(data: object) -> TypeGuard[QuantizeJSON_V2]:
    """
    A type guard for the Zarr V2 form of the Quantize codec JSON
    """
    return (
        check_codecjson_v2(data)
        and data["id"] == "quantize"
        and "digits" in data
        and isinstance(data["digits"], int)  # type: ignore[typeddict-item]
        and data["digits"] > 0  # type: ignore[typeddict-item]
        and ("dtype" not in data or isinstance(data["dtype"], str))  # type: ignore[typeddict-item]
    )


def check_json_v3(data: object) -> TypeGuard[QuantizeJSON_V3]:
    """
    A type guard for the Zarr V3 form of the Quantize codec JSON
    """
    return (
        check_named_required_config(data)
        and data["name"] == "quantize"
        and "digits" in data["configuration"]
        and isinstance(data["configuration"]["digits"], int)
        and "dtype" in data["configuration"]
        and check_dtype_spec_v3(data["configuration"]["dtype"])
    )


class Quantize(_NumcodecsArrayArrayCodec):
    """
    A wrapper around the numcodecs.Quantize codec that provides Zarr V3 compatibility.

    This class does not have a stable API.
    """

    codec_name = "numcodecs.quantize"
    _codec_id = "quantize"
    codec_config: QuantizeConfig

    @overload
    def to_json(self, zarr_format: Literal[2]) -> QuantizeJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> QuantizeJSON_V3: ...
    def to_json(self, zarr_format: ZarrFormat) -> QuantizeJSON_V2 | QuantizeJSON_V3:
        _warn_unstable_specification(self)
        return super().to_json(zarr_format)  # type: ignore[return-value]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        if self.codec_config.get("dtype") is None:
            dtype = array_spec.dtype.to_native_dtype()
            return type(self)(**{**self.codec_config, "dtype": str(dtype)})
        return self

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
        if check_codecjson_v2(data):
            return cls._from_json_v2(data)
        return cls._from_json_v3(data)
