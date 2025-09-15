from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Self, TypedDict, TypeGuard, cast, overload

import numcodecs
from numcodecs.zstd import Zstd
from packaging.version import Version
from typing_extensions import ReadOnly

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, CodecJSON, NamedRequiredConfig, ZarrFormat
from zarr.errors import CodecValidationError

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


class ZstdConfig_V2(TypedDict):
    level: int


class ZstdConfig_V3(TypedDict):
    level: int
    checksum: bool


class ZstdJSON_V2(ZstdConfig_V2):
    """
    The JSON form of the ZStandard codec in Zarr v2.
    """

    id: ReadOnly[Literal["zstd"]]


class ZstdJSON_V3(NamedRequiredConfig[Literal["zstd"], ZstdConfig_V3]):
    """
    The JSON form of the ZStandard codec in Zarr v3.
    """


def check_json_v2(data: object) -> TypeGuard[ZstdJSON_V2]:
    return isinstance(data, Mapping) and set(data.keys()).issuperset({"id", "level"})


def check_json_v3(data: object) -> TypeGuard[ZstdJSON_V3]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) == {"name", "configuration"}
        and data["name"] == "zstd"
        and isinstance(data["configuration"], Mapping)
        and set(data["configuration"].keys()) == {"level", "checksum"}
    )


def parse_zstd_level(data: JSON) -> int:
    if isinstance(data, int):
        if data >= 23:
            raise ValueError(f"Value must be less than or equal to 22. Got {data} instead.")
        return data
    raise TypeError(f"Got value with type {type(data)}, but expected an int.")


def parse_checksum(data: JSON) -> bool:
    if isinstance(data, bool):
        return data
    raise TypeError(f"Expected bool. Got {type(data)}.")


@dataclass(frozen=True)
class ZstdCodec(BytesBytesCodec):
    is_fixed_size = True

    level: int = 0
    checksum: bool = False

    def __init__(self, *, level: int = 0, checksum: bool = False) -> None:
        # numcodecs 0.13.0 introduces the checksum attribute for the zstd codec
        _numcodecs_version = Version(numcodecs.__version__)
        if _numcodecs_version < Version("0.13.0"):
            raise RuntimeError(
                "numcodecs version >= 0.13.0 is required to use the zstd codec. "
                f"Version {_numcodecs_version} is currently installed."
            )

        level_parsed = parse_zstd_level(level)
        checksum_parsed = parse_checksum(checksum)

        object.__setattr__(self, "level", level_parsed)
        object.__setattr__(self, "checksum", checksum_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data)  # type: ignore[arg-type]

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if check_json_v2(data):
            if "checksum" in data:
                return cls(level=data["level"], checksum=data["checksum"])  # type: ignore[typeddict-item]
            else:
                return cls(level=data["level"])

        msg = (
            "Invalid Zarr V2 JSON representation of the zstd codec. "
            f"Got {data!r}, expected a Mapping with keys ('id', 'level')"
        )
        raise CodecValidationError(msg)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if check_json_v3(data):
            return cls(
                level=data["configuration"]["level"], checksum=data["configuration"]["checksum"]
            )
        msg = (
            "Invalid Zarr V3 JSON representation of the zstd codec. "
            f"Got {data!r}, expected a Mapping with keys ('name', 'configuration') "
            "Where the 'configuration' key is a Mapping with keys ('level', 'checksum')"
        )
        raise CodecValidationError(msg)

    def to_dict(self) -> dict[str, JSON]:
        return cast(dict[str, JSON], self.to_json(zarr_format=3))

    @overload
    def to_json(self, zarr_format: Literal[2]) -> ZstdJSON_V2: ...

    @overload
    def to_json(self, zarr_format: Literal[3]) -> ZstdJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> ZstdJSON_V2 | ZstdJSON_V3:
        if zarr_format == 2:
            return {"id": "zstd", "level": self.level}
        else:
            return {
                "name": "zstd",
                "configuration": {"level": self.level, "checksum": self.checksum},
            }

    @cached_property
    def _zstd_codec(self) -> Zstd:
        config_dict = {"level": self.level, "checksum": self.checksum}
        return Zstd.from_config(config_dict)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, self._zstd_codec.decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, self._zstd_codec.encode, chunk_bytes, chunk_spec.prototype
        )

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
