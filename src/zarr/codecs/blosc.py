from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Final, Literal, NotRequired, TypedDict, TypeGuard, overload

import numcodecs
from numcodecs.blosc import Blosc
from packaging.version import Version

from zarr.abc.codec import BytesBytesCodec, CodecJSON, CodecJSON_V2, CodecValidationError
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import (
    JSON,
    NamedRequiredConfig,
    ZarrFormat,
)
from zarr.core.dtype.common import HasItemSize
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
BLOSC_SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")

BloscCname = Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]
BLOSC_CNAME: Final = ("lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib")


class BloscConfigV2(TypedDict):
    cname: BloscCname
    clevel: int
    shuffle: int
    blocksize: int
    typesize: NotRequired[int]


class BloscConfigV3(TypedDict):
    cname: BloscCname
    clevel: int
    shuffle: BloscShuffle
    blocksize: int
    typesize: int


class BloscJSON_V2(CodecJSON_V2[Literal["blosc"]], BloscConfigV2):
    """
    The JSON form of the Blosc codec in Zarr V2.
    """


class BloscJSON_V3(NamedRequiredConfig[Literal["blosc"], BloscConfigV3]):
    """
    The JSON form of the Blosc codec in Zarr V3.
    """


def check_json_v2(data: CodecJSON) -> TypeGuard[BloscJSON_V2]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) == {"id", "clevel", "cname", "shuffle", "blocksize"}
        and data["id"] == "blosc"
    )


def check_json_v3(data: CodecJSON) -> TypeGuard[BloscJSON_V3]:
    return (
        isinstance(data, Mapping)
        and set(data.keys()) == {"name", "configuration"}
        and data["name"] == "blosc"
        and isinstance(data["configuration"], Mapping)
        and set(data["configuration"].keys())
        == {"cname", "clevel", "shuffle", "blocksize", "typesize"}
    )


def parse_cname(value: object) -> BloscCname:
    if value not in BLOSC_CNAME:
        raise ValueError(f"Value must be one of {BLOSC_CNAME}. Got {value} instead.")
    return value


# See https://zarr.readthedocs.io/en/stable/user-guide/performance.html#configuring-blosc
numcodecs.blosc.use_threads = False


def parse_typesize(data: JSON) -> int:
    if isinstance(data, int):
        if data > 0:
            return data
        else:
            raise ValueError(
                f"Value must be greater than 0. Got {data}, which is less or equal to 0."
            )
    raise TypeError(f"Value must be an int. Got {type(data)} instead.")


# todo: real validation
def parse_clevel(data: JSON) -> int:
    if isinstance(data, int):
        return data
    raise TypeError(f"Value should be an int. Got {type(data)} instead.")


def parse_blocksize(data: JSON) -> int:
    if isinstance(data, int):
        return data
    raise TypeError(f"Value should be an int. Got {type(data)} instead.")


def parse_shuffle(data: object) -> BloscShuffle:
    if data in BLOSC_SHUFFLE:
        return data  # type: ignore[return-value]
    raise TypeError(f"Value must be one of {BLOSC_SHUFFLE}. Got {data} instead.")


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    is_fixed_size = False

    typesize: int | None
    cname: BloscCname
    clevel: int
    shuffle: BloscShuffle | None
    blocksize: int

    def __init__(
        self,
        *,
        typesize: int | None = None,
        cname: BloscCname = "zstd",
        clevel: int = 5,
        shuffle: BloscShuffle | None = None,
        blocksize: int = 0,
    ) -> None:
        typesize_parsed = parse_typesize(typesize) if typesize is not None else None
        cname_parsed = parse_cname(cname)
        clevel_parsed = parse_clevel(clevel)
        shuffle_parsed = parse_shuffle(shuffle) if shuffle is not None else None
        blocksize_parsed = parse_blocksize(blocksize)

        object.__setattr__(self, "typesize", typesize_parsed)
        object.__setattr__(self, "cname", cname_parsed)
        object.__setattr__(self, "clevel", clevel_parsed)
        object.__setattr__(self, "shuffle", shuffle_parsed)
        object.__setattr__(self, "blocksize", blocksize_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        return cls.from_json(data, zarr_format=3)

    def to_dict(self) -> dict[str, JSON]:
        return self.to_json(zarr_format=3)

    @classmethod
    def _from_json_v2(cls, data: CodecJSON) -> Self:
        if check_json_v2(data):
            return cls(
                cname=data["cname"],
                clevel=data["clevel"],
                shuffle=BLOSC_SHUFFLE[data["shuffle"]],
                blocksize=data["blocksize"],
                typesize=data.get("typesize", None),
            )
        msg = (
            "Invalid Zarr V2 JSON representation of the blosc codec. "
            f"Got {data!r}, expected a Mapping with keys ('id', 'cname', 'clevel', 'shuffle', 'blocksize', 'typesize')"
        )
        raise CodecValidationError(msg)

    @classmethod
    def _from_json_v3(cls, data: CodecJSON) -> Self:
        if check_json_v3(data):
            return cls(
                typesize=data["configuration"]["typesize"],
                cname=data["configuration"]["cname"],
                clevel=data["configuration"]["clevel"],
                shuffle=data["configuration"]["shuffle"],
                blocksize=data["configuration"]["blocksize"],
            )
        msg = (
            "Invalid Zarr V3 JSON representation of the blosc codec. "
            f"Got {data!r}, expected a Mapping with keys ('name', 'configuration')"
            "Where the 'configuration' key is a Mapping with keys ('cname', 'clevel', 'shuffle', 'blocksize', 'typesize')"
        )
        raise CodecValidationError(msg)

    @overload
    def to_json(self, zarr_format: Literal[2]) -> BloscJSON_V2: ...
    @overload
    def to_json(self, zarr_format: Literal[3]) -> BloscJSON_V3: ...

    def to_json(self, zarr_format: ZarrFormat) -> BloscJSON_V2 | BloscJSON_V3:
        if self.typesize is None or self.shuffle is None:
            raise ValueError("typesize and blocksize need to be set for encoding.")
        if zarr_format == 2:
            return {
                "id": "blosc",
                "clevel": self.clevel,
                "cname": self.cname,
                "shuffle": BLOSC_SHUFFLE.index(self.shuffle),
                "blocksize": self.blocksize,
            }
        elif zarr_format == 3:
            return {
                "name": "blosc",
                "configuration": {
                    "clevel": self.clevel,
                    "cname": self.cname,
                    "shuffle": self.shuffle,
                    "typesize": self.typesize,
                    "blocksize": self.blocksize,
                },
            }
        raise ValueError(
            f"Unsupported Zarr format {zarr_format}. Expected 2 or 3."
        )  # pragma: no cover

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        item_size = 1
        if isinstance(array_spec.dtype, HasItemSize):
            item_size = array_spec.dtype.item_size
        new_codec = self
        if new_codec.typesize is None:
            new_codec = replace(new_codec, typesize=item_size)
        if new_codec.shuffle is None:
            new_codec = replace(new_codec, shuffle="bitshuffle" if item_size == 1 else "shuffle")

        return new_codec

    @cached_property
    def _blosc_codec(self) -> Blosc:
        if self.shuffle is None:
            raise ValueError("`shuffle` needs to be set for decoding and encoding.")
        config_dict = {
            "cname": self.cname,
            "clevel": self.clevel,
            "shuffle": BLOSC_SHUFFLE.index(self.shuffle),
            "blocksize": self.blocksize,
        }
        # See https://github.com/zarr-developers/numcodecs/pull/713
        if Version(numcodecs.__version__) >= Version("0.16.0"):
            config_dict["typesize"] = self.typesize
        return Blosc.from_config(config_dict)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, self._blosc_codec.decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        # Since blosc only support host memory, we convert the input and output of the encoding
        # between numpy array and buffer
        return await asyncio.to_thread(
            lambda chunk: chunk_spec.prototype.buffer.from_bytes(
                self._blosc_codec.encode(chunk.as_numpy_array())
            ),
            chunk_bytes,
        )

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
