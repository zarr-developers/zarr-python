from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
)

import numcodecs
import numpy as np
from numcodecs.blosc import Blosc

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import JSON, parse_enum, parse_name, to_thread

if TYPE_CHECKING:
    from zarr.v3.common import ArraySpec
    from typing_extensions import Self
    from zarr.v3.common import BytesLike, RuntimeConfiguration


class BloscShuffle(Enum):
    noshuffle = "noshuffle"
    shuffle = "shuffle"
    bitshuffle = "bitshuffle"

    @classmethod
    def from_int(cls, num: int) -> Self:
        blosc_shuffle_int_to_str = {
            0: "noshuffle",
            1: "shuffle",
            2: "bitshuffle",
        }
        if num not in blosc_shuffle_int_to_str:
            raise ValueError(f"Value must be between 0 and 2. Got {num}.")
        return BloscShuffle[blosc_shuffle_int_to_str[num]]


class BloscCname(Enum):
    lz4 = "lz4"
    lz4hc = "lz4hc"
    blosclz = "blosclz"
    zstd = "zstd"
    snappy = "snappy"
    zlib = "zlib"


# See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
numcodecs.blosc.use_threads = False


def parse_typesize(data: JSON) -> int:
    if isinstance(data, int):
        if data >= 0:
            return data
        else:
            msg = f"Value must be greater than or equal to 0. Got {data}, which is less than 0."
            raise ValueError(msg)
    msg = f"Value must be an int. Got {type(data)} instead."
    raise TypeError(msg)


def parse_cname(data: JSON) -> BloscCname:
    return parse_enum(data, BloscCname)


# todo: real validation
def parse_clevel(data: JSON) -> int:
    if isinstance(data, int):
        return data
    msg = f"Value should be an int, got {type(data)} instead"
    raise TypeError(msg)


def parse_shuffle(data: JSON) -> BloscShuffle:
    return parse_enum(data, BloscShuffle)


def parse_blocksize(data: JSON) -> int:
    if isinstance(data, int):
        return data
    msg = f"Value should be an int, got {type(data)} instead"
    raise TypeError(msg)


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    is_fixed_size = False

    typesize: int
    cname: BloscCname = BloscCname.zstd
    clevel: int = 5
    shuffle: BloscShuffle = BloscShuffle.noshuffle
    blocksize: int = 0

    def __init__(
        self,
        *,
        typesize,
        cname=BloscCname.zstd,
        clevel=5,
        shuffle=BloscShuffle.noshuffle,
        blocksize=0,
    ) -> None:
        typesize_parsed = parse_typesize(typesize)
        cname_parsed = parse_cname(cname)
        clevel_parsed = parse_clevel(clevel)
        shuffle_parsed = parse_shuffle(shuffle)
        blocksize_parsed = parse_blocksize(blocksize)

        object.__setattr__(self, "typesize", typesize_parsed)
        object.__setattr__(self, "cname", cname_parsed)
        object.__setattr__(self, "clevel", clevel_parsed)
        object.__setattr__(self, "shuffle", shuffle_parsed)
        object.__setattr__(self, "blocksize", blocksize_parsed)

    @classmethod
    def from_dict(cls, data: Dict[str, JSON]) -> Self:
        parse_name(data["name"], "blosc")
        return cls(**data["configuration"])

    def to_dict(self) -> Dict[str, JSON]:
        return {
            "name": "blosc",
            "configuration": {
                "typesize": self.typesize,
                "cname": self.cname,
                "clevel": self.clevel,
                "shuffle": self.shuffle,
                "blocksize": self.blocksize,
            },
        }

    def evolve(self, array_spec: ArraySpec) -> Self:
        new_codec = self
        if new_codec.typesize == 0:
            new_codec = replace(new_codec, typesize=array_spec.dtype.itemsize)

        return new_codec

    @lru_cache
    def get_blosc_codec(self) -> Blosc:
        map_shuffle_str_to_int = {
            BloscShuffle.noshuffle: 0,
            BloscShuffle.shuffle: 1,
            BloscShuffle.bitshuffle: 2,
        }
        config_dict = {
            "cname": self.cname.name,
            "clevel": self.clevel,
            "shuffle": map_shuffle_str_to_int[self.shuffle],
            "blocksize": self.blocksize,
        }
        return Blosc.from_config(config_dict)

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        return await to_thread(self.get_blosc_codec().decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=chunk_spec.dtype)
        return await to_thread(self.get_blosc_codec().encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
