from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

import numcodecs
from numcodecs.blosc import Blosc
from packaging.version import Version

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, parse_enum, parse_named_configuration
from zarr.core.dtype.common import HasItemSize
from zarr.registry import register_codec

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer


class BloscShuffle(Enum):
    """
    Enum for shuffle filter used by blosc.
    """

    noshuffle = "noshuffle"
    shuffle = "shuffle"
    bitshuffle = "bitshuffle"

    @classmethod
    def from_int(cls, num: int) -> BloscShuffle:
        blosc_shuffle_int_to_str = {
            0: "noshuffle",
            1: "shuffle",
            2: "bitshuffle",
        }
        if num not in blosc_shuffle_int_to_str:
            raise ValueError(f"Value must be between 0 and 2. Got {num}.")
        return BloscShuffle[blosc_shuffle_int_to_str[num]]


class BloscCname(Enum):
    """
    Enum for compression library used by blosc.
    """

    lz4 = "lz4"
    lz4hc = "lz4hc"
    blosclz = "blosclz"
    zstd = "zstd"
    snappy = "snappy"
    zlib = "zlib"


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


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    """blosc codec"""

    is_fixed_size = False

    typesize: int | None
    cname: BloscCname = BloscCname.zstd
    clevel: int = 5
    shuffle: BloscShuffle | None = BloscShuffle.noshuffle
    blocksize: int = 0

    def __init__(
        self,
        *,
        typesize: int | None = None,
        cname: BloscCname | str = BloscCname.zstd,
        clevel: int = 5,
        shuffle: BloscShuffle | str | None = None,
        blocksize: int = 0,
    ) -> None:
        typesize_parsed = parse_typesize(typesize) if typesize is not None else None
        cname_parsed = parse_enum(cname, BloscCname)
        clevel_parsed = parse_clevel(clevel)
        shuffle_parsed = parse_enum(shuffle, BloscShuffle) if shuffle is not None else None
        blocksize_parsed = parse_blocksize(blocksize)

        object.__setattr__(self, "typesize", typesize_parsed)
        object.__setattr__(self, "cname", cname_parsed)
        object.__setattr__(self, "clevel", clevel_parsed)
        object.__setattr__(self, "shuffle", shuffle_parsed)
        object.__setattr__(self, "blocksize", blocksize_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "blosc")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        if self.typesize is None:
            raise ValueError("`typesize` needs to be set for serialization.")
        if self.shuffle is None:
            raise ValueError("`shuffle` needs to be set for serialization.")
        return {
            "name": "blosc",
            "configuration": {
                "typesize": self.typesize,
                "cname": self.cname.value,
                "clevel": self.clevel,
                "shuffle": self.shuffle.value,
                "blocksize": self.blocksize,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        item_size = 1
        if isinstance(array_spec.dtype, HasItemSize):
            item_size = array_spec.dtype.item_size
        new_codec = self
        if new_codec.typesize is None:
            new_codec = replace(new_codec, typesize=item_size)
        if new_codec.shuffle is None:
            new_codec = replace(
                new_codec,
                shuffle=(BloscShuffle.bitshuffle if item_size == 1 else BloscShuffle.shuffle),
            )

        return new_codec

    @cached_property
    def _blosc_codec(self) -> Blosc:
        if self.shuffle is None:
            raise ValueError("`shuffle` needs to be set for decoding and encoding.")
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
