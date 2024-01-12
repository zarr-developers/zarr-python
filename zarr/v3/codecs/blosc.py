from __future__ import annotations
from functools import lru_cache

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
    Type,
)

import numcodecs
import numpy as np
from attr import evolve, frozen, field
from numcodecs.blosc import Blosc

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import BytesLike, to_thread

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata, CodecMetadata, DataType


BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]

# See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
numcodecs.blosc.use_threads = False


@frozen
class BloscCodecConfigurationMetadata:
    typesize: int
    cname: Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"] = "zstd"
    clevel: int = 5
    shuffle: BloscShuffle = "noshuffle"
    blocksize: int = 0


blosc_shuffle_int_to_str: Dict[int, BloscShuffle] = {
    0: "noshuffle",
    1: "shuffle",
    2: "bitshuffle",
}


@frozen
class BloscCodecMetadata:
    configuration: BloscCodecConfigurationMetadata
    name: Literal["blosc"] = field(default="blosc", init=False)


@frozen
class BloscCodec(BytesBytesCodec):
    configuration: BloscCodecConfigurationMetadata
    is_fixed_size = False

    @classmethod
    def from_metadata(cls, codec_metadata: CodecMetadata) -> BloscCodec:
        assert isinstance(codec_metadata, BloscCodecMetadata)
        return cls(configuration=codec_metadata.configuration)

    @classmethod
    def get_metadata_class(cls) -> Type[BloscCodecMetadata]:
        return BloscCodecMetadata

    def evolve(self, *, data_type: DataType, **_kwargs) -> BloscCodec:
        new_codec = self
        if new_codec.configuration.typesize == 0:
            new_configuration = evolve(new_codec.configuration, typesize=data_type.byte_count)
            new_codec = evolve(new_codec, configuration=new_configuration)

        return new_codec

    @lru_cache
    def get_blosc_codec(self) -> Blosc:
        map_shuffle_str_to_int = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        config_dict = {
            "cname": self.configuration.cname,
            "clevel": self.configuration.clevel,
            "shuffle": map_shuffle_str_to_int[self.configuration.shuffle],
            "blocksize": self.configuration.blocksize,
        }
        return Blosc.from_config(config_dict)

    async def decode(self, chunk_bytes: bytes, _chunk_metadata: ChunkMetadata) -> BytesLike:
        return await to_thread(self.get_blosc_codec().decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
        chunk_metadata: ChunkMetadata,
    ) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=chunk_metadata.dtype)
        return await to_thread(self.get_blosc_codec().encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_metadata: ChunkMetadata) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
