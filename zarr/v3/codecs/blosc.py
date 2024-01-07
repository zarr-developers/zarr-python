from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    Literal,
    Optional,
    Type,
)

import numcodecs
import numpy as np
from attr import asdict, evolve, frozen, field
from numcodecs.blosc import Blosc

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.common import RuntimeConfiguration
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import to_thread
from zarr.v3.metadata.v3 import CodecMetadata
from zarr.v3.types import BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata


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
class BloscCodecMetadata(CodecMetadata):
    configuration: BloscCodecConfigurationMetadata
    name: Literal["blosc"] = field(default="blosc", init=False)


@frozen
class BloscCodec(BytesBytesCodec):
    array_metadata: ChunkMetadata
    configuration: BloscCodecConfigurationMetadata
    blosc_codec: Blosc
    is_fixed_size = False

    @classmethod
    def from_metadata(
        cls, codec_metadata: CodecMetadata, array_metadata: ChunkMetadata
    ) -> BloscCodec:
        assert isinstance(codec_metadata, BloscCodecMetadata)
        configuration = codec_metadata.configuration
        if configuration.typesize == 0:
            configuration = evolve(configuration, typesize=array_metadata.data_type.byte_count)
        config_dict = asdict(codec_metadata.configuration)
        config_dict.pop("typesize", None)
        map_shuffle_str_to_int = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        config_dict["shuffle"] = map_shuffle_str_to_int[config_dict["shuffle"]]
        return cls(
            array_metadata=array_metadata,
            configuration=configuration,
            blosc_codec=Blosc.from_config(config_dict),
        )

    @classmethod
    def get_metadata_class(cls) -> Type[BloscCodecMetadata]:
        return BloscCodecMetadata

    async def decode(self, chunk_bytes: bytes, config: RuntimeConfiguration) -> BytesLike:
        return await to_thread(self.blosc_codec.decode, chunk_bytes)

    async def encode(self, chunk_bytes: bytes, config: RuntimeConfiguration) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=self.array_metadata.dtype)
        return await to_thread(self.blosc_codec.encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
