from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Type,
)

import numcodecs
import numpy as np
from numcodecs.blosc import Blosc

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.abc.metadata import Metadata
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import BytesLike, RuntimeConfiguration, to_thread
from zarr.v3.common import NamedConfig

if TYPE_CHECKING:
    from zarr.v3.metadata import CoreArrayMetadata
    from typing_extensions import Self

BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
BloscCname = Literal["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]
# See https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
numcodecs.blosc.use_threads = False


def parse_typesize(data: Any) -> int:
    if isinstance(data, int):
        if data >= 0:
            return data
        else:
            msg = f"Value must be greater than or equal to 0. Got {data}, which is less than 0."
            raise ValueError(msg)
    msg = f"Value must be an int. Got {type(data)} instead."
    raise TypeError(msg)


def parse_cname(data: Any) -> BloscCname:
    if data in ["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"]:
        return data
    msg = f'Value must be one of ["lz4", "lz4hc", "blosclz", "zstd", "snappy", "zlib"], got {data} instead.'
    raise ValueError(msg)


# todo: real validation
def parse_clevel(data: Any) -> int:
    if isinstance(data, int):
        return data
    msg = f"Value should be an int, got {type(data)} instead"
    raise TypeError(msg)


def parse_shuffle(data: Any) -> BloscShuffle:
    if data in ["noshuffle", "shuffle", "bitshuffle"]:
        return data
    msg = f'Value must be one of ["noshuffle", "shuffle", "bitshuffle"], got {data} instead.'
    raise ValueError(msg)


def parse_blocksize(data: Any) -> int:
    if isinstance(data, int):
        return data
    msg = f"Value should be an int, got {type(data)} instead"
    raise TypeError(msg)


def parse_name(data: Any) -> Literal["blosc"]:
    if data == "blosc":
        return data
    msg = f"Expected 'blosc', got {data} instead."
    raise ValueError(msg)


@dataclass(frozen=True)
class BloscCodecConfigurationMetadata(Metadata):
    typesize: int
    cname: BloscCname = "zstd"
    clevel: int = 5
    shuffle: BloscShuffle = "noshuffle"
    blocksize: int = 0

    def __init__(
        self,
        typesize: int,
        cname: BloscCname = "zstd",
        clevel: int = 5,
        shuffle: BloscShuffle = "noshuffle",
        blocksize: int = 0,
    ):
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


blosc_shuffle_int_to_str: Dict[int, BloscShuffle] = {
    0: "noshuffle",
    1: "shuffle",
    2: "bitshuffle",
}


@dataclass(frozen=True)
class BloscCodecMetadata(Metadata):
    configuration: BloscCodecConfigurationMetadata
    name: Literal["blosc"] = field(default="blosc", init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        _ = parse_name(data.pop("name"))
        return cls(**data)


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: BloscCodecConfigurationMetadata
    blosc_codec: Blosc
    is_fixed_size = False

    @classmethod
    def from_metadata(
        cls, codec_metadata: NamedConfig, array_metadata: CoreArrayMetadata
    ) -> BloscCodec:
        assert isinstance(codec_metadata, BloscCodecMetadata)
        configuration = codec_metadata.configuration
        if configuration.typesize == 0:
            configuration = replace(configuration, typesize=array_metadata.dtype.byte_count)
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

    async def decode(
        self, chunk_bytes: bytes, runtime_configuration: RuntimeConfiguration
    ) -> BytesLike:
        return await to_thread(self.blosc_codec.decode, chunk_bytes)

    async def encode(
        self, chunk_bytes: bytes, runtime_configuration: RuntimeConfiguration
    ) -> Optional[BytesLike]:
        chunk_array = np.frombuffer(chunk_bytes, dtype=self.array_metadata.dtype)
        return await to_thread(self.blosc_codec.encode, chunk_array)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("blosc", BloscCodec)
