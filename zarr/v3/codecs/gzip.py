from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Type,
)

from attr import frozen, field
from numcodecs.gzip import GZip

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.common import RuntimeConfiguration
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import to_thread
from zarr.v3.metadata.v3 import CodecMetadata
from zarr.v3.types import BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata


@frozen
class GzipCodecConfigurationMetadata:
    level: int = 5


@frozen
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = field(default="gzip", init=False)


@frozen
class GzipCodec(BytesBytesCodec):
    array_metadata: ChunkMetadata
    configuration: GzipCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: CodecMetadata, array_metadata: ChunkMetadata
    ) -> GzipCodec:
        assert isinstance(codec_metadata, GzipCodecMetadata)

        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    @classmethod
    def get_metadata_class(cls) -> Type[GzipCodecMetadata]:
        return GzipCodecMetadata

    async def decode(self, chunk_bytes: bytes, config: RuntimeConfiguration) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).decode, chunk_bytes)

    async def encode(self, chunk_bytes: bytes, config: RuntimeConfiguration) -> Optional[BytesLike]:
        return await to_thread(GZip(self.configuration.level).encode, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec)
