from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
)

from attr import frozen, field
from numcodecs.gzip import GZip

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import BytesLike, to_thread

if TYPE_CHECKING:
    from zarr.v3.metadata import CoreArrayMetadata


@frozen
class GzipCodecConfigurationMetadata:
    level: int = 5


@frozen
class GzipCodecMetadata:
    configuration: GzipCodecConfigurationMetadata
    name: Literal["gzip"] = field(default="gzip", init=False)


@frozen
class GzipCodec(BytesBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: GzipCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: GzipCodecMetadata, array_metadata: CoreArrayMetadata
    ) -> GzipCodec:
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    async def decode(
        self,
        chunk_bytes: bytes,
    ) -> BytesLike:
        return await to_thread(GZip(self.configuration.level).decode, chunk_bytes)

    async def encode(
        self,
        chunk_bytes: bytes,
    ) -> Optional[BytesLike]:
        return await to_thread(GZip(self.configuration.level).encode, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("gzip", GzipCodec, GzipCodecMetadata)
