from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Type,
)

from attr import frozen, field
from zstandard import ZstdCompressor, ZstdDecompressor

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import to_thread
from zarr.v3.metadata.v3 import CodecMetadata
from zarr.v3.types import BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata


@frozen
class ZstdCodecConfigurationMetadata:
    level: int = 0
    checksum: bool = False


@frozen
class ZstdCodecMetadata:
    configuration: ZstdCodecConfigurationMetadata
    name: Literal["zstd"] = field(default="zstd", init=False)


@frozen
class ZstdCodec(BytesBytesCodec):
    array_metadata: ChunkMetadata
    configuration: ZstdCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: CodecMetadata, array_metadata: ChunkMetadata
    ) -> ZstdCodec:
        assert isinstance(codec_metadata, ZstdCodecMetadata)
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    @classmethod
    def get_metadata_class(cls) -> Type[ZstdCodecMetadata]:
        return ZstdCodecMetadata

    def _compress(self, data: bytes) -> bytes:
        ctx = ZstdCompressor(
            level=self.configuration.level, write_checksum=self.configuration.checksum
        )
        return ctx.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        ctx = ZstdDecompressor()
        return ctx.decompress(data)

    async def decode(self, chunk_bytes: bytes, config: RuntimeConfiguration) -> BytesLike:
        return await to_thread(self._decompress, chunk_bytes)

    async def encode(self, chunk_bytes: bytes, config: RuntimeConfiguration) -> Optional[BytesLike]:
        return await to_thread(self._compress, chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int) -> int:
        raise NotImplementedError


register_codec("zstd", ZstdCodec)
