from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Type,
)

import numpy as np
from attr import frozen, field
from crc32c import crc32c

from zarr.v3.abc.codec import BytesBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata import ChunkMetadata, CodecMetadata, RuntimeConfiguration


@frozen
class Crc32cCodecMetadata:
    name: Literal["crc32c"] = field(default="crc32c", init=False)


@frozen
class Crc32cCodec(BytesBytesCodec):
    is_fixed_size = True

    @classmethod
    def from_metadata(cls, codec_metadata: CodecMetadata) -> Crc32cCodec:
        assert isinstance(codec_metadata, Crc32cCodecMetadata)
        return cls()

    @classmethod
    def get_metadata_class(cls) -> Type[Crc32cCodecMetadata]:
        return Crc32cCodecMetadata

    async def decode(
        self,
        chunk_bytes: bytes,
        _chunk_metadata: ChunkMetadata,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        crc32_bytes = chunk_bytes[-4:]
        inner_bytes = chunk_bytes[:-4]

        assert np.uint32(crc32c(inner_bytes)).tobytes() == bytes(crc32_bytes)
        return inner_bytes

    async def encode(
        self,
        chunk_bytes: bytes,
        _chunk_metadata: ChunkMetadata,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        return chunk_bytes + np.uint32(crc32c(chunk_bytes)).tobytes()

    def compute_encoded_size(self, input_byte_length: int, _chunk_metadata: ChunkMetadata) -> int:
        return input_byte_length + 4


register_codec("crc32c", Crc32cCodec)
