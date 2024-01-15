from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Type,
)

import numpy as np
from attr import frozen, field

from zarr.v3.abc.codec import ArrayBytesCodec
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import BytesLike

if TYPE_CHECKING:
    from zarr.v3.metadata import CodecMetadata, ChunkMetadata, ArrayMetadata, RuntimeConfiguration


Endian = Literal["big", "little"]


@frozen
class BytesCodecConfigurationMetadata:
    endian: Optional[Endian] = "little"


@frozen
class BytesCodecMetadata:
    configuration: BytesCodecConfigurationMetadata
    name: Literal["bytes"] = field(default="bytes", init=False)


@frozen
class BytesCodec(ArrayBytesCodec):
    configuration: BytesCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(cls, codec_metadata: CodecMetadata) -> BytesCodec:
        assert isinstance(codec_metadata, BytesCodecMetadata)
        return cls(configuration=codec_metadata.configuration)

    @classmethod
    def get_metadata_class(cls) -> Type[BytesCodecMetadata]:
        return BytesCodecMetadata

    def validate(self, array_metadata: ArrayMetadata) -> None:
        assert (
            not array_metadata.data_type.has_endianness or self.configuration.endian is not None
        ), "The `endian` configuration needs to be specified for multi-byte data types."

    def _get_byteorder(self, array: np.ndarray) -> Endian:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            import sys

            return sys.byteorder

    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_metadata: ChunkMetadata,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        if chunk_metadata.dtype.itemsize > 0:
            if self.configuration.endian == "little":
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(f"{prefix}{chunk_metadata.data_type.to_numpy_shortname()}")
        else:
            dtype = np.dtype(f"|{chunk_metadata.data_type.to_numpy_shortname()}")
        chunk_array = np.frombuffer(chunk_bytes, dtype)

        # ensure correct chunk shape
        if chunk_array.shape != chunk_metadata.chunk_shape:
            chunk_array = chunk_array.reshape(
                chunk_metadata.chunk_shape,
            )
        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
        _chunk_metadata: ChunkMetadata,
        _runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        if chunk_array.dtype.itemsize > 1:
            byteorder = self._get_byteorder(chunk_array)
            if self.configuration.endian != byteorder:
                new_dtype = chunk_array.dtype.newbyteorder(self.configuration.endian)
                chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int, _chunk_metadata: ChunkMetadata) -> int:
        return input_byte_length


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
