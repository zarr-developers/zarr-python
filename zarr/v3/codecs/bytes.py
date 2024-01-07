from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Type,
    TypedDict,
)

import numpy as np

from zarr.v3.abc.codec import ArrayBytesCodec
from zarr.v3.common import RuntimeConfiguration
from zarr.v3.codecs.registry import register_codec
from zarr.v3.types import BytesLike
from zarr.v3.metadata.v3 import CodecMetadata, to_numpy_shortname

if TYPE_CHECKING:
    from zarr.v3.common import ChunkMetadata, ChunkMetadataDict


class BytesCodecConfigurationMetadataDict(TypedDict):
    endian: Optional[Literal["big", "little"]]


class BytesCodecConfigurationMetadata:
    endian: Optional[Literal["big", "little"]] = "little"

    def __init__(self, endian: Optional[Literal["big", "little"]]):
        self.endian = endian

    def to_dict(self) -> BytesCodecConfigurationMetadataDict:
        return {"endian": self.endian}


class BytesCodecMetadataDict(TypedDict):
    configuration: BytesCodecConfigurationMetadataDict
    name: Literal["bytes"]


class BytesCodecMetadata:
    configuration: BytesCodecConfigurationMetadata
    name: Literal["bytes"]

    def __init__(self, configuration: BytesCodecConfigurationMetadata):
        # note: the only degree of freedom for this class is the "endian" property of `BytesConfigurationMetadata`
        self.configuration = configuration
        self.name = "bytes"

    def to_dict(self) -> BytesCodecMetadataDict:
        return {"configuration": self.configuration.to_dict(), "name": self.name}


class BytesCodecDict(TypedDict):
    array_metadata: ChunkMetadataDict
    configuration: BytesCodecConfigurationMetadataDict


class BytesCodec(ArrayBytesCodec):
    array_metadata: ChunkMetadata
    configuration: BytesCodecConfigurationMetadata
    is_fixed_size = True

    def __init__(
        self, array_metadata: ChunkMetadata, configuration: BytesCodecConfigurationMetadata
    ) -> None:
        self.array_metadata = array_metadata
        self.configuration = configuration

    @classmethod
    def from_metadata(
        cls, codec_metadata: CodecMetadata, array_metadata: ChunkMetadata
    ) -> BytesCodec:
        assert isinstance(codec_metadata, BytesCodecMetadata)
        assert (
            array_metadata.dtype.itemsize == 1 or codec_metadata.configuration.endian is not None
        ), "The `endian` configuration needs to be specified for multi-byte data types."
        return cls(
            array_metadata=array_metadata,
            configuration=codec_metadata.configuration,
        )

    @classmethod
    def get_metadata_class(cls) -> Type[BytesCodecMetadata]:
        return BytesCodecMetadata

    def _get_byteorder(self, array: np.ndarray) -> Literal["big", "little"]:
        if array.dtype.byteorder == "<":
            return "little"
        elif array.dtype.byteorder == ">":
            return "big"
        else:
            import sys

            return sys.byteorder

    async def decode(self, chunk_bytes: BytesLike, config: RuntimeConfiguration) -> np.ndarray:
        short_name = to_numpy_shortname(self.array_metadata.dtype)
        if self.array_metadata.dtype.itemsize > 0:
            if self.configuration.endian == "little":
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(f"{prefix}{short_name}")
        else:
            dtype = np.dtype(f"|{short_name}")
        chunk_array = np.frombuffer(chunk_bytes, dtype)

        # ensure correct chunk shape
        if chunk_array.shape != self.array_metadata.chunk_shape:
            chunk_array = chunk_array.reshape(
                self.array_metadata.chunk_shape,
            )
        return chunk_array

    async def encode(
        self, chunk_array: np.ndarray, config: RuntimeConfiguration
    ) -> Optional[BytesLike]:
        if chunk_array.dtype.itemsize > 1:
            byteorder = self._get_byteorder(chunk_array)
            if self.configuration.endian != byteorder:
                new_dtype = chunk_array.dtype.newbyteorder(self.configuration.endian)
                chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length

    def to_dict(self) -> BytesCodecDict:
        return {
            "array_metadata": self.array_metadata.to_dict(),
            "configuration": self.configuration.to_dict(),
        }


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
