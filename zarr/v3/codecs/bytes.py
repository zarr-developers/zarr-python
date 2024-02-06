from __future__ import annotations
from dataclasses import dataclass, field

from typing import (
    TYPE_CHECKING,
)

import numpy as np

from zarr.v3.abc.codec import ArrayBytesCodec
from zarr.v3.abc.metadata import Metadata
from zarr.v3.codecs.registry import register_codec
from zarr.v3.common import BytesLike, RuntimeConfiguration
from zarr.v3.common import NamedConfig

if TYPE_CHECKING:
    from zarr.v3.metadata import CoreArrayMetadata
    from typing_extensions import Self
    from typing import Any, Dict, Literal, Optional, Type


def parse_endian(data: Any) -> Literal["big", "little"]:
    if data in ("big", "little"):
        return data
    msg = f"Expected on of ('big', 'little'), got {data} instead."
    raise ValueError(msg)


def parse_name(data: Any) -> Literal["bytes"]:
    if data == "bytes":
        return data
    msg = f"Expected 'bytes', got {data} instead."
    raise ValueError(msg)


@dataclass(frozen=True)
class BytesCodecConfigurationMetadata(Metadata):
    endian: Optional[Literal["big", "little"]] = "little"

    def __init__(self, endian: Literal["big", "little"]):
        endian_parsed = parse_endian(endian)
        object.__setattr__(self, "endian", endian_parsed)


@dataclass(frozen=True)
class BytesCodecMetadata(Metadata):
    configuration: BytesCodecConfigurationMetadata
    name: Literal["bytes"] = field(default="bytes", init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        _ = parse_name(data.pop("name"))
        return cls(**data)


@dataclass(frozen=True)
class BytesCodec(ArrayBytesCodec):
    array_metadata: CoreArrayMetadata
    configuration: BytesCodecConfigurationMetadata
    is_fixed_size = True

    @classmethod
    def from_metadata(
        cls, codec_metadata: NamedConfig, array_metadata: CoreArrayMetadata
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

    async def decode(
        self, chunk_bytes: BytesLike, runtime_configuration: RuntimeConfiguration
    ) -> np.ndarray:
        if self.array_metadata.dtype.itemsize > 0:
            if self.configuration.endian == "little":
                prefix = "<"
            else:
                prefix = ">"
            dtype = np.dtype(f"{prefix}{self.array_metadata.dtype.str[1:]}")
        else:
            dtype = np.dtype(f"|{self.array_metadata.dtype.str[1:]}")
        chunk_array = np.frombuffer(chunk_bytes, dtype)

        # ensure correct chunk shape
        if chunk_array.shape != self.array_metadata.chunk_shape:
            chunk_array = chunk_array.reshape(
                self.array_metadata.chunk_shape,
            )
        return chunk_array

    async def encode(
        self, chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
    ) -> Optional[BytesLike]:
        if chunk_array.dtype.itemsize > 1:
            byteorder = self._get_byteorder(chunk_array)
            if self.configuration.endian != byteorder:
                new_dtype = chunk_array.dtype.newbyteorder(self.configuration.endian)
                chunk_array = chunk_array.astype(new_dtype)
        return chunk_array.tobytes()

    def compute_encoded_size(self, input_byte_length: int) -> int:
        return input_byte_length


register_codec("bytes", BytesCodec)

# compatibility with earlier versions of ZEP1
register_codec("endian", BytesCodec)
