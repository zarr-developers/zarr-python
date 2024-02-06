# Notes:
# 1. These are missing methods described in the spec. I expected to see these method definitions:
# def compute_encoded_representation_type(self, decoded_representation_type):
# def encode(self, decoded_value):
# def decode(self, encoded_value, decoded_representation_type):
# def partial_decode(self, input_handle, decoded_representation_type, decoded_regions):
# def compute_encoded_size(self, input_size):
# 2. Understand why array metadata is included on all codecs


from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Optional, Type

import numpy as np

from zarr.v3.common import BytesLike, NamedConfig, SliceSelection
from zarr.v3.common import RuntimeConfiguration
from zarr.v3.store import StorePath


if TYPE_CHECKING:
    from zarr.v3.metadata import CoreArrayMetadata


class Codec(ABC):
    metadata: NamedConfig
    is_fixed_size: bool
    array_metadata: CoreArrayMetadata

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int) -> int:
        pass

    def resolve_metadata(self) -> CoreArrayMetadata:
        return self.array_metadata

    @classmethod
    @abstractmethod
    def from_metadata(
        cls, codec_metadata: "NamedConfig", array_metadata: CoreArrayMetadata
    ) -> Codec:
        pass

    @classmethod
    @abstractmethod
    def get_metadata_class(cls) -> "Type[NamedConfig]":
        pass


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self, chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self, chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self, chunk_array: BytesLike, runtime_configuration: RuntimeConfiguration
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self, chunk_array: np.ndarray, runtime_configuration: RuntimeConfiguration
    ) -> Optional[BytesLike]:
        pass


class ArrayBytesCodecPartialDecodeMixin:
    @abstractmethod
    async def decode_partial(
        self,
        store_path: StorePath,
        selection: SliceSelection,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodecPartialEncodeMixin:
    @abstractmethod
    async def encode_partial(
        self,
        store_path: StorePath,
        chunk_array: np.ndarray,
        selection: SliceSelection,
    ) -> None:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self, chunk_array: BytesLike, runtime_configuration: RuntimeConfiguration
    ) -> BytesLike:
        pass

    @abstractmethod
    async def encode(
        self, chunk_array: BytesLike, runtime_configuration: RuntimeConfiguration
    ) -> Optional[BytesLike]:
        pass
