from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Optional, Type

import numpy as np

from zarr.v3.common import BytesLike, NamedConfig, SliceSelection
from zarr.v3.common import RuntimeConfiguration
from zarr.v3.store import StorePath


if TYPE_CHECKING:
    from zarr.v3.metadata import (
        ArraySpec,
        ArrayMetadata,
        DataType,
        CodecMetadata,
        RuntimeConfiguration,
    )


class Codec(ABC):
    metadata: NamedConfig
    is_fixed_size: bool

    @classmethod
    @abstractmethod
    def from_metadata(cls, codec_metadata: "NamedConfig") -> Codec:
        pass

    @classmethod
    def get_metadata_class(cls) -> Type[CodecMetadata]:
        pass

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        pass

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec

    def evolve(self, *, ndim: int, data_type: DataType) -> Codec:
        return self

    def validate(self, array_metadata: ArrayMetadata) -> None:
        pass


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        pass


class ArrayBytesCodecPartialDecodeMixin:
    @abstractmethod
    async def decode_partial(
        self,
        store_path: StorePath,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodecPartialEncodeMixin:
    @abstractmethod
    async def encode_partial(
        self,
        store_path: StorePath,
        chunk_array: np.ndarray,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        pass
