from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Optional

import numpy as np

from zarr.v3.common import BytesLike


if TYPE_CHECKING:
    from zarr.v3.metadata import CoreArrayMetadata


class Codec(ABC):
    supports_partial_decode: bool
    supports_partial_encode: bool
    is_fixed_size: bool
    array_metadata: CoreArrayMetadata

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int) -> int:
        pass

    def resolve_metadata(self) -> CoreArrayMetadata:
        return self.array_metadata


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[np.ndarray]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
    ) -> np.ndarray:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
    ) -> Optional[BytesLike]:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: BytesLike,
    ) -> BytesLike:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: BytesLike,
    ) -> Optional[BytesLike]:
        pass
