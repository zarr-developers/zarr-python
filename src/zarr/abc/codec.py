from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from zarr.abc.metadata import Metadata

from zarr.buffer import Buffer, NDBuffer
from zarr.common import ArraySpec
from zarr.store import StorePath


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.common import SliceSelection
    from zarr.metadata import ArrayMetadata
    from zarr.config import RuntimeConfiguration


class Codec(Metadata):
    is_fixed_size: bool

    @abstractmethod
    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        pass

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec

    def evolve(self, array_spec: ArraySpec) -> Self:
        return self

    def validate(self, array_metadata: ArrayMetadata) -> None:
        pass


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> NDBuffer:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[NDBuffer]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: Buffer,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> NDBuffer:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[Buffer]:
        pass


class ArrayBytesCodecPartialDecodeMixin:
    @abstractmethod
    async def decode_partial(
        self,
        store_path: StorePath,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[NDBuffer]:
        pass


class ArrayBytesCodecPartialEncodeMixin:
    @abstractmethod
    async def encode_partial(
        self,
        store_path: StorePath,
        chunk_array: NDBuffer,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_array: Buffer,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Buffer:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_array: Buffer,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[Buffer]:
        pass
