from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable

import numpy as np
from zarr.abc.metadata import Metadata
from zarr.abc.store import ByteGetter, ByteSetter


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.common import ArraySpec, BytesLike, SliceSelection
    from zarr.metadata import ArrayV3Metadata
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

    def validate(self, array_metadata: ArrayV3Metadata) -> None:
        pass


class ArrayArrayCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[BytesLike | None]:
        pass


class ArrayBytesCodecPartialDecodeMixin:
    @abstractmethod
    async def decode_partial(
        self,
        batch_info: Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        pass


class ArrayBytesCodecPartialEncodeMixin:
    @abstractmethod
    async def encode_partial(
        self,
        batch_info: Iterable[tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[BytesLike | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[BytesLike | None]:
        pass
