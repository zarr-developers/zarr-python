from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

import numpy as np
from zarr.abc.metadata import Metadata


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.common import ArraySpec, BytesLike, SliceSelection
    from zarr.metadata import ArrayMetadata


@runtime_checkable
class ByteGetter(Protocol):
    async def get(self, byte_range: tuple[int, int | None] | None = None) -> BytesLike | None: ...


@runtime_checkable
class ByteSetter(Protocol):
    async def get(self, byte_range: tuple[int, int | None] | None = None) -> BytesLike | None: ...

    async def set(self, value: BytesLike, byte_range: tuple[int, int] | None = None) -> None: ...

    async def delete(self) -> None: ...


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
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        pass


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
    ) -> Iterable[BytesLike | None]:
        pass


class ArrayBytesCodecPartialDecodeMixin:
    @abstractmethod
    async def decode_partial(
        self,
        batch_info: Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        pass


class ArrayBytesCodecPartialEncodeMixin:
    @abstractmethod
    async def encode_partial(
        self,
        batch_info: Iterable[tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
    ) -> None:
        pass


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
    ) -> Iterable[BytesLike | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
    ) -> Iterable[BytesLike | None]:
        pass
