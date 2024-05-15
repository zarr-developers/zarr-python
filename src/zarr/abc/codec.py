from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Iterable, Protocol, TypeVar, runtime_checkable

import numpy as np
from zarr.abc.metadata import Metadata
from zarr.common import BytesLike


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.common import ArraySpec, SliceSelection
    from zarr.metadata import ArrayMetadata


@runtime_checkable
class ByteGetter(Protocol):
    async def get(self, byte_range: tuple[int, int | None] | None = None) -> BytesLike | None: ...


@runtime_checkable
class ByteSetter(Protocol):
    async def get(self, byte_range: tuple[int, int | None] | None = None) -> BytesLike | None: ...

    async def set(self, value: BytesLike, byte_range: tuple[int, int] | None = None) -> None: ...

    async def delete(self) -> None: ...


CodecInput = TypeVar("CodecInput", bound=np.ndarray | BytesLike)
CodecOutput = TypeVar("CodecOutput", bound=np.ndarray | BytesLike)


class _Codec(Generic[CodecInput, CodecOutput], Metadata):
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

    @abstractmethod
    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[CodecOutput | None, ArraySpec]],
    ) -> Iterable[CodecInput | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[CodecInput | None, ArraySpec]],
    ) -> Iterable[CodecOutput | None]:
        pass


class ArrayArrayCodec(_Codec[np.ndarray, np.ndarray]):
    pass


class ArrayBytesCodec(_Codec[np.ndarray, BytesLike]):
    pass


class BytesBytesCodec(_Codec[BytesLike, BytesLike]):
    pass


Codec = ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec


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


class CodecPipeline(Metadata):
    @abstractmethod
    def evolve(self, array_spec: ArraySpec) -> Self:
        pass

    @classmethod
    @abstractmethod
    def from_list(cls, codecs: list[Codec]) -> Self:
        pass

    @property
    @abstractmethod
    def supports_partial_decode(self) -> bool:
        pass

    @property
    @abstractmethod
    def supports_partial_encode(self) -> bool:
        pass

    @abstractmethod
    def validate(self, array_metadata: ArrayMetadata) -> None:
        pass

    @abstractmethod
    def compute_encoded_size(self, byte_length: int, array_spec: ArraySpec) -> int:
        pass

    @abstractmethod
    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        pass

    @abstractmethod
    async def decode_partial(
        self,
        batch_info: Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        pass

    @abstractmethod
    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
    ) -> Iterable[BytesLike | None]:
        pass

    @abstractmethod
    async def encode_partial(
        self,
        batch_info: Iterable[tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
    ) -> None:
        pass

    @abstractmethod
    async def read(
        self,
        batch_info: Iterable[tuple[ByteGetter, ArraySpec, SliceSelection, SliceSelection]],
        out: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    async def write(
        self,
        batch_info: Iterable[tuple[ByteSetter, ArraySpec, SliceSelection, SliceSelection]],
        value: np.ndarray,
    ) -> None:
        pass
