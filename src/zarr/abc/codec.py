from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, Optional, Tuple, TypeVar

import numpy as np
from zarr.abc.metadata import Metadata

from zarr.common import ArraySpec, concurrent_map
from zarr.store import StorePath


if TYPE_CHECKING:
    from typing_extensions import Self
    from zarr.common import BytesLike, SliceSelection
    from zarr.metadata import ArrayMetadata
    from zarr.config import RuntimeConfiguration

T = TypeVar("T")
U = TypeVar("U")


def noop_for_none(
    func: Callable[[Optional[T], ArraySpec, RuntimeConfiguration], Awaitable[U]],
) -> Callable[[T, ArraySpec, RuntimeConfiguration], Awaitable[U]]:
    async def wrap(
        chunk: Optional[T], chunk_spec: ArraySpec, runtime_configuration: RuntimeConfiguration
    ) -> U:
        if chunk is None:
            return None
        return await func(chunk, chunk_spec, runtime_configuration)

    return wrap


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
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        pass

    async def decode_batch(
        self,
        chunk_arrays_and_specs: Iterable[Tuple[np.ndarray, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray]:
        return await concurrent_map(
            [
                (chunk_array, chunk_spec, runtime_configuration)
                for chunk_array, chunk_spec in chunk_arrays_and_specs
            ],
            noop_for_none(self.decode),
            runtime_configuration.concurrency,
        )

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[np.ndarray]:
        pass

    async def encode_batch(
        self,
        chunk_arrays_and_specs: Iterable[Tuple[Optional[np.ndarray], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        return await concurrent_map(
            [
                (chunk_array, chunk_spec, runtime_configuration)
                for chunk_array, chunk_spec in chunk_arrays_and_specs
            ],
            noop_for_none(self.encode),
            runtime_configuration.concurrency,
        )


class ArrayBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        pass

    async def decode_batch(
        self,
        chunk_bytes_and_specs: Iterable[Tuple[BytesLike, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray]:
        return await concurrent_map(
            [
                (chunk_bytes, chunk_spec, runtime_configuration)
                for chunk_bytes, chunk_spec in chunk_bytes_and_specs
            ],
            noop_for_none(self.decode),
            runtime_configuration.concurrency,
        )

    @abstractmethod
    async def encode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        pass

    async def encode_batch(
        self,
        chunk_arrays_and_specs: Iterable[Tuple[Optional[np.ndarray], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[BytesLike]]:
        return await concurrent_map(
            [
                (chunk_array, chunk_spec, runtime_configuration)
                for chunk_array, chunk_spec in chunk_arrays_and_specs
            ],
            noop_for_none(self.encode),
            runtime_configuration.concurrency,
        )


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

    async def decode_partial_batched(
        self,
        batch_info: Iterable[Tuple[StorePath, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[np.ndarray]]:
        return await concurrent_map(
            [
                (store_path, selection, chunk_spec, runtime_configuration)
                for store_path, selection, chunk_spec in batch_info
            ],
            self.decode_partial,
            runtime_configuration.concurrency,
        )


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

    async def encode_partial_batched(
        self,
        batch_info: Iterable[Tuple[StorePath, np.ndarray, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        await concurrent_map(
            [
                (store_path, chunk_array, selection, chunk_spec, runtime_configuration)
                for store_path, chunk_array, selection, chunk_spec in batch_info
            ],
            self.encode_partial,
            runtime_configuration.concurrency,
        )


class BytesBytesCodec(Codec):
    @abstractmethod
    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        pass

    async def decode_batch(
        self,
        chunk_bytes_and_specs: Iterable[Tuple[BytesLike, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[BytesLike]:
        return await concurrent_map(
            [
                (chunk_bytes, chunk_spec, runtime_configuration)
                for chunk_bytes, chunk_spec in chunk_bytes_and_specs
            ],
            noop_for_none(self.decode),
            runtime_configuration.concurrency,
        )

    @abstractmethod
    async def encode(
        self,
        chunk_array: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> Optional[BytesLike]:
        pass

    async def encode_batch(
        self,
        chunk_bytes_and_specs: Iterable[Tuple[Optional[BytesLike], ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[Optional[BytesLike]]:
        return await concurrent_map(
            [
                (chunk_bytes, chunk_spec, runtime_configuration)
                for chunk_bytes, chunk_spec in chunk_bytes_and_specs
            ],
            noop_for_none(self.encode),
            runtime_configuration.concurrency,
        )
