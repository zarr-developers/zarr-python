from __future__ import annotations

from abc import abstractmethod
from typing import Awaitable, Callable, Iterable, TypeVar

import numpy as np

from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    ByteGetter,
    ByteSetter,
    BytesBytesCodec,
)
from zarr.common import ArraySpec, BytesLike, SliceSelection, concurrent_map
from zarr.config import RuntimeConfiguration


CodecInput = TypeVar("CodecInput", bound=np.ndarray | BytesLike)
CodecOutput = TypeVar("CodecOutput", bound=np.ndarray | BytesLike)


async def batching_helper(
    func: Callable[[CodecInput, ArraySpec, RuntimeConfiguration], Awaitable[CodecOutput | None]],
    batch_info: Iterable[tuple[CodecInput | None, ArraySpec]],
    runtime_configuration: RuntimeConfiguration,
) -> list[CodecOutput | None]:
    return await concurrent_map(
        [
            (chunk_array, chunk_spec, runtime_configuration)
            for chunk_array, chunk_spec in batch_info
        ],
        noop_for_none(func),
        runtime_configuration.concurrency,
    )


def noop_for_none(
    func: Callable[[CodecInput, ArraySpec, RuntimeConfiguration], Awaitable[CodecOutput | None]],
) -> Callable[[CodecInput | None, ArraySpec, RuntimeConfiguration], Awaitable[CodecOutput | None]]:
    async def wrap(
        chunk: CodecInput | None, chunk_spec: ArraySpec, runtime_configuration: RuntimeConfiguration
    ) -> CodecOutput | None:
        if chunk is None:
            return None
        return await func(chunk, chunk_spec, runtime_configuration)

    return wrap


class ArrayArrayCodecBatchMixin(ArrayArrayCodec):
    @abstractmethod
    async def decode_single(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        pass

    async def decode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        return await batching_helper(
            self.decode_single, chunk_arrays_and_specs, runtime_configuration
        )

    @abstractmethod
    async def encode_single(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray | None:
        pass

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        return await batching_helper(
            self.encode_single, chunk_arrays_and_specs, runtime_configuration
        )


class ArrayBytesCodecBatchMixin(ArrayBytesCodec):
    @abstractmethod
    async def decode_single(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        pass

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        return await batching_helper(
            self.decode_single, chunk_bytes_and_specs, runtime_configuration
        )

    @abstractmethod
    async def encode_single(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike | None:
        pass

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[BytesLike | None]:
        return await batching_helper(
            self.encode_single, chunk_arrays_and_specs, runtime_configuration
        )


class ArrayBytesCodecPartialDecodeBatchMixin(ArrayBytesCodecPartialDecodeMixin):
    @abstractmethod
    async def decode_partial_single(
        self,
        byte_getter: ByteGetter,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray | None:
        pass

    async def decode_partial(
        self,
        batch_info: Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[np.ndarray | None]:
        return await concurrent_map(
            [
                (byte_getter, selection, chunk_spec, runtime_configuration)
                for byte_getter, selection, chunk_spec in batch_info
            ],
            self.decode_partial_single,
            runtime_configuration.concurrency,
        )


class ArrayBytesCodecPartialEncodeBatchMixin(ArrayBytesCodecPartialEncodeMixin):
    @abstractmethod
    async def encode_partial_single(
        self,
        byte_setter: ByteSetter,
        chunk_array: np.ndarray,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        pass

    async def encode_partial(
        self,
        batch_info: Iterable[tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> None:
        await concurrent_map(
            [
                (byte_setter, chunk_array, selection, chunk_spec, runtime_configuration)
                for byte_setter, chunk_array, selection, chunk_spec in batch_info
            ],
            self.encode_partial_single,
            runtime_configuration.concurrency,
        )


class BytesBytesCodecBatchMixin(BytesBytesCodec):
    @abstractmethod
    async def decode_single(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike:
        pass

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[BytesLike | None]:
        return await batching_helper(
            self.decode_single, chunk_bytes_and_specs, runtime_configuration
        )

    @abstractmethod
    async def encode_single(
        self,
        chunk_array: BytesLike,
        chunk_spec: ArraySpec,
        runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike | None:
        pass

    async def encode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
        runtime_configuration: RuntimeConfiguration,
    ) -> Iterable[BytesLike | None]:
        return await batching_helper(
            self.encode_single, chunk_bytes_and_specs, runtime_configuration
        )
