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
from zarr.config import config


CodecInput = TypeVar("CodecInput", bound=np.ndarray | BytesLike)
CodecOutput = TypeVar("CodecOutput", bound=np.ndarray | BytesLike)


async def batching_helper(
    func: Callable[[CodecInput, ArraySpec], Awaitable[CodecOutput | None]],
    batch_info: Iterable[tuple[CodecInput | None, ArraySpec]],
) -> list[CodecOutput | None]:
    return await concurrent_map(
        [(chunk_array, chunk_spec) for chunk_array, chunk_spec in batch_info],
        noop_for_none(func),
        config.get("async.concurrency"),
    )


def noop_for_none(
    func: Callable[[CodecInput, ArraySpec], Awaitable[CodecOutput | None]],
) -> Callable[[CodecInput | None, ArraySpec], Awaitable[CodecOutput | None]]:
    async def wrap(chunk: CodecInput | None, chunk_spec: ArraySpec) -> CodecOutput | None:
        if chunk is None:
            return None
        return await func(chunk, chunk_spec)

    return wrap


class ArrayArrayCodecBatchMixin(ArrayArrayCodec):
    @abstractmethod
    async def decode_single(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
    ) -> np.ndarray:
        pass

    async def decode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        return await batching_helper(self.decode_single, chunk_arrays_and_specs)

    @abstractmethod
    async def encode_single(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
    ) -> np.ndarray | None:
        pass

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        return await batching_helper(self.encode_single, chunk_arrays_and_specs)


class ArrayBytesCodecBatchMixin(ArrayBytesCodec):
    @abstractmethod
    async def decode_single(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
    ) -> np.ndarray:
        pass

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        return await batching_helper(self.decode_single, chunk_bytes_and_specs)

    @abstractmethod
    async def encode_single(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
    ) -> BytesLike | None:
        pass

    async def encode(
        self,
        chunk_arrays_and_specs: Iterable[tuple[np.ndarray | None, ArraySpec]],
    ) -> Iterable[BytesLike | None]:
        return await batching_helper(self.encode_single, chunk_arrays_and_specs)


class ArrayBytesCodecPartialDecodeBatchMixin(ArrayBytesCodecPartialDecodeMixin):
    @abstractmethod
    async def decode_partial_single(
        self,
        byte_getter: ByteGetter,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
    ) -> np.ndarray | None:
        pass

    async def decode_partial(
        self,
        batch_info: Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]],
    ) -> Iterable[np.ndarray | None]:
        return await concurrent_map(
            [
                (byte_getter, selection, chunk_spec)
                for byte_getter, selection, chunk_spec in batch_info
            ],
            self.decode_partial_single,
            config.get("async.concurrency"),
        )


class ArrayBytesCodecPartialEncodeBatchMixin(ArrayBytesCodecPartialEncodeMixin):
    @abstractmethod
    async def encode_partial_single(
        self,
        byte_setter: ByteSetter,
        chunk_array: np.ndarray,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
    ) -> None:
        pass

    async def encode_partial(
        self,
        batch_info: Iterable[tuple[ByteSetter, np.ndarray, SliceSelection, ArraySpec]],
    ) -> None:
        await concurrent_map(
            [
                (byte_setter, chunk_array, selection, chunk_spec)
                for byte_setter, chunk_array, selection, chunk_spec in batch_info
            ],
            self.encode_partial_single,
            config.get("async.concurrency"),
        )


class BytesBytesCodecBatchMixin(BytesBytesCodec):
    @abstractmethod
    async def decode_single(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
    ) -> BytesLike:
        pass

    async def decode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
    ) -> Iterable[BytesLike | None]:
        return await batching_helper(self.decode_single, chunk_bytes_and_specs)

    @abstractmethod
    async def encode_single(
        self,
        chunk_array: BytesLike,
        chunk_spec: ArraySpec,
    ) -> BytesLike | None:
        pass

    async def encode(
        self,
        chunk_bytes_and_specs: Iterable[tuple[BytesLike | None, ArraySpec]],
    ) -> Iterable[BytesLike | None]:
        return await batching_helper(self.encode_single, chunk_bytes_and_specs)
