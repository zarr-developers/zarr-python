from __future__ import annotations

from abc import abstractmethod
from typing import Awaitable, Callable, Generic, Iterable, TypeVar


from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    ArrayBytesCodecPartialDecodeMixin,
    ArrayBytesCodecPartialEncodeMixin,
    ByteGetter,
    ByteSetter,
    BytesBytesCodec,
)
from zarr.buffer import Buffer, NDBuffer
from zarr.common import ArraySpec, SliceSelection, concurrent_map
from zarr.config import config


CodecInput = TypeVar("CodecInput", bound=NDBuffer | Buffer)
CodecOutput = TypeVar("CodecOutput", bound=NDBuffer | Buffer)


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


class CodecBatchMixin(Generic[CodecInput, CodecOutput]):
    """The default interface from the Codec class expects batches of codecs.
    However, many codec implementation operate on single codecs.
    This mixin provides abstract methods for decode_single and encode_single and
    implements batching through concurrent processing.

    Use ArrayArrayCodecBatchMixin, ArrayBytesCodecBatchMixin and BytesBytesCodecBatchMixin
    for subclassing.
    """

    @abstractmethod
    async def decode_single(self, chunk_data: CodecOutput, chunk_spec: ArraySpec) -> CodecInput:
        pass

    async def decode(
        self, chunk_data_and_specs: Iterable[tuple[CodecOutput | None, ArraySpec]]
    ) -> Iterable[CodecInput | None]:
        return await batching_helper(self.decode_single, chunk_data_and_specs)

    @abstractmethod
    async def encode_single(
        self, chunk_data: CodecInput, chunk_spec: ArraySpec
    ) -> CodecOutput | None:
        pass

    async def encode(
        self, chunk_data_and_specs: Iterable[tuple[CodecInput | None, ArraySpec]]
    ) -> Iterable[CodecOutput | None]:
        return await batching_helper(self.encode_single, chunk_data_and_specs)


class ArrayArrayCodecBatchMixin(CodecBatchMixin[NDBuffer, NDBuffer], ArrayArrayCodec):
    pass


class ArrayBytesCodecBatchMixin(CodecBatchMixin[NDBuffer, Buffer], ArrayBytesCodec):
    pass


class BytesBytesCodecBatchMixin(CodecBatchMixin[Buffer, Buffer], BytesBytesCodec):
    pass


class ArrayBytesCodecPartialDecodeBatchMixin(ArrayBytesCodecPartialDecodeMixin):
    @abstractmethod
    async def decode_partial_single(
        self, byte_getter: ByteGetter, selection: SliceSelection, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        pass

    async def decode_partial(
        self, batch_info: Iterable[tuple[ByteGetter, SliceSelection, ArraySpec]]
    ) -> Iterable[NDBuffer | None]:
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
        chunk_array: NDBuffer,
        selection: SliceSelection,
        chunk_spec: ArraySpec,
    ) -> None:
        pass

    async def encode_partial(
        self, batch_info: Iterable[tuple[ByteSetter, NDBuffer, SliceSelection, ArraySpec]]
    ) -> None:
        await concurrent_map(
            [
                (byte_setter, chunk_array, selection, chunk_spec)
                for byte_setter, chunk_array, selection, chunk_spec in batch_info
            ],
            self.encode_partial_single,
            config.get("async.concurrency"),
        )
