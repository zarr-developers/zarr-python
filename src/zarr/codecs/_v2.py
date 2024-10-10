from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numcodecs
from numcodecs.compat import ensure_bytes, ensure_ndarray

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec
from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
from zarr.registry import get_ndbuffer_class

if TYPE_CHECKING:
    import numcodecs.abc

    from zarr.core.array_spec import ArraySpec


@dataclass(frozen=True)
class V2Compressor(ArrayBytesCodec):
    compressor: numcodecs.abc.Codec | None

    is_fixed_size = False

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        if self.compressor is not None:
            chunk_numpy_array = ensure_ndarray(
                await asyncio.to_thread(self.compressor.decode, chunk_bytes.as_array_like())
            )
        else:
            chunk_numpy_array = ensure_ndarray(chunk_bytes.as_array_like())

        # ensure correct dtype
        if str(chunk_numpy_array.dtype) != chunk_spec.dtype and not chunk_spec.dtype.hasobject:
            chunk_numpy_array = chunk_numpy_array.view(chunk_spec.dtype)

        return get_ndbuffer_class().from_numpy_array(chunk_numpy_array)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> Buffer | None:
        chunk_numpy_array = chunk_array.as_numpy_array()
        if self.compressor is not None:
            if (
                not chunk_numpy_array.flags.c_contiguous
                and not chunk_numpy_array.flags.f_contiguous
            ):
                chunk_numpy_array = chunk_numpy_array.copy(order="A")
            encoded_chunk_bytes = ensure_bytes(
                await asyncio.to_thread(self.compressor.encode, chunk_numpy_array)
            )
        else:
            encoded_chunk_bytes = ensure_bytes(chunk_numpy_array)

        return default_buffer_prototype().buffer.from_bytes(encoded_chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class V2Filters(ArrayArrayCodec):
    filters: tuple[numcodecs.abc.Codec, ...] | None

    is_fixed_size = False

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        # apply filters in reverse order
        if self.filters is not None:
            for filter in self.filters[::-1]:
                chunk_ndarray = await asyncio.to_thread(filter.decode, chunk_ndarray)

        # ensure correct chunk shape
        if chunk_ndarray.shape != chunk_spec.shape:
            chunk_ndarray = chunk_ndarray.reshape(
                chunk_spec.shape,
                order=chunk_spec.order,
            )

        return get_ndbuffer_class().from_ndarray_like(chunk_ndarray)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        chunk_ndarray = chunk_array.as_ndarray_like().ravel(order=chunk_spec.order)

        if self.filters is not None:
            for filter in self.filters:
                chunk_ndarray = await asyncio.to_thread(filter.encode, chunk_ndarray)

        return get_ndbuffer_class().from_ndarray_like(chunk_ndarray)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
