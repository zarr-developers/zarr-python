from __future__ import annotations

from dataclasses import dataclass

import numcodecs
from numcodecs.compat import ensure_bytes, ensure_ndarray

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec
from zarr.array_spec import ArraySpec
from zarr.buffer import Buffer, NDBuffer, cpu
from zarr.common import JSON, to_thread


@dataclass(frozen=True)
class V2Compressor(ArrayBytesCodec):
    compressor: dict[str, JSON] | None

    is_fixed_size = False

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        if self.compressor is not None:
            compressor = numcodecs.get_codec(self.compressor)
            chunk_numpy_array = ensure_ndarray(
                await to_thread(compressor.decode, chunk_bytes.as_array_like())
            )
        else:
            chunk_numpy_array = ensure_ndarray(chunk_bytes.as_array_like())

        # ensure correct dtype
        if str(chunk_numpy_array.dtype) != chunk_spec.dtype:
            chunk_numpy_array = chunk_numpy_array.view(chunk_spec.dtype)

        return cpu.NDBuffer.from_numpy_array(chunk_numpy_array)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        _chunk_spec: ArraySpec,
    ) -> Buffer | None:
        chunk_numpy_array = chunk_array.as_numpy_array()
        if self.compressor is not None:
            compressor = numcodecs.get_codec(self.compressor)
            if (
                not chunk_numpy_array.flags.c_contiguous
                and not chunk_numpy_array.flags.f_contiguous
            ):
                chunk_numpy_array = chunk_numpy_array.copy(order="A")
            encoded_chunk_bytes = ensure_bytes(
                await to_thread(compressor.encode, chunk_numpy_array)
            )
        else:
            encoded_chunk_bytes = ensure_bytes(chunk_numpy_array)

        return cpu.Buffer.from_bytes(encoded_chunk_bytes)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class V2Filters(ArrayArrayCodec):
    filters: list[dict[str, JSON]]

    is_fixed_size = False

    async def _decode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer:
        chunk_ndarray = chunk_array.as_ndarray_like()
        # apply filters in reverse order
        if self.filters is not None:
            for filter_metadata in self.filters[::-1]:
                filter = numcodecs.get_codec(filter_metadata)
                chunk_ndarray = await to_thread(filter.decode, chunk_ndarray)

        # ensure correct chunk shape
        if chunk_ndarray.shape != chunk_spec.shape:
            chunk_ndarray = chunk_ndarray.reshape(
                chunk_spec.shape,
                order=chunk_spec.order,
            )

        return cpu.NDBuffer.from_ndarray_like(chunk_ndarray)

    async def _encode_single(
        self,
        chunk_array: NDBuffer,
        chunk_spec: ArraySpec,
    ) -> NDBuffer | None:
        chunk_ndarray = chunk_array.as_ndarray_like().ravel(order=chunk_spec.order)

        for filter_metadata in self.filters:
            filter = numcodecs.get_codec(filter_metadata)
            chunk_ndarray = await to_thread(filter.encode, chunk_ndarray)

        return cpu.NDBuffer.from_ndarray_like(chunk_ndarray)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
