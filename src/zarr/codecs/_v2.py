from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec
from zarr.common import JSON, ArraySpec, BytesLike, to_thread
from zarr.config import RuntimeConfiguration

import numcodecs
from numcodecs.compat import ensure_bytes, ensure_ndarray


@dataclass(frozen=True)
class V2Compressor(ArrayBytesCodec):
    compressor: dict[str, JSON] | None

    is_fixed_size = False

    async def decode(
        self,
        chunk_bytes: BytesLike,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        if chunk_bytes is None:
            return None

        if self.compressor is not None:
            compressor = numcodecs.get_codec(self.compressor)
            chunk_array = ensure_ndarray(await to_thread(compressor.decode, chunk_bytes))
        else:
            chunk_array = ensure_ndarray(chunk_bytes)

        # ensure correct dtype
        if str(chunk_array.dtype) != chunk_spec.dtype:
            chunk_array = chunk_array.view(chunk_spec.dtype)

        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> BytesLike | None:
        if self.compressor is not None:
            compressor = numcodecs.get_codec(self.compressor)
            if not chunk_array.flags.c_contiguous and not chunk_array.flags.f_contiguous:
                chunk_array = chunk_array.copy(order="A")
            encoded_chunk_bytes = ensure_bytes(await to_thread(compressor.encode, chunk_array))
        else:
            encoded_chunk_bytes = ensure_bytes(chunk_array)

        return encoded_chunk_bytes

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class V2Filters(ArrayArrayCodec):
    filters: list[dict[str, JSON]]
    order: Literal["C", "F"]

    is_fixed_size = False

    async def decode(
        self,
        chunk_array: np.ndarray,
        chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray:
        # apply filters in reverse order
        if self.filters is not None:
            for filter_metadata in self.filters[::-1]:
                filter = numcodecs.get_codec(filter_metadata)
                chunk_array = await to_thread(filter.decode, chunk_array)

        # ensure correct chunk shape
        if chunk_array.shape != chunk_spec.shape:
            chunk_array = chunk_array.reshape(
                chunk_spec.shape,
                order=self.order,
            )

        return chunk_array

    async def encode(
        self,
        chunk_array: np.ndarray,
        _chunk_spec: ArraySpec,
        _runtime_configuration: RuntimeConfiguration,
    ) -> np.ndarray | None:
        chunk_array = chunk_array.ravel(order=self.order)

        for filter_metadata in self.filters:
            filter = numcodecs.get_codec(filter_metadata)
            chunk_array = await to_thread(filter.encode, chunk_array)

        return chunk_array

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
