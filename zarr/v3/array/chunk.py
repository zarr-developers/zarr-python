from typing import Any, Literal, Optional, Tuple
from zarr.util import is_total_slice
from zarr.v3.array.base import ChunkKeyEncodingMetadata
from zarr.v3.common import BytesLike, ChunkCoords, SliceSelection, to_thread
import numpy as np
import numcodecs
from zarr.v3.sharding import ShardingCodec
from numcodecs.compat import ensure_bytes, ensure_ndarray
from zarr.v3.store import StorePath


async def _read_chunk_v2(
    fill_value: Any,
    chunk_key_encoding: ChunkKeyEncodingMetadata,
    store_path,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    out: np.ndarray,
):
    store_path = store_path / chunk_key_encoding.encode_chunk_key(chunk_coords)

    chunk_array = await _decode_chunk_v2(await store_path.get_async())
    if chunk_array is not None:
        tmp = chunk_array[chunk_selection]
        out[out_selection] = tmp
    else:
        out[out_selection] = fill_value


async def _decode_chunk_v2(
    self,
    dtype: np.dtype,
    chunks: ChunkCoords,
    order: Literal["C", "F"],
    filters: Tuple[Any],
    chunk_bytes: Optional[BytesLike],
) -> Optional[np.ndarray]:
    if chunk_bytes is None:
        return None

    if self.metadata.compressor is not None:
        compressor = numcodecs.get_codec(self.metadata.compressor)
        chunk_array = ensure_ndarray(await to_thread(compressor.decode, chunk_bytes))
    else:
        chunk_array = ensure_ndarray(chunk_bytes)

    # ensure correct dtype
    if str(chunk_array.dtype) != dtype:
        chunk_array = chunk_array.view(dtype)

    # apply filters in reverse order
    for filter_metadata in reversed(filters):
        filter = numcodecs.get_codec(filter_metadata)
        chunk_array = await to_thread(filter.decode, chunk_array)

    # ensure correct chunk shape
    if chunk_array.shape != chunks:
        chunk_array = chunk_array.reshape(
            chunks,
            order=order,
        )

    return chunk_array


async def _write_chunk_v2(
    self,
    value: np.ndarray,
    chunk_shape: ChunkCoords,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
):
    store_path = self.store_path / self._encode_chunk_key(chunk_coords)

    if is_total_slice(chunk_selection, chunk_shape):
        # write entire chunks
        if np.isscalar(value):
            chunk_array = np.empty(
                chunk_shape,
                dtype=self.metadata.dtype,
                order=self.metadata.order,
            )
            chunk_array.fill(value)
        else:
            chunk_array = value[out_selection]
        await self._write_chunk_to_store(store_path, chunk_array)

    else:
        # writing partial chunks
        # read chunk first
        tmp = await self._decode_chunk(await store_path.get_async())

        # merge new value
        if tmp is None:
            chunk_array = np.empty(
                chunk_shape,
                dtype=self.metadata.dtype,
                order=self.metadata.order,
            )
            chunk_array.fill(self.metadata.fill_value)
        else:
            chunk_array = tmp.copy(
                order=self.metadata.order,
            )  # make a writable copy
        chunk_array[chunk_selection] = value[out_selection]

        await self._write_chunk_to_store(store_path, chunk_array)


async def _write_chunk_to_store_v2(self, store_path: StorePath, chunk_array: np.ndarray):
    chunk_bytes: Optional[BytesLike]
    if np.all(chunk_array == self.metadata.fill_value):
        # chunks that only contain fill_value will be removed
        await store_path.delete_async()
    else:
        chunk_bytes = await self._encode_chunk(chunk_array)
        if chunk_bytes is None:
            await store_path.delete_async()
        else:
            await store_path.set_async(chunk_bytes)


async def _encode_chunk_v2(self, chunk_array: np.ndarray) -> Optional[BytesLike]:
    chunk_array = chunk_array.ravel(order=self.metadata.order)

    if self.metadata.filters is not None:
        for filter_metadata in self.metadata.filters:
            filter = numcodecs.get_codec(filter_metadata)
            chunk_array = await to_thread(filter.encode, chunk_array)

    if self.metadata.compressor is not None:
        compressor = numcodecs.get_codec(self.metadata.compressor)
        if not chunk_array.flags.c_contiguous and not chunk_array.flags.f_contiguous:
            chunk_array = chunk_array.copy(order="A")
        encoded_chunk_bytes = ensure_bytes(await to_thread(compressor.encode, chunk_array))
    else:
        encoded_chunk_bytes = ensure_bytes(chunk_array)

    return encoded_chunk_bytes


def _encode_chunk_key_v2(self, chunk_coords: ChunkCoords) -> str:
    chunk_identifier = self.metadata.dimension_separator.join(map(str, chunk_coords))
    return "0" if chunk_identifier == "" else chunk_identifier


async def _write_chunk_v3(
    self,
    value: np.ndarray,
    chunk_shape: ChunkCoords,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
):
    chunk_key_encoding = self.metadata.chunk_key_encoding
    chunk_key = chunk_key_encoding.encode_chunk_key(chunk_coords)
    store_path = self.store_path / chunk_key

    if is_total_slice(chunk_selection, chunk_shape):
        # write entire chunks
        if np.isscalar(value):
            chunk_array = np.empty(
                chunk_shape,
                dtype=self.metadata.dtype,
            )
            chunk_array.fill(value)
        else:
            chunk_array = value[out_selection]
        await self._write_chunk_to_store(store_path, chunk_array)

    elif len(self.codec_pipeline.codecs) == 1 and isinstance(
        self.codec_pipeline.codecs[0], ShardingCodec
    ):
        sharding_codec = self.codec_pipeline.codecs[0]
        # print("encode_partial", chunk_coords, chunk_selection, repr(self))
        await sharding_codec.encode_partial(
            store_path,
            value[out_selection],
            chunk_selection,
        )
    else:
        # writing partial chunks
        # read chunk first
        chunk_bytes = await store_path.get_async()

        # merge new value
        if chunk_bytes is None:
            chunk_array = np.empty(
                chunk_shape,
                dtype=self.metadata.dtype,
            )
            chunk_array.fill(self.metadata.fill_value)
        else:
            chunk_array = (
                await self.codec_pipeline.decode(chunk_bytes)
            ).copy()  # make a writable copy
        chunk_array[chunk_selection] = value[out_selection]

        await self._write_chunk_to_store(store_path, chunk_array)


async def _write_chunk_to_store_v3(self, store_path: StorePath, chunk_array: np.ndarray):
    if np.all(chunk_array == self.metadata.fill_value):
        # chunks that only contain fill_value will be removed
        await store_path.delete_async()
    else:
        chunk_bytes = await self.codec_pipeline.encode(chunk_array)
        if chunk_bytes is None:
            await store_path.delete_async()
        else:
            await store_path.set_async(chunk_bytes)


async def _read_chunk_v3(
    self,
    fill_value: Any,
    store_path,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    out: np.ndarray,
):
    chunk_key_encoding = self.metadata.chunk_key_encoding
    chunk_key = chunk_key_encoding.encode_chunk_key(chunk_coords)
    store_path = self.store_path / chunk_key

    if len(self.codec_pipeline.codecs) == 1 and isinstance(
        self.codec_pipeline.codecs[0], ShardingCodec
    ):
        chunk_array = await self.codec_pipeline.codecs[0].decode_partial(
            store_path, chunk_selection
        )
        if chunk_array is not None:
            out[out_selection] = chunk_array
        else:
            out[out_selection] = fill_value
    else:
        chunk_bytes = await store_path.get_async()
        if chunk_bytes is not None:
            chunk_array = await self.codec_pipeline.decode(chunk_bytes)
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp
        else:
            out[out_selection] = fill_value
