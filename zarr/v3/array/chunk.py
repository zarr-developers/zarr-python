from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from attr import frozen
from zarr.util import is_total_slice
from zarr.v3.array.codecs import CodecPipeline
from zarr.v3.common import BytesLike, ChunkCoords, SliceSelection, to_thread
import numpy as np
import numcodecs
from zarr.v3.sharding import ShardingCodec
from numcodecs.compat import ensure_bytes, ensure_ndarray
from zarr.v3.store import StorePath


@frozen
class RegularChunkGridConfigurationMetadata:
    chunk_shape: ChunkCoords


@frozen
class RegularChunkGridMetadata:
    configuration: RegularChunkGridConfigurationMetadata
    name: Literal["regular"] = "regular"


@frozen
class DefaultChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "/"


@frozen
class DefaultChunkKeyEncodingMetadata:
    configuration: DefaultChunkKeyEncodingConfigurationMetadata = (
        DefaultChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["default"] = "default"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == "c":
            return ()
        return tuple(map(int, chunk_key[1:].split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.configuration.separator.join(map(str, ("c",) + chunk_coords))


@frozen
class V2ChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "."


@frozen
class V2ChunkKeyEncodingMetadata:
    configuration: V2ChunkKeyEncodingConfigurationMetadata = (
        V2ChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["v2"] = "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingMetadata = Union[DefaultChunkKeyEncodingMetadata, V2ChunkKeyEncodingMetadata]


async def read_chunk_v2(
    fill_value: Any,
    chunk_key_encoding: ChunkKeyEncodingMetadata,
    store_path,
    chunks: ChunkCoords,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    compressor: Optional[Dict[str, Any]],
    filters: List[Optional[Dict[str, Any]]],
    dtype: np.dtype,
    order: Literal["C", "F"],
    out: np.ndarray,
):
    store_path = store_path / chunk_key_encoding.encode_chunk_key(chunk_coords)
    chunk_bytes = await store_path.get_async()
    chunk_array = await _decode_chunk_v2(
        compressor=compressor,
        dtype=dtype,
        chunks=chunks,
        filters=filters,
        order=order,
        chunk_bytes=chunk_bytes,
    )
    if chunk_array is not None:
        tmp = chunk_array[chunk_selection]
        out[out_selection] = tmp
    else:
        out[out_selection] = fill_value


async def _decode_chunk_v2(
    compressor: Optional[Dict[str, Any]],
    dtype: np.dtype,
    chunks: ChunkCoords,
    order: Literal["C", "F"],
    filters: Tuple[Any],
    chunk_bytes: Optional[BytesLike],
) -> Optional[np.ndarray]:

    if chunk_bytes is None:
        return None

    if compressor is not None:
        compressor = numcodecs.get_codec(compressor)
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
    value: np.ndarray,
    chunk_key_encoding: ChunkKeyEncodingMetadata,
    store_path: StorePath,
    dtype: np.dtype,
    order: Literal["C", "F"],
    compressor: Any,
    filters: List[Any],
    chunks: ChunkCoords,
    chunk_shape: ChunkCoords,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    fill_value: Any,
    codec_pipeline: Any,
):
    store_path = store_path / chunk_key_encoding.encode_chunk_key(chunk_coords)

    if is_total_slice(chunk_selection, chunk_shape):
        # write entire chunks
        if np.isscalar(value):
            chunk_array = np.empty(
                chunk_shape,
                dtype=dtype,
                order=order,
            )
            chunk_array.fill(value)
        else:
            chunk_array = value[out_selection]

        await _write_chunk_to_store_v2(
            fill_value=fill_value,
            store_path=store_path,
            chunk_array=chunk_array,
            order=order,
            compressor=compressor,
            filters=filters,
            codec_pipeline=codec_pipeline,
        )

    else:
        # writing partial chunks
        # read chunk first
        chunk_bytes = await store_path.get_async()
        tmp = await _decode_chunk_v2(
            compressor=compressor,
            dtype=dtype,
            chunks=chunks,
            order=order,
            filters=filters,
            chunk_bytes=chunk_bytes,
        )
        # tmp2 = await codec_pipeline.decode(chunk_bytes)
        # merge new value
        if tmp is None:
            chunk_array = np.empty(
                chunk_shape,
                dtype=dtype,
                order=order,
            )
            chunk_array.fill(fill_value)
        else:
            chunk_array = tmp.copy(
                order=order,
            )  # make a writable copy
        chunk_array[chunk_selection] = value[out_selection]

        await _write_chunk_to_store_v2(
            fill_value=fill_value,
            store_path=store_path,
            chunk_array=chunk_array,
            order=order,
            compressor=compressor,
            filters=filters,
            codec_pipeline=codec_pipeline,
        )


async def _write_chunk_to_store_v2(
    fill_value: Any,
    store_path: StorePath,
    chunk_array: np.ndarray,
    order: Literal["C", "F"],
    compressor: Dict[str, Any],
    filters,
    codec_pipeline: CodecPipeline,
):
    chunk_bytes: Optional[BytesLike]
    if np.all(chunk_array == fill_value):
        # chunks that only contain fill_value will be removed
        await store_path.delete_async()
    else:
        chunk_bytes_old = await _encode_chunk_v2(
            chunk_array=chunk_array, order=order, compressor=compressor, filters=filters
        )
        chunk_bytes = await codec_pipeline.encode(chunk_array)
        assert chunk_bytes_old == chunk_bytes
        if chunk_bytes is None:
            await store_path.delete_async()
        else:
            await store_path.set_async(chunk_bytes)


async def _encode_chunk_v2(
    chunk_array: np.ndarray,
    order: Literal["C", "F"],
    compressor: Any,
    filters: List[Any],
) -> Optional[BytesLike]:
    chunk_array = chunk_array.ravel(order=order)

    for filter_metadata in filters:
        filter = numcodecs.get_codec(filter_metadata)
        chunk_array = await to_thread(filter.encode, chunk_array)

    if compressor is not None:
        compressor = numcodecs.get_codec(compressor)
        if not chunk_array.flags.c_contiguous and not chunk_array.flags.f_contiguous:
            chunk_array = chunk_array.copy(order="A")
        encoded_chunk_bytes = ensure_bytes(await to_thread(compressor.encode, chunk_array))
    else:
        encoded_chunk_bytes = ensure_bytes(chunk_array)

    return encoded_chunk_bytes


async def _write_chunk_v3(
    chunk_key_encoding: ChunkKeyEncodingMetadata,
    store_path: StorePath,
    dtype: np.dtype,
    codec_pipeline,
    value: np.ndarray,
    chunk_shape: ChunkCoords,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    fill_value: Any,
):
    chunk_key_encoding = chunk_key_encoding
    chunk_key = chunk_key_encoding.encode_chunk_key(chunk_coords)
    store_path = store_path / chunk_key

    if is_total_slice(chunk_selection, chunk_shape):
        # write entire chunks
        if np.isscalar(value):
            chunk_array = np.empty(
                chunk_shape,
                dtype=dtype,
            )
            chunk_array.fill(value)
        else:
            chunk_array = value[out_selection]
        await _write_chunk_to_store_v3(
            store_path=store_path,
            chunk_array=chunk_array,
            codec_pipeline=codec_pipeline,
            fill_value=fill_value,
        )

    elif len(codec_pipeline.codecs) == 1 and isinstance(codec_pipeline.codecs[0], ShardingCodec):
        sharding_codec = codec_pipeline.codecs[0]
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
                dtype=dtype,
            )
            chunk_array.fill(fill_value)
        else:
            chunk_array = (await codec_pipeline.decode(chunk_bytes)).copy()  # make a writable copy
        chunk_array[chunk_selection] = value[out_selection]

        await _write_chunk_to_store_v3(
            store_path=store_path,
            chunk_array=chunk_array,
            fill_value=fill_value,
            codec_pipeline=codec_pipeline,
        )


async def _write_chunk_to_store_v3(
    store_path: StorePath, chunk_array: np.ndarray, fill_value: Any, codec_pipeline: List[Any]
):
    if np.all(chunk_array == fill_value):
        # chunks that only contain fill_value will be removed
        await store_path.delete_async()
    else:
        chunk_bytes = await codec_pipeline.encode(chunk_array)
        if chunk_bytes is None:
            await store_path.delete_async()
        else:
            await store_path.set_async(chunk_bytes)


async def _read_chunk_v3(
    chunk_key_encoding: ChunkKeyEncodingMetadata,
    fill_value: Any,
    store_path,
    codec_pipeline,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    out: np.ndarray,
):
    chunk_key = chunk_key_encoding.encode_chunk_key(chunk_coords)
    store_path = store_path / chunk_key

    if len(codec_pipeline.codecs) == 1 and isinstance(codec_pipeline.codecs[0], ShardingCodec):
        chunk_array = await codec_pipeline.codecs[0].decode_partial(store_path, chunk_selection)
        if chunk_array is not None:
            out[out_selection] = chunk_array
        else:
            out[out_selection] = fill_value
    else:
        chunk_bytes = await store_path.get_async()
        if chunk_bytes is not None:
            chunk_array = await codec_pipeline.decode(chunk_bytes)
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp
        else:
            out[out_selection] = fill_value
