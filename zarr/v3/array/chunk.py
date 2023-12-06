from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from attr import frozen
from zarr.util import is_total_slice
from zarr.v3.common import BytesLike, ChunkCoords, SliceSelection, to_thread
import numpy as np
import numcodecs
from zarr.v3.codecs.sharding import ShardingCodec, CodecPipeline
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
    _chunk_prefix: str = "c"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        if chunk_key == self._chunk_prefix:
            return ()
        return tuple(
            map(int, chunk_key[len(self._chunk_prefix) :].split(self.configuration.separator))
        )

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        return self.configuration.separator.join(map(str, (self._chunk_prefix,) + chunk_coords))


""" @frozen
class V2ChunkKeyEncodingConfigurationMetadata:
    separator: Literal[".", "/"] = "." 
"""


@frozen
class V2ChunkKeyEncodingMetadata:
    configuration: DefaultChunkKeyEncodingConfigurationMetadata = (
        DefaultChunkKeyEncodingConfigurationMetadata()
    )
    name: Literal["v2"] = "v2"

    def decode_chunk_key(self, chunk_key: str) -> ChunkCoords:
        return tuple(map(int, chunk_key.split(self.configuration.separator)))

    def encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.configuration.separator.join(map(str, chunk_coords))
        return "0" if chunk_identifier == "" else chunk_identifier


ChunkKeyEncodingMetadata = DefaultChunkKeyEncodingMetadata


async def write_chunk(
    chunk_key_encoding: ChunkKeyEncodingMetadata,
    store_path: StorePath,
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
                dtype=value.dtype,
            )
            chunk_array.fill(value)
        else:
            chunk_array = value[out_selection]
        await write_chunk_to_store(
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
                dtype=value.dtype,
            )
            chunk_array.fill(fill_value)
        else:
            chunk_array = (await codec_pipeline.decode(chunk_bytes)).copy()  # make a writable copy
        chunk_array[chunk_selection] = value[out_selection]

        await write_chunk_to_store(
            store_path=store_path,
            chunk_array=chunk_array,
            fill_value=fill_value,
            codec_pipeline=codec_pipeline,
        )


async def write_chunk_to_store(
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


async def read_chunk(
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
