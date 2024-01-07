from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

import json
from asyncio import AbstractEventLoop
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from zarr.util import is_total_slice
from zarr.v3.abc.store import ReadStore, WriteStore
from zarr.v3.codecs.sharding import CodecPipeline, ShardingCodec
from zarr.v3.common import RuntimeConfiguration, concurrent_map
from zarr.v3.store import Store, StorePath

from zarr.v3.types import Attributes, BytesLike, ChunkCoords, Selection, SliceSelection


class BaseArray(ABC):
    @abstractproperty
    def store_path(self) -> str:  # TODO: rename to `path`?
        """Path to this array in the underlying store."""
        ...

    @abstractproperty
    def dtype(self) -> np.dtype:
        """Data type of the array elements.

        Returns
        -------
        dtype
            array data type
        """
        ...

    @abstractproperty
    def ndim(self) -> int:
        """Number of array dimensions (axes).

        Returns
        -------
        int
            number of array dimensions (axes)
        """
        ...

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions.

        Returns
        -------
        tuple of int
            array dimensions
        """
        ...

    @abstractproperty
    def size(self) -> int:
        """Number of elements in the array.

        Returns
        -------
        int
            number of elements in an array.
        """

    @abstractproperty
    def attrs(self) -> Attributes:
        """Array attributes.

        Returns
        -------
        dict
            user defined attributes
        """
        ...

    @abstractproperty
    def info(self) -> Any:
        """Report some diagnostic information about the array.

        Returns
        -------
        out
        """
        ...


class AsynchronousArray(BaseArray):
    """This class can be implemented as a v2 or v3 array"""

    @classmethod
    @abstractmethod
    async def from_json(cls, zarr_json: Any, store: ReadStore) -> AsynchronousArray:
        ...

    @classmethod
    @abstractmethod
    async def open(cls, store: ReadStore) -> AsynchronousArray:
        ...

    @classmethod
    @abstractmethod
    async def create(cls, store: WriteStore, *, shape, **kwargs) -> AsynchronousArray:
        ...

    @abstractmethod
    async def getitem(self, selection: Selection):
        ...

    @abstractmethod
    async def setitem(self, selection: Selection, value: np.ndarray) -> None:
        ...


class SynchronousArray(BaseArray):
    """
    This class can be implemented as a v2 or v3 array
    """

    @classmethod
    @abstractmethod
    def from_json(cls, zarr_json: Any, store: ReadStore) -> SynchronousArray:
        ...

    @classmethod
    @abstractmethod
    def open(cls, store: ReadStore) -> SynchronousArray:
        ...

    @classmethod
    @abstractmethod
    def create(cls, store: WriteStore, *, shape, **kwargs) -> SynchronousArray:
        ...

    @abstractmethod
    def __getitem__(self, selection: Selection):  # TODO: type as np.ndarray | scalar
        ...

    @abstractmethod
    def __setitem__(self, selection: Selection, value: np.ndarray) -> None:
        ...


async def write_chunk_to_store(
    store_path: StorePath,
    chunk_array: np.ndarray,
    fill_value: Any,
    codec_pipeline: List[Any],
    config: RuntimeConfiguration,
):
    if np.all(chunk_array == fill_value):
        # chunks that only contain fill_value will be removed
        await store_path.delete_async()
    else:
        chunk_bytes = await codec_pipeline.encode(chunk_array, config)
        if chunk_bytes is None:
            await store_path.delete_async()
        else:
            await store_path.set_async(chunk_bytes)


@runtime_checkable
class ChunkKeyEncoder(Protocol):
    def encode_key(self, coords: ChunkCoords, **kwargs) -> str:
        ...

    def decode_key(self, key: str, **kwargs) -> ChunkCoords:
        ...


async def write_chunk(
    chunk_key_encoding: ChunkKeyEncoder,
    store_path: StorePath,
    codec_pipeline,
    value: np.ndarray,
    chunk_shape: ChunkCoords,
    chunk_coords: ChunkCoords,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    fill_value: Any,
    config: RuntimeConfiguration,
):
    chunk_key = chunk_key_encoding.encode_key(chunk_coords)
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
            config=config,
        )

    elif len(codec_pipeline.codecs) == 1 and isinstance(codec_pipeline.codecs[0], ShardingCodec):
        sharding_codec = codec_pipeline.codecs[0]
        # print("encode_partial", chunk_coords, chunk_selection, repr(self))
        await sharding_codec.encode_partial(
            store_path,
            value[out_selection],
            chunk_selection,
            config=config,
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
            chunk_array = (
                await codec_pipeline.decode(chunk_bytes, config=config)
            ).copy()  # make a writable copy
        chunk_array[chunk_selection] = value[out_selection]

        await write_chunk_to_store(
            store_path=store_path,
            chunk_array=chunk_array,
            fill_value=fill_value,
            codec_pipeline=codec_pipeline,
            config=config,
        )


async def read_chunk(
    chunk_key: str,
    store_path,
    codec_pipeline,
    chunk_selection: SliceSelection,
    out_selection: SliceSelection,
    out: np.ndarray,
    config: RuntimeConfiguration,
) -> None:
    store_path = store_path / chunk_key

    if len(codec_pipeline.codecs) == 1 and isinstance(codec_pipeline.codecs[0], ShardingCodec):
        chunk_array = await codec_pipeline.codecs[0].decode_partial(
            store_path, chunk_selection, config=config
        )
        if chunk_array is not None:
            out[out_selection] = chunk_array
    else:
        chunk_bytes = await store_path.get_async()
        if chunk_bytes is not None:
            chunk_array = await codec_pipeline.decode(chunk_bytes, config=config)
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp


async def read_chunks(
    chunk_keys: Iterable[str],
    store_path: Store,
    codec_pipeline: CodecPipeline,
    chunk_selections: Iterable[SliceSelection],
    out_selections: Iterable[SliceSelection],
    out: np.ndarray,
    config: RuntimeConfiguration,
):

    await concurrent_map(
        [
            (
                chunk_key,
                store_path,
                codec_pipeline,
                chunk_selection,
                out_selection,
                out,
                config,
            )
            for chunk_key, chunk_selection, out_selection in zip(
                chunk_keys, chunk_selections, out_selections
            )
        ],
        read_chunk,
        config.concurrency,
    )


class ChunkKeyEncodingV3(ChunkKeyEncoder):
    separator: Literal[".", "/"]
    prefix = "c"

    def __init__(self, separator: Literal[".", "/"]):
        if separator not in (".", "/"):
            raise ValueError(f'Separator must be "." or "/", got {separator}')
        self.separator = separator

    def decode_key(self, key: str) -> ChunkCoords:
        if key == self.prefix:
            return ()
        return tuple(map(int, key[len(self.prefix) :].split(self.separator)))

    def encode_key(self, chunk_coords: ChunkCoords) -> str:
        return self.separator.join(map(str, (self.prefix,) + chunk_coords))


class ChunkKeyEncodingV2(ChunkKeyEncoder):
    separator: Literal["/", "."]

    def __init__(self, separator: Literal[".", "/"]):
        if separator not in (".", "/"):
            raise ValueError(f'Separator must be "." or "/", got {separator}')
        self.separator = separator

    def decode_key(self, key: str) -> ChunkCoords:
        return tuple(map(int, key.split(self.separator)))

    def encode_key(self, coords: ChunkCoords) -> str:
        chunk_identifier = self.separator.join(map(str, coords))
        return "0" if chunk_identifier == "" else chunk_identifier


class ChunkKeyEncoderABC(ABC):
    @abstractmethod
    def encode_key(self, coords: ChunkCoords) -> str:
        pass

    @abstractmethod
    def decode_key(self, key: str) -> ChunkCoords:
        pass
