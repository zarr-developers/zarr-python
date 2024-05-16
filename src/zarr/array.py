# Notes on what I've changed here:
# 1. Split Array into AsyncArray and Array
# 3. Added .size and .attrs methods
# 4. Temporarily disabled the creation of ArrayV2
# 5. Added from_dict to AsyncArray

# Questions to consider:
# 1. Was splitting the array into two classes really necessary?

from __future__ import annotations

from dataclasses import dataclass, replace

import json
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from zarr.abc.codec import Codec


# from zarr.array_v2 import ArrayV2
from zarr.buffer import Buffer, Factory, NDArrayLike, NDBuffer
from zarr.codecs import BytesCodec
from zarr.codecs.pipeline import CodecPipeline
from zarr.common import (
    ZARR_JSON,
    ArraySpec,
    ChunkCoords,
    Selection,
    SliceSelection,
    ZarrFormat,
    concurrent_map,
)
from zarr.config import config

from zarr.indexing import BasicIndexer, all_chunk_coords, is_total_slice
from zarr.chunk_grids import RegularChunkGrid
from zarr.chunk_key_encodings import DefaultChunkKeyEncoding, V2ChunkKeyEncoding
from zarr.metadata import ArrayMetadata, parse_indexing_order
from zarr.store import StoreLike, StorePath, make_store_path
from zarr.sync import sync


def parse_array_metadata(data: Any) -> ArrayMetadata:
    if isinstance(data, ArrayMetadata):
        return data
    elif isinstance(data, dict):
        return ArrayMetadata.from_dict(data)
    else:
        raise TypeError


@dataclass(frozen=True)
class AsyncArray:
    metadata: ArrayMetadata
    store_path: StorePath
    order: Literal["C", "F"]

    @property
    def codecs(self) -> CodecPipeline:
        return self.metadata.codecs

    def __init__(
        self,
        metadata: ArrayMetadata,
        store_path: StorePath,
        order: Literal["C", "F"] | None = None,
    ):
        metadata_parsed = parse_array_metadata(metadata)
        order_parsed = parse_indexing_order(order or config.get("array.order"))

        object.__setattr__(self, "metadata", metadata_parsed)
        object.__setattr__(self, "store_path", store_path)
        object.__setattr__(self, "order", order_parsed)

    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        chunk_shape: ChunkCoords,
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[Iterable[Union[Codec, Dict[str, Any]]]] = None,
        dimension_names: Optional[Iterable[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        zarr_format: ZarrFormat = 3,
    ) -> AsyncArray:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZARR_JSON).exists()

        codecs = list(codecs) if codecs is not None else [BytesCodec()]

        if fill_value is None:
            if dtype == np.dtype("bool"):
                fill_value = False
            else:
                fill_value = 0

        metadata = ArrayMetadata(
            shape=shape,
            data_type=dtype,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            chunk_key_encoding=(
                V2ChunkKeyEncoding(separator=chunk_key_encoding[1])
                if chunk_key_encoding[0] == "v2"
                else DefaultChunkKeyEncoding(separator=chunk_key_encoding[1])
            ),
            fill_value=fill_value,
            codecs=codecs,
            dimension_names=tuple(dimension_names) if dimension_names else None,
            attributes=attributes or {},
        )

        array = cls(
            metadata=metadata,
            store_path=store_path,
        )

        await array._save_metadata()
        return array

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: Dict[str, Any],
    ) -> AsyncArray:
        metadata = ArrayMetadata.from_dict(data)
        async_array = cls(metadata=metadata, store_path=store_path)
        return async_array

    @classmethod
    async def open(
        cls,
        store: StoreLike,
    ) -> AsyncArray:
        store_path = make_store_path(store)
        zarr_json_bytes = await (store_path / ZARR_JSON).get()
        assert zarr_json_bytes is not None
        return cls.from_dict(
            store_path,
            json.loads(zarr_json_bytes.to_bytes()),
        )

    @classmethod
    async def open_auto(
        cls,
        store: StoreLike,
    ) -> AsyncArray:  # TODO: Union[AsyncArray, ArrayV2]
        store_path = make_store_path(store)
        v3_metadata_bytes = await (store_path / ZARR_JSON).get()
        if v3_metadata_bytes is not None:
            return cls.from_dict(
                store_path,
                json.loads(v3_metadata_bytes.to_bytes()),
            )
        else:
            raise ValueError("no v2 support yet")
            # return await ArrayV2.open(store_path)

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def shape(self) -> ChunkCoords:
        return self.metadata.shape

    @property
    def size(self) -> int:
        return np.prod(self.metadata.shape).item()

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.metadata.dtype

    @property
    def attrs(self) -> dict[str, Any]:
        return self.metadata.attributes

    async def getitem(
        self, selection: Selection, *, factory: Factory.Create = NDBuffer.create
    ) -> NDArrayLike:
        assert isinstance(self.metadata.chunk_grid, RegularChunkGrid)
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunk_grid.chunk_shape,
        )

        # setup output array
        out = factory(
            shape=indexer.shape, dtype=self.metadata.dtype, order=self.order, fill_value=0
        )

        # reading chunks and decoding them
        await concurrent_map(
            [
                (chunk_coords, chunk_selection, out_selection, out)
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._read_chunk,
            config.get("async.concurrency"),
        )
        return out.as_ndarray_like()

    async def _save_metadata(self) -> None:
        await (self.store_path / ZARR_JSON).set(Buffer.from_bytes(self.metadata.to_bytes()))

    async def _read_chunk(
        self,
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
        out: NDBuffer,
    ) -> None:
        chunk_spec = self.metadata.get_chunk_spec(chunk_coords, self.order)
        chunk_key_encoding = self.metadata.chunk_key_encoding
        chunk_key = chunk_key_encoding.encode_chunk_key(chunk_coords)
        store_path = self.store_path / chunk_key

        if self.codecs.supports_partial_decode:
            chunk_array = await self.codecs.decode_partial(store_path, chunk_selection, chunk_spec)
            if chunk_array is not None:
                out[out_selection] = chunk_array
            else:
                out[out_selection] = self.metadata.fill_value
        else:
            chunk_bytes = await store_path.get()
            if chunk_bytes is not None:
                chunk_array = await self.codecs.decode(chunk_bytes, chunk_spec)
                tmp = chunk_array[chunk_selection]
                out[out_selection] = tmp
            else:
                out[out_selection] = self.metadata.fill_value

    async def setitem(
        self,
        selection: Selection,
        value: NDArrayLike,
        factory: Factory.NDArrayLike = NDBuffer.from_ndarray_like,
    ) -> None:
        assert isinstance(self.metadata.chunk_grid, RegularChunkGrid)
        chunk_shape = self.metadata.chunk_grid.chunk_shape
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=chunk_shape,
        )

        sel_shape = indexer.shape

        # check value shape
        if np.isscalar(value):
            value = np.asanyarray(value)
        else:
            if not hasattr(value, "shape"):
                value = np.asarray(value, self.metadata.dtype)
            assert value.shape == sel_shape
            if value.dtype.name != self.metadata.dtype.name:
                value = value.astype(self.metadata.dtype, order="A")

        # We accept any ndarray like object from the user and convert it
        # to a NDBuffer (or subclass). From this point onwards, we only pass
        # Buffer and NDBuffer between components.
        value = factory(value)

        # merging with existing data and encoding chunks
        await concurrent_map(
            [
                (
                    value,
                    chunk_shape,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._write_chunk,
            config.get("async.concurrency"),
        )

    async def _write_chunk(
        self,
        value: NDBuffer,
        chunk_shape: ChunkCoords,
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
    ) -> None:
        chunk_spec = self.metadata.get_chunk_spec(chunk_coords, self.order)
        chunk_key_encoding = self.metadata.chunk_key_encoding
        chunk_key = chunk_key_encoding.encode_chunk_key(chunk_coords)
        store_path = self.store_path / chunk_key

        if is_total_slice(chunk_selection, chunk_shape):
            # write entire chunks
            if np.isscalar(value):
                chunk_array = NDBuffer.create(
                    shape=chunk_shape, dtype=self.metadata.dtype, fill_value=value
                )
            else:
                chunk_array = value[out_selection]
            await self._write_chunk_to_store(store_path, chunk_array, chunk_spec)

        elif self.codecs.supports_partial_encode:
            # print("encode_partial", chunk_coords, chunk_selection, repr(self))
            await self.codecs.encode_partial(
                store_path,
                value[out_selection],
                chunk_selection,
                chunk_spec,
            )
        else:
            # writing partial chunks
            # read chunk first
            chunk_bytes = await store_path.get()

            # merge new value
            if chunk_bytes is None:
                chunk_array = NDBuffer.create(
                    shape=chunk_shape,
                    dtype=self.metadata.dtype,
                    fill_value=self.metadata.fill_value,
                )
            else:
                chunk_array = (
                    await self.codecs.decode(chunk_bytes, chunk_spec)
                ).copy()  # make a writable copy
            chunk_array[chunk_selection] = value[out_selection]

            await self._write_chunk_to_store(store_path, chunk_array, chunk_spec)

    async def _write_chunk_to_store(
        self, store_path: StorePath, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> None:
        if chunk_array.all_equal(self.metadata.fill_value):
            # chunks that only contain fill_value will be removed
            await store_path.delete()
        else:
            chunk_bytes = await self.codecs.encode(chunk_array, chunk_spec)
            if chunk_bytes is None:
                await store_path.delete()
            else:
                await store_path.set(chunk_bytes)

    async def resize(self, new_shape: ChunkCoords) -> AsyncArray:
        if len(new_shape) != len(self.metadata.shape):
            raise ValueError(
                "The new shape must have the same number of dimensions "
                + f"(={len(self.metadata.shape)})."
            )

        new_metadata = replace(self.metadata, shape=new_shape)

        # Remove all chunks outside of the new shape
        assert isinstance(self.metadata.chunk_grid, RegularChunkGrid)
        chunk_shape = self.metadata.chunk_grid.chunk_shape
        chunk_key_encoding = self.metadata.chunk_key_encoding
        old_chunk_coords = set(all_chunk_coords(self.metadata.shape, chunk_shape))
        new_chunk_coords = set(all_chunk_coords(new_shape, chunk_shape))

        async def _delete_key(key: str) -> None:
            await (self.store_path / key).delete()

        await concurrent_map(
            [
                (chunk_key_encoding.encode_chunk_key(chunk_coords),)
                for chunk_coords in old_chunk_coords.difference(new_chunk_coords)
            ],
            _delete_key,
            config.get("async.concurrency"),
        )

        # Write new metadata
        await (self.store_path / ZARR_JSON).set(Buffer.from_bytes(new_metadata.to_bytes()))
        return replace(self, metadata=new_metadata)

    async def update_attributes(self, new_attributes: Dict[str, Any]) -> AsyncArray:
        new_metadata = replace(self.metadata, attributes=new_attributes)

        # Write new metadata
        await (self.store_path / ZARR_JSON).set(Buffer.from_bytes(new_metadata.to_bytes()))
        return replace(self, metadata=new_metadata)

    def __repr__(self) -> str:
        return f"<AsyncArray {self.store_path} shape={self.shape} dtype={self.dtype}>"

    async def info(self):
        return NotImplemented


@dataclass(frozen=True)
class Array:
    _async_array: AsyncArray

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        chunk_shape: ChunkCoords,
        fill_value: Optional[Any] = None,
        chunk_key_encoding: Union[
            Tuple[Literal["default"], Literal[".", "/"]],
            Tuple[Literal["v2"], Literal[".", "/"]],
        ] = ("default", "/"),
        codecs: Optional[Iterable[Union[Codec, Dict[str, Any]]]] = None,
        dimension_names: Optional[Iterable[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
    ) -> Array:
        async_array = sync(
            AsyncArray.create(
                store=store,
                shape=shape,
                dtype=dtype,
                chunk_shape=chunk_shape,
                fill_value=fill_value,
                chunk_key_encoding=chunk_key_encoding,
                codecs=codecs,
                dimension_names=dimension_names,
                attributes=attributes,
                exists_ok=exists_ok,
            ),
        )
        return cls(async_array)

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: Dict[str, Any],
    ) -> Array:
        async_array = AsyncArray.from_dict(store_path=store_path, data=data)
        return cls(async_array)

    @classmethod
    def open(
        cls,
        store: StoreLike,
    ) -> Array:
        async_array = sync(AsyncArray.open(store))
        return cls(async_array)

    @classmethod
    def open_auto(
        cls,
        store: StoreLike,
    ) -> Array:  # TODO: Union[Array, ArrayV2]:
        async_array = sync(
            AsyncArray.open_auto(store),
        )
        return cls(async_array)

    @property
    def ndim(self) -> int:
        return self._async_array.ndim

    @property
    def shape(self) -> ChunkCoords:
        return self._async_array.shape

    @property
    def size(self) -> int:
        return self._async_array.size

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._async_array.dtype

    @property
    def attrs(self) -> dict[str, Any]:
        return self._async_array.attrs

    @property
    def metadata(self) -> ArrayMetadata:
        return self._async_array.metadata

    @property
    def store_path(self) -> StorePath:
        return self._async_array.store_path

    @property
    def order(self) -> Literal["C", "F"]:
        return self._async_array.order

    def __getitem__(self, selection: Selection) -> npt.NDArray[Any]:
        return sync(
            self._async_array.getitem(selection),
        )

    def __setitem__(self, selection: Selection, value: npt.NDArray[Any]) -> None:
        sync(
            self._async_array.setitem(selection, value),
        )

    def resize(self, new_shape: ChunkCoords) -> Array:
        return type(self)(
            sync(
                self._async_array.resize(new_shape),
            )
        )

    def update_attributes(self, new_attributes: Dict[str, Any]) -> Array:
        return type(self)(
            sync(
                self._async_array.update_attributes(new_attributes),
            )
        )

    def __repr__(self) -> str:
        return f"<Array {self.store_path} shape={self.shape} dtype={self.dtype}>"

    def info(self):
        return sync(
            self._async_array.info(),
        )
