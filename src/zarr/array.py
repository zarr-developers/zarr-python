from __future__ import annotations

import json

# Notes on what I've changed here:
# 1. Split Array into AsyncArray and Array
# 3. Added .size and .attrs methods
# 4. Temporarily disabled the creation of ArrayV2
# 5. Added from_dict to AsyncArray
# Questions to consider:
# 1. Was splitting the array into two classes really necessary?
from asyncio import gather
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import Codec
from zarr.abc.store import set_or_delete
from zarr.attributes import Attributes
from zarr.buffer import Factory, NDArrayLike, NDBuffer
from zarr.chunk_grids import RegularChunkGrid
from zarr.chunk_key_encodings import ChunkKeyEncoding, DefaultChunkKeyEncoding, V2ChunkKeyEncoding
from zarr.codecs import BytesCodec
from zarr.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ChunkCoords,
    Selection,
    ZarrFormat,
    concurrent_map,
)
from zarr.config import config
from zarr.indexing import BasicIndexer
from zarr.metadata import ArrayMetadata, ArrayV2Metadata, ArrayV3Metadata, parse_indexing_order
from zarr.store import StoreLike, StorePath, make_store_path
from zarr.sync import sync


def parse_array_metadata(data: Any) -> ArrayMetadata:
    if isinstance(data, ArrayMetadata):
        return data
    elif isinstance(data, dict):
        if data["zarr_format"] == 3:
            return ArrayV3Metadata.from_dict(data)
        elif data["zarr_format"] == 2:
            return ArrayV2Metadata.from_dict(data)
    raise TypeError


@dataclass(frozen=True)
class AsyncArray:
    metadata: ArrayMetadata
    store_path: StorePath
    order: Literal["C", "F"]

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
        # v2 and v3
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        zarr_format: ZarrFormat = 3,
        fill_value: Any | None = None,
        attributes: dict[str, JSON] | None = None,
        # v3 only
        chunk_shape: ChunkCoords | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # v2 only
        chunks: ChunkCoords | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
    ) -> AsyncArray:
        store_path = make_store_path(store)

        if chunk_shape is None:
            if chunks is None:
                raise ValueError("Either chunk_shape or chunks needs to be provided.")
            chunk_shape = chunks
        elif chunks is not None:
            raise ValueError("Only one of chunk_shape or chunks must be provided.")

        if zarr_format == 3:
            if dimension_separator is not None:
                raise ValueError(
                    "dimension_separator cannot be used for arrays with version 3. Use chunk_key_encoding instead."
                )
            if order is not None:
                raise ValueError(
                    "order cannot be used for arrays with version 3. Use a transpose codec instead."
                )
            if filters is not None:
                raise ValueError(
                    "filters cannot be used for arrays with version 3. Use array-to-array codecs instead."
                )
            if compressor is not None:
                raise ValueError(
                    "compressor cannot be used for arrays with version 3. Use bytes-to-bytes codecs instead."
                )
            return await cls._create_v3(
                store_path,
                shape=shape,
                dtype=dtype,
                chunk_shape=chunk_shape,
                fill_value=fill_value,
                chunk_key_encoding=chunk_key_encoding,
                codecs=codecs,
                dimension_names=dimension_names,
                attributes=attributes,
                exists_ok=exists_ok,
            )
        elif zarr_format == 2:
            if codecs is not None:
                raise ValueError(
                    "codecs cannot be used for arrays with version 2. Use filters and compressor instead."
                )
            if chunk_key_encoding is not None:
                raise ValueError(
                    "chunk_key_encoding cannot be used for arrays with version 2. Use dimension_separator instead."
                )
            if dimension_names is not None:
                raise ValueError("dimension_names cannot be used for arrays with version 2.")
            return await cls._create_v2(
                store_path,
                shape=shape,
                dtype=dtype,
                chunks=chunk_shape,
                dimension_separator=dimension_separator,
                fill_value=fill_value,
                order=order,
                filters=filters,
                compressor=compressor,
                attributes=attributes,
                exists_ok=exists_ok,
            )
        else:
            raise ValueError(f"Insupported zarr_format. Got: {zarr_format}")

    @classmethod
    async def _create_v3(
        cls,
        store_path: StorePath,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        chunk_shape: ChunkCoords,
        fill_value: Any | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        attributes: dict[str, JSON] | None = None,
        exists_ok: bool = False,
    ) -> AsyncArray:
        if not exists_ok:
            assert not await (store_path / ZARR_JSON).exists()

        codecs = list(codecs) if codecs is not None else [BytesCodec()]

        if fill_value is None:
            if dtype == np.dtype("bool"):
                fill_value = False
            else:
                fill_value = 0

        if chunk_key_encoding is None:
            chunk_key_encoding = ("default", "/")
        assert chunk_key_encoding is not None

        if isinstance(chunk_key_encoding, tuple):
            chunk_key_encoding = (
                V2ChunkKeyEncoding(separator=chunk_key_encoding[1])
                if chunk_key_encoding[0] == "v2"
                else DefaultChunkKeyEncoding(separator=chunk_key_encoding[1])
            )

        metadata = ArrayV3Metadata(
            shape=shape,
            data_type=dtype,
            chunk_grid=RegularChunkGrid(chunk_shape=chunk_shape),
            chunk_key_encoding=chunk_key_encoding,
            fill_value=fill_value,
            codecs=codecs,
            dimension_names=tuple(dimension_names) if dimension_names else None,
            attributes=attributes or {},
        )

        array = cls(metadata=metadata, store_path=store_path)

        await array._save_metadata(metadata)
        return array

    @classmethod
    async def _create_v2(
        cls,
        store_path: StorePath,
        *,
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        chunks: ChunkCoords,
        dimension_separator: Literal[".", "/"] | None = None,
        fill_value: None | int | float = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        attributes: dict[str, JSON] | None = None,
        exists_ok: bool = False,
    ) -> AsyncArray:
        import numcodecs

        if not exists_ok:
            assert not await (store_path / ZARRAY_JSON).exists()

        if order is None:
            order = "C"

        if dimension_separator is None:
            dimension_separator = "."

        metadata = ArrayV2Metadata(
            shape=shape,
            dtype=np.dtype(dtype),
            chunks=chunks,
            order=order,
            dimension_separator=dimension_separator,
            fill_value=0 if fill_value is None else fill_value,
            compressor=(
                numcodecs.get_codec(compressor).get_config() if compressor is not None else None
            ),
            filters=(
                [numcodecs.get_codec(filter).get_config() for filter in filters]
                if filters is not None
                else None
            ),
            attributes=attributes,
        )
        array = cls(metadata=metadata, store_path=store_path)
        await array._save_metadata(metadata)
        return array

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, JSON],
    ) -> AsyncArray:
        metadata = parse_array_metadata(data)
        async_array = cls(metadata=metadata, store_path=store_path)
        return async_array

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        zarr_format: ZarrFormat | None = 3,
    ) -> AsyncArray:
        store_path = make_store_path(store)

        if zarr_format == 2:
            zarray_bytes, zattrs_bytes = await gather(
                (store_path / ZARRAY_JSON).get(), (store_path / ZATTRS_JSON).get()
            )
            if zarray_bytes is None:
                raise KeyError(store_path)  # filenotfounderror?
        elif zarr_format == 3:
            zarr_json_bytes = await (store_path / ZARR_JSON).get()
            if zarr_json_bytes is None:
                raise KeyError(store_path)  # filenotfounderror?
        elif zarr_format is None:
            zarr_json_bytes, zarray_bytes, zattrs_bytes = await gather(
                (store_path / ZARR_JSON).get(),
                (store_path / ZARRAY_JSON).get(),
                (store_path / ZATTRS_JSON).get(),
            )
            if zarr_json_bytes is not None and zarray_bytes is not None:
                # TODO: revisit this exception type
                # alternatively, we could warn and favor v3
                raise ValueError("Both zarr.json and .zarray objects exist")
            if zarr_json_bytes is None and zarray_bytes is None:
                raise KeyError(store_path)  # filenotfounderror?
            # set zarr_format based on which keys were found
            if zarr_json_bytes is not None:
                zarr_format = 3
            else:
                zarr_format = 2
        else:
            raise ValueError(f"unexpected zarr_format: {zarr_format}")

        if zarr_format == 2:
            # V2 arrays are comprised of a .zarray and .zattrs objects
            assert zarray_bytes is not None
            zarray_dict = json.loads(zarray_bytes.to_bytes())
            zattrs_dict = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}
            zarray_dict["attributes"] = zattrs_dict
            return cls(store_path=store_path, metadata=ArrayV2Metadata.from_dict(zarray_dict))
        else:
            # V3 arrays are comprised of a zarr.json object
            assert zarr_json_bytes is not None
            return cls(
                store_path=store_path,
                metadata=ArrayV3Metadata.from_dict(json.loads(zarr_json_bytes.to_bytes())),
            )

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def shape(self) -> ChunkCoords:
        return self.metadata.shape

    @property
    def chunks(self) -> ChunkCoords:
        try:
            return self.metadata.chunk_grid.chunk_shape  # type: ignore
        except AttributeError as err:  # pragma: no cover
            raise AttributeError(
                f"Chunk grid {self.metadata.chunk_grid} array does not have a chunk shape."
            ) from err

    @property
    def size(self) -> int:
        return np.prod(self.metadata.shape).item()

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.metadata.dtype

    @property
    def attrs(self) -> dict[str, JSON]:
        return self.metadata.attributes

    async def getitem(
        self, selection: Selection, *, factory: Factory.Create = NDBuffer.create
    ) -> NDArrayLike:
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
        )

        # setup output array
        out = factory(
            shape=indexer.shape,
            dtype=self.metadata.dtype,
            order=self.order,
            fill_value=0,  # TODO use fill_value
        )

        # reading chunks and decoding them
        await self.metadata.codec_pipeline.read(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    self.metadata.get_chunk_spec(chunk_coords, self.order),
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            out,
        )
        return out.as_ndarray_like()

    async def _save_metadata(self, metadata: ArrayMetadata) -> None:
        to_save = metadata.to_buffer_dict()
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]
        await gather(*awaitables)

    async def setitem(
        self,
        selection: Selection,
        value: NDArrayLike,
        factory: Factory.NDArrayLike = NDBuffer.from_ndarray_like,
    ) -> None:
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
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
        value_buffer = factory(value)

        # merging with existing data and encoding chunks
        await self.metadata.codec_pipeline.write(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    self.metadata.get_chunk_spec(chunk_coords, self.order),
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            value_buffer,
        )

    async def resize(
        self, new_shape: ChunkCoords, delete_outside_chunks: bool = True
    ) -> AsyncArray:
        assert len(new_shape) == len(self.metadata.shape)
        new_metadata = self.metadata.update_shape(new_shape)

        # Remove all chunks outside of the new shape
        old_chunk_coords = set(self.metadata.chunk_grid.all_chunk_coords(self.metadata.shape))
        new_chunk_coords = set(self.metadata.chunk_grid.all_chunk_coords(new_shape))

        if delete_outside_chunks:

            async def _delete_key(key: str) -> None:
                await (self.store_path / key).delete()

            await concurrent_map(
                [
                    (self.metadata.encode_chunk_key(chunk_coords),)
                    for chunk_coords in old_chunk_coords.difference(new_chunk_coords)
                ],
                _delete_key,
                config.get("async.concurrency"),
            )

        # Write new metadata
        await self._save_metadata(new_metadata)
        return replace(self, metadata=new_metadata)

    async def update_attributes(self, new_attributes: dict[str, JSON]) -> AsyncArray:
        new_metadata = self.metadata.update_attributes(new_attributes)

        # Write new metadata
        await self._save_metadata(new_metadata)
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
        # v2 and v3
        shape: ChunkCoords,
        dtype: npt.DTypeLike,
        zarr_format: ZarrFormat = 3,
        fill_value: Any | None = None,
        attributes: dict[str, JSON] | None = None,
        # v3 only
        chunk_shape: ChunkCoords | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # v2 only
        chunks: ChunkCoords | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
    ) -> Array:
        async_array = sync(
            AsyncArray.create(
                store=store,
                shape=shape,
                dtype=dtype,
                zarr_format=zarr_format,
                attributes=attributes,
                fill_value=fill_value,
                chunk_shape=chunk_shape,
                chunk_key_encoding=chunk_key_encoding,
                codecs=codecs,
                dimension_names=dimension_names,
                chunks=chunks,
                dimension_separator=dimension_separator,
                order=order,
                filters=filters,
                compressor=compressor,
                exists_ok=exists_ok,
            ),
        )
        return cls(async_array)

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, JSON],
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

    @property
    def ndim(self) -> int:
        return self._async_array.ndim

    @property
    def shape(self) -> ChunkCoords:
        return self._async_array.shape

    @property
    def chunks(self) -> ChunkCoords:
        return self._async_array.chunks

    @property
    def size(self) -> int:
        return self._async_array.size

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._async_array.dtype

    @property
    def attrs(self) -> Attributes:
        return Attributes(self)

    @property
    def metadata(self) -> ArrayMetadata:
        return self._async_array.metadata

    @property
    def store_path(self) -> StorePath:
        return self._async_array.store_path

    @property
    def order(self) -> Literal["C", "F"]:
        return self._async_array.order

    def __getitem__(self, selection: Selection) -> NDArrayLike:
        return sync(
            self._async_array.getitem(selection),
        )

    def __setitem__(self, selection: Selection, value: NDArrayLike) -> None:
        sync(
            self._async_array.setitem(selection, value),
        )

    def resize(self, new_shape: ChunkCoords) -> Array:
        return type(self)(
            sync(
                self._async_array.resize(new_shape),
            )
        )

    def update_attributes(self, new_attributes: dict[str, JSON]) -> Array:
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
