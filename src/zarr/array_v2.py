from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numcodecs
import numpy as np


from zarr.common import JSON, ZARRAY_JSON, ZATTRS_JSON, ChunkCoords, Selection, concurrent_map
from zarr.config import RuntimeConfiguration
from zarr.indexing import BasicIndexer, all_chunk_coords
from zarr.metadata import ArrayV2Metadata
from zarr.store import StoreLike, StorePath, make_store_path
from zarr.sync import sync

if TYPE_CHECKING:
    from zarr.array import Array


@dataclass(frozen=True)
class _AsyncArrayProxy:
    array: ArrayV2

    def __getitem__(self, selection: Selection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@dataclass(frozen=True)
class _AsyncArraySelectionProxy:
    array: ArrayV2
    selection: Selection

    async def get(self) -> np.ndarray:
        return await self.array.get_async(self.selection)

    async def set(self, value: np.ndarray):
        return await self.array.set_async(self.selection, value)


@dataclass(frozen=True)
class ArrayV2:
    metadata: ArrayV2Metadata
    attributes: Optional[Dict[str, JSON]]
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration

    @classmethod
    async def create_async(
        cls,
        store: StoreLike,
        *,
        shape: ChunkCoords,
        dtype: np.dtype,
        chunks: ChunkCoords,
        dimension_separator: Literal[".", "/"] = ".",
        fill_value: Optional[Union[None, int, float]] = None,
        order: Literal["C", "F"] = "C",
        filters: Optional[List[Dict[str, JSON]]] = None,
        compressor: Optional[Dict[str, JSON]] = None,
        attributes: Optional[Dict[str, JSON]] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> ArrayV2:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZARRAY_JSON).exists()

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
        )
        array = cls(
            metadata=metadata,
            store_path=store_path,
            attributes=attributes,
            runtime_configuration=runtime_configuration,
        )
        await array._save_metadata()
        return array

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        shape: ChunkCoords,
        dtype: np.dtype,
        chunks: ChunkCoords,
        dimension_separator: Literal[".", "/"] = ".",
        fill_value: Optional[Union[None, int, float]] = None,
        order: Literal["C", "F"] = "C",
        filters: Optional[List[Dict[str, Any]]] = None,
        compressor: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> ArrayV2:
        return sync(
            cls.create_async(
                store,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                order=order,
                dimension_separator=dimension_separator,
                fill_value=0 if fill_value is None else fill_value,
                compressor=compressor,
                filters=filters,
                attributes=attributes,
                exists_ok=exists_ok,
                runtime_configuration=runtime_configuration,
            ),
            runtime_configuration.asyncio_loop,
        )

    @classmethod
    async def open_async(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> ArrayV2:
        store_path = make_store_path(store)
        zarray_bytes, zattrs_bytes = await asyncio.gather(
            (store_path / ZARRAY_JSON).get(),
            (store_path / ZATTRS_JSON).get(),
        )
        assert zarray_bytes is not None
        return cls.from_dict(
            store_path,
            zarray_json=json.loads(zarray_bytes),
            zattrs_json=json.loads(zattrs_bytes) if zattrs_bytes is not None else None,
            runtime_configuration=runtime_configuration,
        )

    @classmethod
    def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> ArrayV2:
        return sync(
            cls.open_async(store, runtime_configuration),
            runtime_configuration.asyncio_loop,
        )

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        zarray_json: Any,
        zattrs_json: Optional[Any],
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> ArrayV2:
        metadata = ArrayV2Metadata.from_dict(zarray_json)
        out = cls(
            store_path=store_path,
            metadata=metadata,
            attributes=zattrs_json,
            runtime_configuration=runtime_configuration,
        )
        out._validate_metadata()
        return out

    async def _save_metadata(self) -> None:
        self._validate_metadata()

        await (self.store_path / ZARRAY_JSON).set(self.metadata.to_bytes())
        if self.attributes is not None and len(self.attributes) > 0:
            await (self.store_path / ZATTRS_JSON).set(
                json.dumps(self.attributes).encode(),
            )
        else:
            await (self.store_path / ZATTRS_JSON).delete()

    def _validate_metadata(self) -> None:
        assert len(self.metadata.shape) == len(
            self.metadata.chunks
        ), "`chunks` and `shape` need to have the same number of dimensions."

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def shape(self) -> ChunkCoords:
        return self.metadata.shape

    @property
    def dtype(self) -> np.dtype:
        return self.metadata.dtype

    @property
    def async_(self) -> _AsyncArrayProxy:
        return _AsyncArrayProxy(self)

    def __getitem__(self, selection: Selection):
        return sync(self.get_async(selection), self.runtime_configuration.asyncio_loop)

    async def get_async(self, selection: Selection):
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=self.metadata.chunks,
        )

        # setup output array
        out = np.zeros(
            indexer.shape,
            dtype=self.metadata.dtype,
            order=self.metadata.order,
        )

        # reading chunks and decoding them
        await self.metadata.codecs.read_batch(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    self.metadata.get_chunk_spec(chunk_coords),
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            out,
            self.runtime_configuration,
        )

        if out.shape:
            return out
        else:
            return out[()]

    def __setitem__(self, selection: Selection, value: np.ndarray) -> None:
        sync(self.set_async(selection, value), self.runtime_configuration.asyncio_loop)

    async def set_async(self, selection: Selection, value: np.ndarray) -> None:
        chunk_shape = self.metadata.chunks
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_shape=chunk_shape,
        )

        sel_shape = indexer.shape

        # check value shape
        if np.isscalar(value):
            # setting a scalar value
            pass
        else:
            if not hasattr(value, "shape"):
                value = np.asarray(value, self.metadata.dtype)
            assert value.shape == sel_shape
            if value.dtype != self.metadata.dtype:
                value = value.astype(self.metadata.dtype, order="A")

        # merging with existing data and encoding chunks
        await self.metadata.codecs.write_batch(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    self.metadata.get_chunk_spec(chunk_coords),
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            value,
            self.runtime_configuration,
        )

    async def resize_async(self, new_shape: ChunkCoords) -> ArrayV2:
        assert len(new_shape) == len(self.metadata.shape)
        new_metadata = replace(self.metadata, shape=new_shape)

        # Remove all chunks outside of the new shape
        chunk_shape = self.metadata.chunks
        old_chunk_coords = set(all_chunk_coords(self.metadata.shape, chunk_shape))
        new_chunk_coords = set(all_chunk_coords(new_shape, chunk_shape))

        async def _delete_key(key: str) -> None:
            await (self.store_path / key).delete()

        await concurrent_map(
            [
                (self.metadata.encode_chunk_key(chunk_coords),)
                for chunk_coords in old_chunk_coords.difference(new_chunk_coords)
            ],
            _delete_key,
        )

        # Write new metadata
        await (self.store_path / ZARRAY_JSON).set(new_metadata.to_bytes())
        return replace(self, metadata=new_metadata)

    def resize(self, new_shape: ChunkCoords) -> ArrayV2:
        return sync(self.resize_async(new_shape), self.runtime_configuration.asyncio_loop)

    async def convert_to_v3_async(self) -> Array:
        from sys import byteorder as sys_byteorder

        from zarr.abc.codec import Codec
        from zarr.array import Array
        from zarr.common import ZARR_JSON
        from zarr.chunk_grids import RegularChunkGrid
        from zarr.chunk_key_encodings import V2ChunkKeyEncoding
        from zarr.metadata import ArrayMetadata, DataType

        from zarr.codecs import (
            BloscCodec,
            BloscShuffle,
            BytesCodec,
            GzipCodec,
            TransposeCodec,
        )

        data_type = DataType.from_dtype(self.metadata.dtype)
        endian: Literal["little", "big"]
        if self.metadata.dtype.byteorder == "=":
            endian = sys_byteorder
        elif self.metadata.dtype.byteorder == ">":
            endian = "big"
        else:
            endian = "little"

        assert (
            self.metadata.filters is None or len(self.metadata.filters) == 0
        ), "Filters are not supported by v3."

        codecs: List[Codec] = []

        if self.metadata.order == "F":
            codecs.append(TransposeCodec(order=tuple(reversed(range(self.metadata.ndim)))))
        codecs.append(BytesCodec(endian=endian))

        if self.metadata.compressor is not None:
            v2_codec = numcodecs.get_codec(self.metadata.compressor).get_config()
            assert v2_codec["id"] in (
                "blosc",
                "gzip",
            ), "Only blosc and gzip are supported by v3."
            if v2_codec["id"] == "blosc":
                codecs.append(
                    BloscCodec(
                        typesize=data_type.byte_count,
                        cname=v2_codec["cname"],
                        clevel=v2_codec["clevel"],
                        shuffle=BloscShuffle.from_int(v2_codec.get("shuffle", 0)),
                        blocksize=v2_codec.get("blocksize", 0),
                    )
                )
            elif v2_codec["id"] == "gzip":
                codecs.append(GzipCodec(level=v2_codec.get("level", 5)))

        new_metadata = ArrayMetadata(
            shape=self.metadata.shape,
            chunk_grid=RegularChunkGrid(chunk_shape=self.metadata.chunks),
            data_type=data_type,
            fill_value=0 if self.metadata.fill_value is None else self.metadata.fill_value,
            chunk_key_encoding=V2ChunkKeyEncoding(separator=self.metadata.dimension_separator),
            codecs=codecs,
            attributes=self.attributes or {},
            dimension_names=None,
        )

        new_metadata_bytes = new_metadata.to_bytes()
        await (self.store_path / ZARR_JSON).set(new_metadata_bytes)

        return Array.from_dict(
            store_path=self.store_path,
            data=json.loads(new_metadata_bytes),
            runtime_configuration=self.runtime_configuration,
        )

    async def update_attributes_async(self, new_attributes: Dict[str, Any]) -> ArrayV2:
        await (self.store_path / ZATTRS_JSON).set(json.dumps(new_attributes).encode())
        return replace(self, attributes=new_attributes)

    def update_attributes(self, new_attributes: Dict[str, Any]) -> ArrayV2:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    def convert_to_v3(self) -> Array:
        return sync(self.convert_to_v3_async(), loop=self.runtime_configuration.asyncio_loop)

    def __repr__(self):
        return f"<Array_v2 {self.store_path} shape={self.shape} dtype={self.dtype}>"
