from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numcodecs
import numpy as np
from attr import evolve, frozen
from numcodecs.compat import ensure_bytes, ensure_ndarray

from zarrita.common import (
    ZARRAY_JSON,
    ZATTRS_JSON,
    BytesLike,
    ChunkCoords,
    Selection,
    SliceSelection,
    concurrent_map,
    to_thread,
)
from zarrita.indexing import BasicIndexer, all_chunk_coords, is_total_slice
from zarrita.metadata import ArrayV2Metadata, RuntimeConfiguration
from zarrita.store import StoreLike, StorePath, make_store_path
from zarrita.sync import sync

if TYPE_CHECKING:
    from zarrita.array import Array


@frozen
class _AsyncArrayProxy:
    array: ArrayV2

    def __getitem__(self, selection: Selection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@frozen
class _AsyncArraySelectionProxy:
    array: ArrayV2
    selection: Selection

    async def get(self) -> np.ndarray:
        return await self.array.get_async(self.selection)

    async def set(self, value: np.ndarray):
        return await self.array.set_async(self.selection, value)


@frozen
class ArrayV2:
    metadata: ArrayV2Metadata
    attributes: Optional[Dict[str, Any]]
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
        filters: Optional[List[Dict[str, Any]]] = None,
        compressor: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> ArrayV2:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZARRAY_JSON).exists_async()

        metadata = ArrayV2Metadata(
            shape=shape,
            dtype=np.dtype(dtype),
            chunks=chunks,
            order=order,
            dimension_separator=dimension_separator,
            fill_value=0 if fill_value is None else fill_value,
            compressor=numcodecs.get_codec(compressor).get_config()
            if compressor is not None
            else None,
            filters=[numcodecs.get_codec(filter).get_config() for filter in filters]
            if filters is not None
            else None,
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
            (store_path / ZARRAY_JSON).get_async(),
            (store_path / ZATTRS_JSON).get_async(),
        )
        assert zarray_bytes is not None
        return cls.from_json(
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
    def from_json(
        cls,
        store_path: StorePath,
        zarray_json: Any,
        zattrs_json: Optional[Any],
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> ArrayV2:
        metadata = ArrayV2Metadata.from_json(zarray_json)
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

        await (self.store_path / ZARRAY_JSON).set_async(self.metadata.to_bytes())
        if self.attributes is not None and len(self.attributes) > 0:
            await (self.store_path / ZATTRS_JSON).set_async(
                json.dumps(self.attributes).encode(),
            )
        else:
            await (self.store_path / ZATTRS_JSON).delete_async()

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
        await concurrent_map(
            [
                (chunk_coords, chunk_selection, out_selection, out)
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            self._read_chunk,
        )

        if out.shape:
            return out
        else:
            return out[()]

    async def _read_chunk(
        self,
        chunk_coords: ChunkCoords,
        chunk_selection: SliceSelection,
        out_selection: SliceSelection,
        out: np.ndarray,
    ):
        store_path = self.store_path / self._encode_chunk_key(chunk_coords)

        chunk_array = await self._decode_chunk(await store_path.get_async())
        if chunk_array is not None:
            tmp = chunk_array[chunk_selection]
            out[out_selection] = tmp
        else:
            out[out_selection] = self.metadata.fill_value

    async def _decode_chunk(
        self, chunk_bytes: Optional[BytesLike]
    ) -> Optional[np.ndarray]:
        if chunk_bytes is None:
            return None

        if self.metadata.compressor is not None:
            compressor = numcodecs.get_codec(self.metadata.compressor)
            chunk_array = ensure_ndarray(
                await to_thread(compressor.decode, chunk_bytes)
            )
        else:
            chunk_array = ensure_ndarray(chunk_bytes)

        # ensure correct dtype
        if str(chunk_array.dtype) != self.metadata.dtype:
            chunk_array = chunk_array.view(self.metadata.dtype)

        # apply filters in reverse order
        if self.metadata.filters is not None:
            for filter_metadata in self.metadata.filters[::-1]:
                filter = numcodecs.get_codec(filter_metadata)
                chunk_array = await to_thread(filter.decode, chunk_array)

        # ensure correct chunk shape
        if chunk_array.shape != self.metadata.chunks:
            chunk_array = chunk_array.reshape(
                self.metadata.chunks,
                order=self.metadata.order,
            )

        return chunk_array

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
        )

    async def _write_chunk(
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

    async def _write_chunk_to_store(
        self, store_path: StorePath, chunk_array: np.ndarray
    ):
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

    async def _encode_chunk(self, chunk_array: np.ndarray) -> Optional[BytesLike]:
        chunk_array = chunk_array.ravel(order=self.metadata.order)

        if self.metadata.filters is not None:
            for filter_metadata in self.metadata.filters:
                filter = numcodecs.get_codec(filter_metadata)
                chunk_array = await to_thread(filter.encode, chunk_array)

        if self.metadata.compressor is not None:
            compressor = numcodecs.get_codec(self.metadata.compressor)
            if (
                not chunk_array.flags.c_contiguous
                and not chunk_array.flags.f_contiguous
            ):
                chunk_array = chunk_array.copy(order="A")
            encoded_chunk_bytes = ensure_bytes(
                await to_thread(compressor.encode, chunk_array)
            )
        else:
            encoded_chunk_bytes = ensure_bytes(chunk_array)

        return encoded_chunk_bytes

    def _encode_chunk_key(self, chunk_coords: ChunkCoords) -> str:
        chunk_identifier = self.metadata.dimension_separator.join(
            map(str, chunk_coords)
        )
        return "0" if chunk_identifier == "" else chunk_identifier

    async def resize_async(self, new_shape: ChunkCoords) -> ArrayV2:
        assert len(new_shape) == len(self.metadata.shape)
        new_metadata = evolve(self.metadata, shape=new_shape)

        # Remove all chunks outside of the new shape
        chunk_shape = self.metadata.chunks
        old_chunk_coords = set(all_chunk_coords(self.metadata.shape, chunk_shape))
        new_chunk_coords = set(all_chunk_coords(new_shape, chunk_shape))

        async def _delete_key(key: str) -> None:
            await (self.store_path / key).delete_async()

        await concurrent_map(
            [
                (self._encode_chunk_key(chunk_coords),)
                for chunk_coords in old_chunk_coords.difference(new_chunk_coords)
            ],
            _delete_key,
        )

        # Write new metadata
        await (self.store_path / ZARRAY_JSON).set_async(new_metadata.to_bytes())
        return evolve(self, metadata=new_metadata)

    def resize(self, new_shape: ChunkCoords) -> ArrayV2:
        return sync(
            self.resize_async(new_shape), self.runtime_configuration.asyncio_loop
        )

    async def convert_to_v3_async(self) -> Array:
        from sys import byteorder as sys_byteorder

        from zarrita.array import Array
        from zarrita.common import ZARR_JSON
        from zarrita.metadata import (
            ArrayMetadata,
            BloscCodecConfigurationMetadata,
            BloscCodecMetadata,
            BytesCodecConfigurationMetadata,
            BytesCodecMetadata,
            CodecMetadata,
            DataType,
            GzipCodecConfigurationMetadata,
            GzipCodecMetadata,
            RegularChunkGridConfigurationMetadata,
            RegularChunkGridMetadata,
            TransposeCodecConfigurationMetadata,
            TransposeCodecMetadata,
            V2ChunkKeyEncodingConfigurationMetadata,
            V2ChunkKeyEncodingMetadata,
            blosc_shuffle_int_to_str,
            dtype_to_data_type,
        )

        data_type = DataType[dtype_to_data_type[self.metadata.dtype.str]]
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

        codecs: List[CodecMetadata] = []

        if self.metadata.order == "F":
            codecs.append(
                TransposeCodecMetadata(
                    configuration=TransposeCodecConfigurationMetadata(order="F")
                )
            )
        codecs.append(
            BytesCodecMetadata(
                configuration=BytesCodecConfigurationMetadata(endian=endian)
            )
        )

        if self.metadata.compressor is not None:
            v2_codec = numcodecs.get_codec(self.metadata.compressor).get_config()
            assert v2_codec["id"] in (
                "blosc",
                "gzip",
            ), "Only blosc and gzip are supported by v3."
            if v2_codec["id"] == "blosc":
                shuffle = blosc_shuffle_int_to_str[v2_codec.get("shuffle", 0)]
                codecs.append(
                    BloscCodecMetadata(
                        configuration=BloscCodecConfigurationMetadata(
                            typesize=data_type.byte_count,
                            cname=v2_codec["cname"],
                            clevel=v2_codec["clevel"],
                            shuffle=shuffle,
                            blocksize=v2_codec.get("blocksize", 0),
                        )
                    )
                )
            elif v2_codec["id"] == "gzip":
                codecs.append(
                    GzipCodecMetadata(
                        configuration=GzipCodecConfigurationMetadata(
                            level=v2_codec.get("level", 5)
                        )
                    )
                )

        new_metadata = ArrayMetadata(
            shape=self.metadata.shape,
            chunk_grid=RegularChunkGridMetadata(
                configuration=RegularChunkGridConfigurationMetadata(
                    chunk_shape=self.metadata.chunks
                )
            ),
            data_type=data_type,
            fill_value=0
            if self.metadata.fill_value is None
            else self.metadata.fill_value,
            chunk_key_encoding=V2ChunkKeyEncodingMetadata(
                configuration=V2ChunkKeyEncodingConfigurationMetadata(
                    separator=self.metadata.dimension_separator
                )
            ),
            codecs=codecs,
            attributes=self.attributes or {},
        )

        new_metadata_bytes = new_metadata.to_bytes()
        await (self.store_path / ZARR_JSON).set_async(new_metadata_bytes)

        return Array.from_json(
            store_path=self.store_path,
            zarr_json=json.loads(new_metadata_bytes),
            runtime_configuration=self.runtime_configuration,
        )

    async def update_attributes_async(self, new_attributes: Dict[str, Any]) -> ArrayV2:
        await (self.store_path / ZATTRS_JSON).set_async(
            json.dumps(new_attributes).encode()
        )
        return evolve(self, attributes=new_attributes)

    def update_attributes(self, new_attributes: Dict[str, Any]) -> ArrayV2:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    def convert_to_v3(self) -> Array:
        return sync(
            self.convert_to_v3_async(), loop=self.runtime_configuration.asyncio_loop
        )

    def __repr__(self):
        return f"<Array_v2 {self.store_path} shape={self.shape} dtype={self.dtype}>"
