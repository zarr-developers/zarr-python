from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numcodecs
import numpy as np
from attr import evolve, frozen, asdict, field
from numcodecs.compat import ensure_bytes, ensure_ndarray
from zarr.v3.array.chunk import (
    read_chunk,
    write_chunk,
)
from zarr.v3.codecs import CodecPipeline, bytes_codec
from zarr.v3.common import (
    ZARRAY_JSON,
    ZATTRS_JSON,
    BytesLike,
    ChunkCoords,
    Selection,
    SliceSelection,
    concurrent_map,
    make_cattr,
    to_thread,
)
from zarr.v3.array.indexing import BasicIndexer, all_chunk_coords, is_total_slice
from zarr.v3.array.base import ChunkMetadata, RuntimeConfiguration
from zarr.v3.metadata import (
    DefaultChunkKeyEncodingConfigurationMetadata,
    DefaultChunkKeyEncodingMetadata,
    V2ChunkKeyEncodingMetadata,
)

from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import sync

if TYPE_CHECKING:
    import zarr.v3.array.v3 as v3


@frozen
class _AsyncArrayProxy:
    array: Array

    def __getitem__(self, selection: Selection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@frozen
class _AsyncArraySelectionProxy:
    array: Array
    selection: Selection

    async def get(self) -> np.ndarray:
        return await self.array.get_async(self.selection)

    async def set(self, value: np.ndarray):
        return await self.array.set_async(self.selection, value)


@frozen
class ArrayMetadata:
    shape: ChunkCoords
    chunks: ChunkCoords
    dtype: np.dtype
    fill_value: Union[None, int, float] = 0
    order: Literal["C", "F"] = "C"
    filters: Optional[List[Dict[str, Any]]] = None
    dimension_separator: Literal[".", "/"] = "."
    compressor: Optional[Dict[str, Any]] = None
    zarr_format: Literal[2] = 2
    attributes: Dict[str, Any] = {}

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            raise TypeError

        return json.dumps(asdict(self), default=_json_convert).encode()

    @classmethod
    def from_json(cls, zarr_json: Any) -> ArrayMetadata:
        return make_cattr().structure(zarr_json, cls)


@frozen
class Array:
    metadata: ArrayMetadata
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration
    attributes: Dict[str, Any]
    chunk_key_encoding: Any

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
    ) -> Array:
        store_path = make_store_path(store)
        if not exists_ok:
            assert not await (store_path / ZARRAY_JSON).exists_async()

        metadata = ArrayMetadata(
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

        chunk_key_encoding = V2ChunkKeyEncodingMetadata(
            configuration=DefaultChunkKeyEncodingConfigurationMetadata(
                separator=metadata.dimension_separator
            )
        )

        array = cls(
            metadata=metadata,
            store_path=store_path,
            attributes=attributes,
            chunk_key_encoding=chunk_key_encoding,
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
    ) -> Array:
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
    ) -> Array:
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
    ) -> Array:
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
    ) -> Array:
        metadata = ArrayMetadata.from_json(zarray_json)

        chunk_key_encoding = V2ChunkKeyEncodingMetadata(
            configuration=DefaultChunkKeyEncodingConfigurationMetadata(
                separator=metadata.dimension_separator
            )
        )

        out = cls(
            store_path=store_path,
            metadata=metadata,
            attributes=zattrs_json,
            chunk_key_encoding=chunk_key_encoding,
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
    def attrs(self) -> dict[str, Any]:
        return self.metadata.attributes

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

        if self.metadata.filters is None:
            filters = []
        else:
            filters = self.metadata.filters

        if self.metadata.compressor is None:
            codecs = [bytes_codec()]
        else:
            codecs = [self.metadata.compressor] + filters

        codec_pipeline = CodecPipeline.from_metadata(
            codecs,
            ChunkMetadata(
                array_shape=self.metadata.shape,
                chunk_shape=self.metadata.chunks,
                dtype=self.metadata.dtype,
                fill_value=self.metadata.fill_value,
            ),
        )

        await read_chunk(
            chunk_key_encoding=self.chunk_key_encoding,
            fill_value=self.metadata.fill_value,
            store_path=self.store_path,
            chunk_coords=chunk_coords,
            chunk_selection=chunk_selection,
            codec_pipeline=codec_pipeline,
            out_selection=out_selection,
            out=out,
            config=self.runtime_configuration,
        )

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

        if self.metadata.filters is None:
            filters = []
        else:
            filters = self.metadata.filters

        if self.metadata.compressor is None:
            codecs = [bytes_codec()]
        else:
            codecs = [self.metadata.compressor] + filters

        codec_pipeline = CodecPipeline.from_metadata(
            codecs,
            ChunkMetadata(
                array_shape=self.metadata.shape,
                chunk_shape=self.metadata.chunks,
                dtype=self.metadata.dtype,
                fill_value=self.metadata.fill_value,
            ),
        )

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
                    self.chunk_key_encoding,
                    self.store_path,
                    codec_pipeline,
                    value,
                    chunk_shape,
                    chunk_coords,
                    chunk_selection,
                    out_selection,
                    self.metadata.fill_value,
                    self.runtime_configuration,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            write_chunk,
        )

    async def resize_async(self, new_shape: ChunkCoords) -> Array:
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

    def resize(self, new_shape: ChunkCoords) -> Array:
        return sync(self.resize_async(new_shape), self.runtime_configuration.asyncio_loop)

    async def convert_to_v3_async(self) -> v3.Array:
        from sys import byteorder as sys_byteorder
        import zarr.v3.array.v3 as v3
        from zarr.v3.common import ZARR_JSON

        from zarr.v3.array.base import (
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
                TransposeCodecMetadata(configuration=TransposeCodecConfigurationMetadata(order="F"))
            )
        codecs.append(
            BytesCodecMetadata(configuration=BytesCodecConfigurationMetadata(endian=endian))
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
                        configuration=GzipCodecConfigurationMetadata(level=v2_codec.get("level", 5))
                    )
                )

        new_metadata = v3.ArrayMetadata(
            shape=self.metadata.shape,
            chunk_grid=RegularChunkGridMetadata(
                configuration=RegularChunkGridConfigurationMetadata(
                    chunk_shape=self.metadata.chunks
                )
            ),
            data_type=data_type,
            fill_value=0 if self.metadata.fill_value is None else self.metadata.fill_value,
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

    async def update_attributes_async(self, new_attributes: Dict[str, Any]) -> Array:
        await (self.store_path / ZATTRS_JSON).set_async(json.dumps(new_attributes).encode())
        return evolve(self, attributes=new_attributes)

    def update_attributes(self, new_attributes: Dict[str, Any]) -> Array:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    def convert_to_v3(self) -> Array:
        return sync(self.convert_to_v3_async(), loop=self.runtime_configuration.asyncio_loop)

    def __repr__(self):
        return f"<Array_v2 {self.store_path} shape={self.shape} dtype={self.dtype}>"
