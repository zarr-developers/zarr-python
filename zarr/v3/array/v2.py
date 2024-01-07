from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numcodecs
import numpy as np
import attr
from zarr.v3.array.base import (
    AsynchronousArray,
    ChunkKeyEncodingV2,
    SynchronousArray,
    read_chunk,
    write_chunk,
)

from zarr.v3.array.base import (
    ChunkKeyEncoder,
)
from zarr.v3.codecs import CodecPipeline, bytes_codec
from zarr.v3.common import (
    ZARRAY_JSON,
    ZATTRS_JSON,
    RuntimeConfiguration,
    concurrent_map,
    make_cattr,
    to_thread,
)
from zarr.v3.array.indexing import BasicIndexer, all_chunk_coords, is_total_slice
from zarr.v3.common import ChunkMetadata
from zarr.v3.metadata.v3 import DefaultChunkKeyConfig, DefaultChunkKeyEncoding
import zarr.v3.metadata.v2 as metaV2

from zarr.v3.store import StoreLike, StorePath, make_store_path
from zarr.v3.sync import sync
from zarr.v3.types import Attributes, BytesLike, ChunkCoords, Selection, SliceSelection

if TYPE_CHECKING:
    import zarr.v3.array.v3 as v3


@attr.frozen
class _AsyncArrayProxy:
    array: AsyncArray

    def __getitem__(self, selection: Selection) -> _AsyncArraySelectionProxy:
        return _AsyncArraySelectionProxy(self.array, selection)


@attr.frozen
class _AsyncArraySelectionProxy:
    array: AsyncArray
    selection: Selection

    async def get(self) -> np.ndarray:
        return await self.array.get_async(self.selection)

    async def set(self, value: np.ndarray):
        return await self.array.set_async(self.selection, value)


class ArrayMetadata(metaV2.ArrayMetadata):
    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_bytes(self):
        return self.to_json()

    """
    def to_bytes(self) -> bytes:
        def _json_convert(o):
            if isinstance(o, np.dtype):
                if o.fields is None:
                    return o.str
                else:
                    return o.descr
            raise TypeError

        return json.dumps(attr.asdict(self), default=_json_convert).encode() """

    @classmethod
    def from_json(cls, zarr_json: Any) -> ArrayMetadata:
        return make_cattr().structure(zarr_json, cls)


@attr.frozen
class AsyncArray(AsynchronousArray):
    metadata: ArrayMetadata
    attributes: Attributes
    store_path: StorePath
    runtime_configuration: RuntimeConfiguration
    codec_pipeline: CodecPipeline
    chunk_key_encoding: ChunkKeyEncoder

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def shape(self) -> ChunkCoords:
        return self.metadata.shape

    @property
    def size(self) -> int:
        return np.prod(self.metadata.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.metadata.dtype

    @property
    def attrs(self) -> Attributes:
        return self.attributes

    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        chunks: Tuple[int, ...],
        dimension_separator: Literal[".", "/"] = ".",
        fill_value: Optional[Union[None, int, float]] = None,
        order: Literal["C", "F"] = "C",
        filters: Optional[List[Dict[str, Any]]] = None,
        compressor: Optional[Dict[str, Any]] = None,
        attributes: Attributes = {},
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> AsyncArray:
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

        chunk_key_encoding = ChunkKeyEncodingV2(separator=dimension_separator)

        if metadata.filters is None:
            filters = []
        else:
            filters = metadata.filters

        if metadata.compressor is None:
            codecs = [bytes_codec()]
        else:
            codecs = [metadata.compressor] + filters

        codec_pipeline = CodecPipeline.from_metadata(
            codecs,
            ChunkMetadata(
                array_shape=metadata.shape,
                chunk_shape=metadata.chunks,
                dtype=metadata.dtype,
                fill_value=metadata.fill_value,
            ),
        )

        array = cls(
            metadata=metadata,
            store_path=store_path,
            runtime_configuration=runtime_configuration,
            codec_pipeline=codec_pipeline,
            chunk_key_encoding=chunk_key_encoding,
            attributes=attributes,
        )

        await array._save_metadata()
        return array

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> AsyncArray:
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
    def from_json(
        cls,
        store_path: StorePath,
        zarray_json: Any,
        zattrs_json: Optional[Any],
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> AsyncArray:
        metadata = ArrayMetadata.from_json(zarray_json)

        chunk_key_encoding = DefaultChunkKeyEncoding(
            configuration=DefaultChunkKeyConfig(separator=metadata.dimension_separator)
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

    def __getitem__(self, selection: Selection):
        return sync(self.getitem(selection), self.runtime_configuration.asyncio_loop)

    async def getitem(self, selection: Selection):
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
        out.fill(self.metadata.fill_value)

        chunk_coords, chunk_selections, out_selections = zip(*indexer)
        chunk_keys = map(self.chunk_key_encoding.encode_key, chunk_coords)

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

        await read_chunk(
            chunk_key=self.chunk_key_encoding.encode_key(chunk_coords),
            store_path=self.store_path,
            chunk_selection=chunk_selection,
            codec_pipeline=self.codec_pipeline,
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
                    self.codec_pipeline,
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

    async def resize_async(self, new_shape: ChunkCoords) -> AsyncArray:
        assert len(new_shape) == len(self.metadata.shape)
        new_metadata = attr.evolve(self.metadata, shape=new_shape)

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
        return attr.evolve(self, metadata=new_metadata)

    def resize(self, new_shape: ChunkCoords) -> AsyncArray:
        return sync(self.resize_async(new_shape), self.runtime_configuration.asyncio_loop)

    """     async def convert_to_v3_async(self) -> v3.Array:
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

        return AsyncArray.from_json(
            store_path=self.store_path,
            zarr_json=json.loads(new_metadata_bytes),
            runtime_configuration=self.runtime_configuration,
        )
    """

    async def update_attributes_async(self, new_attributes: Attributes) -> AsyncArray:
        await (self.store_path / ZATTRS_JSON).set_async(json.dumps(new_attributes).encode())
        return attr.evolve(self, attributes=new_attributes)

    def update_attributes(self, new_attributes: Attributes) -> AsyncArray:
        return sync(
            self.update_attributes_async(new_attributes),
            self.runtime_configuration.asyncio_loop,
        )

    """     def convert_to_v3(self) -> AsyncArray:
        return sync(self.convert_to_v3_async(), loop=self.runtime_configuration.asyncio_loop) """

    def __repr__(self):
        return f"<Array_v2 {self.store_path} shape={self.shape} dtype={self.dtype}>"

    async def info(self):
        raise NotImplementedError

    async def setitem(self, key, value):
        raise NotImplementedError


@attr.frozen
class Array(SynchronousArray):
    _async_array: AsyncArray

    @classmethod
    def create(
        cls,
        store: StoreLike,
        *,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        chunks: Tuple[int, ...],
        dimension_separator: Literal[".", "/"] = ".",
        fill_value: Optional[Union[None, int, float]] = None,
        order: Literal["C", "F"] = "C",
        filters: Optional[List[Dict[str, Any]]] = None,
        compressor: Optional[Dict[str, Any]] = None,
        attributes: Attributes = {},
        exists_ok: bool = False,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Array:
        async_array = sync(
            AsyncArray.create(
                store=store,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                fill_value=fill_value,
                dimension_separator=dimension_separator,
                order=order,
                filters=filters,
                compressor=compressor,
                attributes=attributes,
                runtime_configuration=runtime_configuration,
                exists_ok=exists_ok,
            ),
            runtime_configuration.asyncio_loop,
        )
        return cls(async_array)

    @classmethod
    def from_json(
        cls,
        store_path: StorePath,
        zarr_json: Any,
        runtime_configuration: RuntimeConfiguration,
    ) -> Array:
        async_array = AsyncArray.from_json(
            store_path=store_path, zarr_json=zarr_json, runtime_configuration=runtime_configuration
        )
        return cls(async_array)

    @classmethod
    def open(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Array:

        async_array = sync(
            AsyncArray.open(store, runtime_configuration=runtime_configuration),
            runtime_configuration.asyncio_loop,
        )
        async_array._validate_metadata()
        return cls(async_array)

    @classmethod
    def open_auto(
        cls,
        store: StoreLike,
        runtime_configuration: RuntimeConfiguration = RuntimeConfiguration(),
    ) -> Array:  # TODO: Union[Array, ArrayV2]:
        async_array = sync(
            AsyncArray.open_auto(store, runtime_configuration),
            runtime_configuration.asyncio_loop,
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
    def dtype(self) -> np.dtype:
        return self._async_array.dtype

    @property
    def attrs(self) -> dict:
        return self._async_array.attrs

    @property
    def metadata(self) -> ArrayMetadata:
        return self._async_array.metadata

    @property
    def store_path(self) -> str:
        return self._async_array.store_path

    def __getitem__(self, selection: Selection):
        return sync(
            self._async_array.getitem(selection),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def __setitem__(self, selection: Selection, value: np.ndarray) -> None:
        sync(
            self._async_array.setitem(selection, value),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def resize(self, new_shape: ChunkCoords) -> Array:
        return sync(
            self._async_array.resize(new_shape),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def update_attributes(self, new_attributes: Attributes) -> Array:
        return sync(
            self._async_array.update_attributes(new_attributes),
            self._async_array.runtime_configuration.asyncio_loop,
        )

    def __repr__(self):
        return f"<Array {self.store_path} shape={self.shape} dtype={self.dtype}>"

    def info(self):
        return sync(
            self._async_array.info(),
            self._async_array.runtime_configuration.asyncio_loop,
        )
