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
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from zarr.abc.codec import Codec, CodecPipeline
from zarr.abc.store import set_or_delete
from zarr.attributes import Attributes
from zarr.buffer import BufferPrototype, NDArrayLike, NDBuffer, default_buffer_prototype
from zarr.chunk_grids import RegularChunkGrid
from zarr.chunk_key_encodings import ChunkKeyEncoding, DefaultChunkKeyEncoding, V2ChunkKeyEncoding
from zarr.codecs import BytesCodec
from zarr.codecs._v2 import V2Compressor, V2Filters
from zarr.codecs.pipeline import BatchedCodecPipeline
from zarr.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ChunkCoords,
    ZarrFormat,
    concurrent_map,
    product,
)
from zarr.config import config, parse_indexing_order
from zarr.indexing import (
    BasicIndexer,
    BasicSelection,
    BlockIndex,
    BlockIndexer,
    CoordinateIndexer,
    CoordinateSelection,
    Fields,
    Indexer,
    MaskIndexer,
    MaskSelection,
    OIndex,
    OrthogonalIndexer,
    OrthogonalSelection,
    Selection,
    VIndex,
    check_fields,
    check_no_multi_fields,
    is_pure_fancy_indexing,
    is_pure_orthogonal_indexing,
    is_scalar,
    pop_fields,
)
from zarr.metadata import ArrayMetadata, ArrayV2Metadata, ArrayV3Metadata
from zarr.store import StoreLike, StorePath, make_store_path
from zarr.sync import sync


def parse_array_metadata(data: Any) -> ArrayV2Metadata | ArrayV3Metadata:
    if isinstance(data, ArrayV2Metadata | ArrayV3Metadata):
        return data
    elif isinstance(data, dict):
        if data["zarr_format"] == 3:
            return ArrayV3Metadata.from_dict(data)
        elif data["zarr_format"] == 2:
            return ArrayV2Metadata.from_dict(data)
    raise TypeError


def create_codec_pipeline(metadata: ArrayV2Metadata | ArrayV3Metadata) -> BatchedCodecPipeline:
    if isinstance(metadata, ArrayV3Metadata):
        return BatchedCodecPipeline.from_list(metadata.codecs)
    elif isinstance(metadata, ArrayV2Metadata):
        return BatchedCodecPipeline.from_list(
            [V2Filters(metadata.filters or []), V2Compressor(metadata.compressor)]
        )
    else:
        raise AssertionError


@dataclass(frozen=True)
class AsyncArray:
    metadata: ArrayMetadata
    store_path: StorePath
    codec_pipeline: CodecPipeline = field(init=False)
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
        object.__setattr__(self, "codec_pipeline", create_codec_pipeline(metadata=metadata_parsed))

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
        if isinstance(self.metadata.chunk_grid, RegularChunkGrid):
            return self.metadata.chunk_grid.chunk_shape
        else:
            raise ValueError(
                f"chunk attribute is only available for RegularChunkGrid, this array has a {self.metadata.chunk_grid}"
            )

    @property
    def size(self) -> int:
        return np.prod(self.metadata.shape).item()

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.metadata.dtype

    @property
    def attrs(self) -> dict[str, JSON]:
        return self.metadata.attributes

    @property
    def read_only(self) -> bool:
        return bool(~self.store_path.store.writeable)

    @property
    def path(self) -> str:
        """Storage path."""
        return self.store_path.path

    @property
    def name(self) -> str | None:
        """Array name following h5py convention."""
        if self.path:
            # follow h5py convention: add leading slash
            name = self.path
            if name[0] != "/":
                name = "/" + name
            return name
        return None

    @property
    def basename(self) -> str | None:
        """Final component of name."""
        if self.name is not None:
            return self.name.split("/")[-1]
        return None

    async def _get_selection(
        self,
        indexer: Indexer,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLike:
        # check fields are sensible
        out_dtype = check_fields(fields, self.dtype)

        # setup output buffer
        if out is not None:
            if isinstance(out, NDBuffer):
                out_buffer = out
            else:
                raise TypeError(f"out argument needs to be an NDBuffer. Got {type(out)!r}")
            if out_buffer.shape != indexer.shape:
                raise ValueError(
                    f"shape of out argument doesn't match. Expected {indexer.shape}, got {out.shape}"
                )
        else:
            out_buffer = prototype.nd_buffer.create(
                shape=indexer.shape,
                dtype=out_dtype,
                order=self.order,
                fill_value=self.metadata.fill_value,
            )
        if product(indexer.shape) > 0:
            # reading chunks and decoding them
            await self.codec_pipeline.read(
                [
                    (
                        self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                        self.metadata.get_chunk_spec(chunk_coords, self.order, prototype=prototype),
                        chunk_selection,
                        out_selection,
                    )
                    for chunk_coords, chunk_selection, out_selection in indexer
                ],
                out_buffer,
                drop_axes=indexer.drop_axes,
            )
        return out_buffer.as_ndarray_like()

    async def getitem(
        self, selection: BasicSelection, *, prototype: BufferPrototype = default_buffer_prototype
    ) -> NDArrayLike:
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
        )
        return await self._get_selection(indexer, prototype=prototype)

    async def _save_metadata(self, metadata: ArrayMetadata) -> None:
        to_save = metadata.to_buffer_dict()
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]
        await gather(*awaitables)

    async def _set_selection(
        self,
        indexer: Indexer,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        # check fields are sensible
        check_fields(fields, self.dtype)
        fields = check_no_multi_fields(fields)

        # check value shape
        if np.isscalar(value):
            value = np.asanyarray(value, dtype=self.metadata.dtype)
        else:
            if not hasattr(value, "shape"):
                value = np.asarray(value, self.metadata.dtype)
            # assert (
            #     value.shape == indexer.shape
            # ), f"shape of value doesn't match indexer shape. Expected {indexer.shape}, got {value.shape}"
            if not hasattr(value, "dtype") or value.dtype.name != self.metadata.dtype.name:
                value = np.array(value, dtype=self.metadata.dtype, order="A")
        value = cast(NDArrayLike, value)
        # We accept any ndarray like object from the user and convert it
        # to a NDBuffer (or subclass). From this point onwards, we only pass
        # Buffer and NDBuffer between components.
        value_buffer = prototype.nd_buffer.from_ndarray_like(value)

        # merging with existing data and encoding chunks
        await self.codec_pipeline.write(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    self.metadata.get_chunk_spec(chunk_coords, self.order, prototype),
                    chunk_selection,
                    out_selection,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ],
            value_buffer,
            drop_axes=indexer.drop_axes,
        )

    async def setitem(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> None:
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
        )
        return await self._set_selection(indexer, value, prototype=prototype)

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

    async def info(self) -> None:
        raise NotImplementedError


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
    def path(self) -> str:
        """Storage path."""
        return self._async_array.path

    @property
    def name(self) -> str | None:
        """Array name following h5py convention."""
        return self._async_array.name

    @property
    def basename(self) -> str | None:
        """Final component of name."""
        return self._async_array.basename

    @property
    def metadata(self) -> ArrayMetadata:
        return self._async_array.metadata

    @property
    def store_path(self) -> StorePath:
        return self._async_array.store_path

    @property
    def order(self) -> Literal["C", "F"]:
        return self._async_array.order

    @property
    def read_only(self) -> bool:
        return self._async_array.read_only

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> NDArrayLike:
        """
        This method is used by numpy when converting zarr.Array into a numpy array.
        For more information, see https://numpy.org/devdocs/user/basics.interoperability.html#the-array-method
        """
        if copy is False:
            msg = "`copy=False` is not supported. This method always creates a copy."
            raise ValueError(msg)

        arr_np = self[...]

        if dtype is not None:
            arr_np = arr_np.astype(dtype)

        return arr_np

    def __getitem__(self, selection: Selection) -> NDArrayLike:
        """Retrieve data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            An integer index or slice or tuple of int/slice objects specifying the
            requested item or region for each dimension of the array.

        Returns
        -------
        NDArrayLike
             An array-like containing the data for the requested region.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> data = np.arange(100, dtype="uint16")
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=(10,),
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve a single item::

            >>> z[5]
            5

        Retrieve a region via slicing::

            >>> z[:5]
            array([0, 1, 2, 3, 4])
            >>> z[-5:]
            array([95, 96, 97, 98, 99])
            >>> z[5:10]
            array([5, 6, 7, 8, 9])
            >>> z[5:10:2]
            array([5, 7, 9])
            >>> z[::2]
            array([ 0,  2,  4, ..., 94, 96, 98])

        Load the entire array into memory::

            >>> z[...]
            array([ 0,  1,  2, ..., 97, 98, 99])

        Setup a 2-dimensional array::

            >>> data = np.arange(100, dtype="uint16").reshape(10, 10)
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=(10, 10),
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve an item::

            >>> z[2, 2]
            22

        Retrieve a region via slicing::

            >>> z[1:3, 1:3]
            array([[11, 12],
                   [21, 22]])
            >>> z[1:3, :]
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
            >>> z[:, 1:3]
            array([[ 1,  2],
                   [11, 12],
                   [21, 22],
                   [31, 32],
                   [41, 42],
                   [51, 52],
                   [61, 62],
                   [71, 72],
                   [81, 82],
                   [91, 92]])
            >>> z[0:5:2, 0:5:2]
            array([[ 0,  2,  4],
                   [20, 22, 24],
                   [40, 42, 44]])
            >>> z[::2, ::2]
            array([[ 0,  2,  4,  6,  8],
                   [20, 22, 24, 26, 28],
                   [40, 42, 44, 46, 48],
                   [60, 62, 64, 66, 68],
                   [80, 82, 84, 86, 88]])

        Load the entire array into memory::

            >>> z[...]
            array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                   [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                   [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                   [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                   [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                   [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                   [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                   [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])

        Notes
        -----
        Slices with step > 1 are supported, but slices with negative step are not.

        For arrays with a structured dtype, see zarr v2 for examples of how to use
        fields

        Currently the implementation for __getitem__ is provided by
        :func:`vindex` if the indexing is pure fancy indexing (ie a
        broadcast-compatible tuple of integer array indices), or by
        :func:`set_basic_selection` otherwise.

        Effectively, this means that the following indexing modes are supported:

           - integer indexing
           - slice indexing
           - mixed slice and integer indexing
           - boolean indexing
           - fancy indexing (vectorized list of integers)

        For specific indexing options including outer indexing, see the
        methods listed under See Also.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, get_block_selection, set_block_selection,
        vindex, oindex, blocks, __setitem__

        """
        fields, pure_selection = pop_fields(selection)
        if is_pure_fancy_indexing(pure_selection, self.ndim):
            return self.vindex[cast(CoordinateSelection | MaskSelection, selection)]
        elif is_pure_orthogonal_indexing(pure_selection, self.ndim):
            return self.get_orthogonal_selection(pure_selection, fields=fields)
        else:
            return self.get_basic_selection(cast(BasicSelection, pure_selection), fields=fields)

    def __setitem__(self, selection: Selection, value: npt.ArrayLike) -> None:
        """Modify data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            An integer index or slice or tuple of int/slice specifying the requested
            region for each dimension of the array.
        value : npt.ArrayLike
            An array-like containing the data to be stored in the selection.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(
            >>>        shape=(100,),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(5,),
            >>>        dtype="i4",
            >>>       )

        Set all array elements to the same scalar value::

            >>> z[...] = 42
            >>> z[...]
            array([42, 42, 42, ..., 42, 42, 42])

        Set a portion of the array::

            >>> z[:10] = np.arange(10)
            >>> z[-10:] = np.arange(10)[::-1]
            >>> z[...]
            array([ 0, 1, 2, ..., 2, 1, 0])

        Setup a 2-dimensional array::

            >>> z = zarr.zeros(
            >>>        shape=(5, 5),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(5, 5),
            >>>        dtype="i4",
            >>>       )

        Set all array elements to the same scalar value::

            >>> z[...] = 42

        Set a portion of the array::

            >>> z[0, :] = np.arange(z.shape[1])
            >>> z[:, 0] = np.arange(z.shape[0])
            >>> z[...]
            array([[ 0,  1,  2,  3,  4],
                   [ 1, 42, 42, 42, 42],
                   [ 2, 42, 42, 42, 42],
                   [ 3, 42, 42, 42, 42],
                   [ 4, 42, 42, 42, 42]])

        Notes
        -----
        Slices with step > 1 are supported, but slices with negative step are not.

        For arrays with a structured dtype, see zarr v2 for examples of how to use
        fields

        Currently the implementation for __setitem__ is provided by
        :func:`vindex` if the indexing is pure fancy indexing (ie a
        broadcast-compatible tuple of integer array indices), or by
        :func:`set_basic_selection` otherwise.

        Effectively, this means that the following indexing modes are supported:

           - integer indexing
           - slice indexing
           - mixed slice and integer indexing
           - boolean indexing
           - fancy indexing (vectorized list of integers)

        For specific indexing options including outer indexing, see the
        methods listed under See Also.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__

        """
        fields, pure_selection = pop_fields(selection)
        if is_pure_fancy_indexing(pure_selection, self.ndim):
            self.vindex[cast(CoordinateSelection | MaskSelection, selection)] = value
        elif is_pure_orthogonal_indexing(pure_selection, self.ndim):
            self.set_orthogonal_selection(pure_selection, value, fields=fields)
        else:
            self.set_basic_selection(cast(BasicSelection, pure_selection), value, fields=fields)

    def get_basic_selection(
        self,
        selection: BasicSelection = Ellipsis,
        *,
        out: NDBuffer | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
        fields: Fields | None = None,
    ) -> NDArrayLike:
        """Retrieve data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            A tuple specifying the requested item or region for each dimension of the
            array. May be any combination of int and/or slice or ellipsis for multidimensional arrays.
        out : NDBuffer, optional
            If given, load the selected data directly into this buffer.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer to use for the output data. If not provided, the default buffer prototype is used.

        Returns
        -------
        NDArrayLike
            An array-like containing the data for the requested region.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> data = np.arange(100, dtype="uint16")
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=(3,),
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve a single item::

            >>> z.get_basic_selection(5)
            5

        Retrieve a region via slicing::

            >>> z.get_basic_selection(slice(5))
            array([0, 1, 2, 3, 4])
            >>> z.get_basic_selection(slice(-5, None))
            array([95, 96, 97, 98, 99])
            >>> z.get_basic_selection(slice(5, 10))
            array([5, 6, 7, 8, 9])
            >>> z.get_basic_selection(slice(5, 10, 2))
            array([5, 7, 9])
            >>> z.get_basic_selection(slice(None, None, 2))
            array([  0,  2,  4, ..., 94, 96, 98])

        Setup a 3-dimensional array::

            >>> data = np.arange(1000).reshape(10, 10, 10)
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=(5, 5, 5),
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve an item::

            >>> z.get_basic_selection((1, 2, 3))
            123

        Retrieve a region via slicing and Ellipsis::

            >>> z.get_basic_selection((slice(1, 3), slice(1, 3), 0))
            array([[110, 120],
                   [210, 220]])
            >>> z.get_basic_selection(0, (slice(1, 3), slice(None)))
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
            >>> z.get_basic_selection((..., 5))
            array([[  2  12  22  32  42  52  62  72  82  92]
                   [102 112 122 132 142 152 162 172 182 192]
                   ...
                   [802 812 822 832 842 852 862 872 882 892]
                   [902 912 922 932 942 952 962 972 982 992]]

        Notes
        -----
        Slices with step > 1 are supported, but slices with negative step are not.

        For arrays with a structured dtype, see zarr v2 for examples of how to use
        the `fields` parameter.

        This method provides the implementation for accessing data via the
        square bracket notation (__getitem__). See :func:`__getitem__` for examples
        using the alternative notation.

        See Also
        --------
        set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """

        return sync(
            self._async_array._get_selection(
                BasicIndexer(selection, self.shape, self.metadata.chunk_grid),
                out=out,
                fields=fields,
                prototype=prototype,
            )
        )

    def set_basic_selection(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> None:
        """Modify data for an item or region of the array.

        Parameters
        ----------
        selection : tuple
            A tuple specifying the requested item or region for each dimension of the
            array. May be any combination of int and/or slice or ellipsis for multidimensional arrays.
        value : npt.ArrayLike
            An array-like containing values to be stored into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer used for setting the data. If not provided, the
            default buffer prototype is used.

        Examples
        --------
        Setup a 1-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(
            >>>        shape=(100,),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(100,),
            >>>        dtype="i4",
            >>>       )

        Set all array elements to the same scalar value::

            >>> z.set_basic_selection(..., 42)
            >>> z[...]
            array([42, 42, 42, ..., 42, 42, 42])

        Set a portion of the array::

            >>> z.set_basic_selection(slice(10), np.arange(10))
            >>> z.set_basic_selection(slice(-10, None), np.arange(10)[::-1])
            >>> z[...]
            array([ 0, 1, 2, ..., 2, 1, 0])

        Setup a 2-dimensional array::

            >>> z = zarr.zeros(
            >>>        shape=(5, 5),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(5, 5),
            >>>        dtype="i4",
            >>>       )

        Set all array elements to the same scalar value::

            >>> z.set_basic_selection(..., 42)

        Set a portion of the array::

            >>> z.set_basic_selection((0, slice(None)), np.arange(z.shape[1]))
            >>> z.set_basic_selection((slice(None), 0), np.arange(z.shape[0]))
            >>> z[...]
            array([[ 0,  1,  2,  3,  4],
                   [ 1, 42, 42, 42, 42],
                   [ 2, 42, 42, 42, 42],
                   [ 3, 42, 42, 42, 42],
                   [ 4, 42, 42, 42, 42]])

        Notes
        -----
        For arrays with a structured dtype, see zarr v2 for examples of how to use
        the `fields` parameter.

        This method provides the underlying implementation for modifying data via square
        bracket notation, see :func:`__setitem__` for equivalent examples using the
        alternative notation.

        See Also
        --------
        get_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        set_orthogonal_selection, get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        indexer = BasicIndexer(selection, self.shape, self.metadata.chunk_grid)
        sync(self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype))

    def get_orthogonal_selection(
        self,
        selection: OrthogonalSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> NDArrayLike:
        """Retrieve data by making a selection for each dimension of the array. For
        example, if an array has 2 dimensions, allows selecting specific rows and/or
        columns. The selection for each dimension can be either an integer (indexing a
        single item), a slice, an array of integers, or a Boolean array where True
        values indicate a selection.

        Parameters
        ----------
        selection : tuple
            A selection for each dimension of the array. May be any combination of int,
            slice, integer array or Boolean array.
        out : NDBuffer, optional
            If given, load the selected data directly into this buffer.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer to use for the output data. If not provided, the default buffer prototype is used.

        Returns
        -------
        NDArrayLike
            An array-like containing the data for the requested selection.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> data = np.arange(100).reshape(10, 10)
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=data.shape,
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve rows and columns via any combination of int, slice, integer array and/or
        Boolean array::

            >>> z.get_orthogonal_selection(([1, 4], slice(None)))
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
            >>> z.get_orthogonal_selection((slice(None), [1, 4]))
            array([[ 1,  4],
                   [11, 14],
                   [21, 24],
                   [31, 34],
                   [41, 44],
                   [51, 54],
                   [61, 64],
                   [71, 74],
                   [81, 84],
                   [91, 94]])
            >>> z.get_orthogonal_selection(([1, 4], [1, 4]))
            array([[11, 14],
                   [41, 44]])
            >>> sel = np.zeros(z.shape[0], dtype=bool)
            >>> sel[1] = True
            >>> sel[4] = True
            >>> z.get_orthogonal_selection((sel, sel))
            array([[11, 14],
                   [41, 44]])

        For convenience, the orthogonal selection functionality is also available via the
        `oindex` property, e.g.::

            >>> z.oindex[[1, 4], :]
            array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
            >>> z.oindex[:, [1, 4]]
            array([[ 1,  4],
                   [11, 14],
                   [21, 24],
                   [31, 34],
                   [41, 44],
                   [51, 54],
                   [61, 64],
                   [71, 74],
                   [81, 84],
                   [91, 94]])
            >>> z.oindex[[1, 4], [1, 4]]
            array([[11, 14],
                   [41, 44]])
            >>> sel = np.zeros(z.shape[0], dtype=bool)
            >>> sel[1] = True
            >>> sel[4] = True
            >>> z.oindex[sel, sel]
            array([[11, 14],
                   [41, 44]])

        Notes
        -----
        Orthogonal indexing is also known as outer indexing.

        Slices with step > 1 are supported, but slices with negative step are not.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, set_orthogonal_selection,
        get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        indexer = OrthogonalIndexer(selection, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._get_selection(
                indexer=indexer, out=out, fields=fields, prototype=prototype
            )
        )

    def set_orthogonal_selection(
        self,
        selection: OrthogonalSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> None:
        """Modify data via a selection for each dimension of the array.

        Parameters
        ----------
        selection : tuple
            A selection for each dimension of the array. May be any combination of int,
            slice, integer array or Boolean array.
        value : npt.ArrayLike
            An array-like array containing the data to be stored in the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer used for setting the data. If not provided, the
            default buffer prototype is used.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(
            >>>        shape=(5, 5),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(5, 5),
            >>>        dtype="i4",
            >>>       )


        Set data for a selection of rows::

            >>> z.set_orthogonal_selection(([1, 4], slice(None)), 1)
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1]])

        Set data for a selection of columns::

            >>> z.set_orthogonal_selection((slice(None), [1, 4]), 2)
            >>> z[...]
            array([[0, 2, 0, 0, 2],
                   [1, 2, 1, 1, 2],
                   [0, 2, 0, 0, 2],
                   [0, 2, 0, 0, 2],
                   [1, 2, 1, 1, 2]])

        Set data for a selection of rows and columns::

            >>> z.set_orthogonal_selection(([1, 4], [1, 4]), 3)
            >>> z[...]
            array([[0, 2, 0, 0, 2],
                   [1, 3, 1, 1, 3],
                   [0, 2, 0, 0, 2],
                   [0, 2, 0, 0, 2],
                   [1, 3, 1, 1, 3]])

        Set data from a 2D array::

            >>> values = np.arange(10).reshape(2, 5)
            >>> z.set_orthogonal_selection(([0, 3], ...), values)
            >>> z[...]
            array([[0, 1, 2, 3, 4],
                   [1, 3, 1, 1, 3],
                   [0, 2, 0, 0, 2],
                   [5, 6, 7, 8, 9],
                   [1, 3, 1, 1, 3]])

        For convenience, this functionality is also available via the `oindex` property.
        E.g.::

            >>> z.oindex[[1, 4], [1, 4]] = 4
            >>> z[...]
            array([[0, 1, 2, 3, 4],
                   [1, 4, 1, 1, 4],
                   [0, 2, 0, 0, 2],
                   [5, 6, 7, 8, 9],
                   [1, 4, 1, 1, 4]])

        Notes
        -----
        Orthogonal indexing is also known as outer indexing.

        Slices with step > 1 are supported, but slices with negative step are not.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_coordinate_selection, set_coordinate_selection, get_orthogonal_selection,
        get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        indexer = OrthogonalIndexer(selection, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype)
        )

    def get_mask_selection(
        self,
        mask: MaskSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> NDArrayLike:
        """Retrieve a selection of individual items, by providing a Boolean array of the
        same shape as the array against which the selection is being made, where True
        values indicate a selected item.

        Parameters
        ----------
        selection : ndarray, bool
            A Boolean array of the same shape as the array against which the selection is
            being made.
        out : NDBuffer, optional
            If given, load the selected data directly into this buffer.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer to use for the output data. If not provided, the default buffer prototype is used.

        Returns
        -------
        NDArrayLike
            An array-like containing the data for the requested selection.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> data = np.arange(100).reshape(10, 10)
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=data.shape,
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve items by specifying a mask::

            >>> sel = np.zeros_like(z, dtype=bool)
            >>> sel[1, 1] = True
            >>> sel[4, 4] = True
            >>> z.get_mask_selection(sel)
            array([11, 44])

        For convenience, the mask selection functionality is also available via the
        `vindex` property, e.g.::

            >>> z.vindex[sel]
            array([11, 44])

        Notes
        -----
        Mask indexing is a form of vectorized or inner indexing, and is equivalent to
        coordinate indexing. Internally the mask array is converted to coordinate
        arrays by calling `np.nonzero`.

        See Also
        --------
        get_basic_selection, set_basic_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        set_coordinate_selection, get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__
        """

        indexer = MaskIndexer(mask, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._get_selection(
                indexer=indexer, out=out, fields=fields, prototype=prototype
            )
        )

    def set_mask_selection(
        self,
        mask: MaskSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> None:
        """Modify a selection of individual items, by providing a Boolean array of the
        same shape as the array against which the selection is being made, where True
        values indicate a selected item.

        Parameters
        ----------
        selection : ndarray, bool
            A Boolean array of the same shape as the array against which the selection is
            being made.
        value : npt.ArrayLike
            An array-like containing values to be stored into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(
            >>>        shape=(5, 5),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(5, 5),
            >>>        dtype="i4",
            >>>       )

        Set data for a selection of items::

            >>> sel = np.zeros_like(z, dtype=bool)
            >>> sel[1, 1] = True
            >>> sel[4, 4] = True
            >>> z.set_mask_selection(sel, 1)
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]])

        For convenience, this functionality is also available via the `vindex` property.
        E.g.::

            >>> z.vindex[sel] = 2
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 2, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 2]])

        Notes
        -----
        Mask indexing is a form of vectorized or inner indexing, and is equivalent to
        coordinate indexing. Internally the mask array is converted to coordinate
        arrays by calling `np.nonzero`.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        set_coordinate_selection, get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        indexer = MaskIndexer(mask, self.shape, self.metadata.chunk_grid)
        sync(self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype))

    def get_coordinate_selection(
        self,
        selection: CoordinateSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> NDArrayLike:
        """Retrieve a selection of individual items, by providing the indices
        (coordinates) for each selected item.

        Parameters
        ----------
        selection : tuple
            An integer (coordinate) array for each dimension of the array.
        out : NDBuffer, optional
            If given, load the selected data directly into this buffer.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer to use for the output data. If not provided, the default buffer prototype is used.

        Returns
        -------
        NDArrayLike
            An array-like containing the data for the requested coordinate selection.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> data = np.arange(0, 100, dtype="uint16").reshape((10, 10))
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=(3, 3),
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve items by specifying their coordinates::

            >>> z.get_coordinate_selection(([1, 4], [1, 4]))
            array([11, 44])

        For convenience, the coordinate selection functionality is also available via the
        `vindex` property, e.g.::

            >>> z.vindex[[1, 4], [1, 4]]
            array([11, 44])

        Notes
        -----
        Coordinate indexing is also known as point selection, and is a form of vectorized
        or inner indexing.

        Slices are not supported. Coordinate arrays must be provided for all dimensions
        of the array.

        Coordinate arrays may be multidimensional, in which case the output array will
        also be multidimensional. Coordinate arrays are broadcast against each other
        before being applied. The shape of the output will be the same as the shape of
        each coordinate array after broadcasting.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, set_coordinate_selection,
        get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        indexer = CoordinateIndexer(selection, self.shape, self.metadata.chunk_grid)
        out_array = sync(
            self._async_array._get_selection(
                indexer=indexer, out=out, fields=fields, prototype=prototype
            )
        )

        if hasattr(out_array, "shape"):
            # restore shape
            out_array = np.array(out_array).reshape(indexer.sel_shape)
        return out_array

    def set_coordinate_selection(
        self,
        selection: CoordinateSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> None:
        """Modify a selection of individual items, by providing the indices (coordinates)
        for each item to be modified.

        Parameters
        ----------
        selection : tuple
            An integer (coordinate) array for each dimension of the array.
        value : npt.ArrayLike
            An array-like containing values to be stored into the array.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(
            >>>        shape=(5, 5),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(5, 5),
            >>>        dtype="i4",
            >>>       )

        Set data for a selection of items::

            >>> z.set_coordinate_selection(([1, 4], [1, 4]), 1)
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]])

        For convenience, this functionality is also available via the `vindex` property.
        E.g.::

            >>> z.vindex[[1, 4], [1, 4]] = 2
            >>> z[...]
            array([[0, 0, 0, 0, 0],
                   [0, 2, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 2]])

        Notes
        -----
        Coordinate indexing is also known as point selection, and is a form of vectorized
        or inner indexing.

        Slices are not supported. Coordinate arrays must be provided for all dimensions
        of the array.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        # setup indexer
        indexer = CoordinateIndexer(selection, self.shape, self.metadata.chunk_grid)

        # handle value - need ndarray-like flatten value
        if not is_scalar(value, self.dtype):
            try:
                from numcodecs.compat import ensure_ndarray_like

                value = ensure_ndarray_like(value)  # TODO replace with agnostic
            except TypeError:
                # Handle types like `list` or `tuple`
                value = np.array(value)  # TODO replace with agnostic
        if hasattr(value, "shape") and len(value.shape) > 1:
            value = np.array(value).reshape(-1)

        sync(self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype))

    def get_block_selection(
        self,
        selection: BasicSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> NDArrayLike:
        """Retrieve a selection of individual items, by providing the indices
        (coordinates) for each selected item.

        Parameters
        ----------
        selection : int or slice or tuple of int or slice
            An integer (coordinate) or slice for each dimension of the array.
        out : NDBuffer, optional
            If given, load the selected data directly into this buffer.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer to use for the output data. If not provided, the default buffer prototype is used.

        Returns
        -------
        NDArrayLike
            An array-like containing the data for the requested block selection.

        Examples
        --------
        Setup a 2-dimensional array::

            >>> import zarr
            >>> import numpy as np
            >>> data = np.arange(0, 100, dtype="uint16").reshape((10, 10))
            >>> z = Array.create(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunk_shape=(3, 3),
            >>>        dtype=data.dtype,
            >>>        )
            >>> z[:] = data

        Retrieve items by specifying their block coordinates::

            >>> z.get_block_selection((1, slice(None)))
            array([[30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                   [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])

        Which is equivalent to::

            >>> z[3:6, :]
            array([[30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                   [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])

        For convenience, the block selection functionality is also available via the
        `blocks` property, e.g.::

            >>> z.blocks[1]
            array([[30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                   [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                   [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])

        Notes
        -----
        Block indexing is a convenience indexing method to work on individual chunks
        with chunk index slicing. It has the same concept as Dask's `Array.blocks`
        indexing.

        Slices are supported. However, only with a step size of one.

        Block index arrays may be multidimensional to index multidimensional arrays.
        For example::

            >>> z.blocks[0, 1:3]
            array([[ 3,  4,  5,  6,  7,  8],
                   [13, 14, 15, 16, 17, 18],
                   [23, 24, 25, 26, 27, 28]])

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        set_coordinate_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        indexer = BlockIndexer(selection, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._get_selection(
                indexer=indexer, out=out, fields=fields, prototype=prototype
            )
        )

    def set_block_selection(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype = default_buffer_prototype,
    ) -> None:
        """Modify a selection of individual blocks, by providing the chunk indices
        (coordinates) for each block to be modified.

        Parameters
        ----------
        selection : tuple
            An integer (coordinate) or slice for each dimension of the array.
        value : npt.ArrayLike
            An array-like containing the data to be stored in the block selection.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to set
            data for.
        prototype : BufferPrototype, optional
            The prototype of the buffer used for setting the data. If not provided, the
            default buffer prototype is used.

        Examples
        --------
        Set up a 2-dimensional array::

            >>> import zarr
            >>> z = zarr.zeros(
            >>>        shape=(6, 6),
            >>>        store=StorePath(MemoryStore(mode="w")),
            >>>        chunk_shape=(2, 2),
            >>>        dtype="i4",
            >>>       )

        Set data for a selection of items::

            >>> z.set_block_selection((1, 0), 1)
            >>> z[...]
            array([[0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]])

        For convenience, this functionality is also available via the `blocks` property.
        E.g.::

            >>> z.blocks[2, 1] = 4
            >>> z[...]
            array([[0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [0, 0, 4, 4, 0, 0],
                   [0, 0, 4, 4, 0, 0]])

            >>> z.blocks[:, 2] = 7
            >>> z[...]
            array([[0, 0, 0, 0, 7, 7],
                   [0, 0, 0, 0, 7, 7],
                   [1, 1, 0, 0, 7, 7],
                   [1, 1, 0, 0, 7, 7],
                   [0, 0, 4, 4, 7, 7],
                   [0, 0, 4, 4, 7, 7]])

        Notes
        -----
        Block indexing is a convenience indexing method to work on individual chunks
        with chunk index slicing. It has the same concept as Dask's `Array.blocks`
        indexing.

        Slices are supported. However, only with a step size of one.

        See Also
        --------
        get_basic_selection, set_basic_selection, get_mask_selection, set_mask_selection,
        get_orthogonal_selection, set_orthogonal_selection, get_coordinate_selection,
        get_block_selection, set_block_selection,
        vindex, oindex, blocks, __getitem__, __setitem__

        """
        indexer = BlockIndexer(selection, self.shape, self.metadata.chunk_grid)
        sync(self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype))

    @property
    def vindex(self) -> VIndex:
        """Shortcut for vectorized (inner) indexing, see :func:`get_coordinate_selection`,
        :func:`set_coordinate_selection`, :func:`get_mask_selection` and
        :func:`set_mask_selection` for documentation and examples."""
        return VIndex(self)

    @property
    def oindex(self) -> OIndex:
        """Shortcut for orthogonal (outer) indexing, see :func:`get_orthogonal_selection` and
        :func:`set_orthogonal_selection` for documentation and examples."""
        return OIndex(self)

    @property
    def blocks(self) -> BlockIndex:
        """Shortcut for blocked chunked indexing, see :func:`get_block_selection` and
        :func:`set_block_selection` for documentation and examples."""
        return BlockIndex(self)

    def resize(self, new_shape: ChunkCoords) -> Array:
        """
        Change the shape of the array by growing or shrinking one or more
        dimensions.

        This method does not modify the original Array object. Instead, it returns a new Array
        with the specified shape.

        Examples
        --------
        >>> import zarr
        >>> z = zarr.zeros(shape=(10000, 10000),
        >>>                chunk_shape=(1000, 1000),
        >>>                store=StorePath(MemoryStore(mode="w")),
        >>>                dtype="i4",)
        >>> z.shape
        (10000, 10000)
        >>> z = z.resize(20000, 1000)
        >>> z.shape
        (20000, 1000)
        >>> z2 = z.resize(50, 50)
        >>> z.shape
        (20000, 1000)
        >>> z2.shape
        (50, 50)

        Notes
        -----
        When resizing an array, the data are not rearranged in any way.

        If one or more dimensions are shrunk, any chunks falling outside the
        new array shape will be deleted from the underlying store.
        However, it is noteworthy that the chunks partially falling inside the new array
        (i.e. boundary chunks) will remain intact, and therefore,
        the data falling outside the new array but inside the boundary chunks
        would be restored by a subsequent resize operation that grows the array size.
        """
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

    def info(self) -> None:
        return sync(
            self._async_array.info(),
        )
