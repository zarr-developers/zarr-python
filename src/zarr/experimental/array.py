from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.abc.numcodec import Numcodec
from zarr.core._info import ArrayInfo
from zarr.core.array import (
    _append,
    _get_coordinate_selection,
    _get_mask_selection,
    _get_orthogonal_selection,
    _getitem,
    _info_complete,
    _iter_chunk_coords,
    _iter_chunk_regions,
    _iter_shard_coords,
    _iter_shard_keys,
    _iter_shard_regions,
    _nbytes_stored,
    _nchunks_initialized,
    _nshards_initialized,
    _resize,
    _setitem,
    _update_attributes,
    create_codec_pipeline,
    get_array_metadata,
    parse_array_metadata,
)
from zarr.core.array_spec import ArrayConfig, ArrayConfigLike, parse_array_config
from zarr.core.buffer import (
    BufferPrototype,
    NDArrayLikeOrScalar,
    NDBuffer,
)
from zarr.core.common import (
    JSON,
    MemoryOrder,
    ShapeLike,
    ZarrFormat,
    ceildiv,
    product,
)
from zarr.core.indexing import (
    BasicSelection,
    CoordinateSelection,
    Fields,
    MaskSelection,
    OrthogonalSelection,
)
from zarr.core.metadata import (
    ArrayMetadata,
    ArrayMetadataDict,
    ArrayV2Metadata,
    ArrayV3Metadata,
)
from zarr.core.sync import sync
from zarr.storage._common import StorePath, make_store_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Self

    import numpy.typing as npt

    from zarr.abc.codec import CodecPipeline
    from zarr.abc.store import Store
    from zarr.storage import StoreLike


class Array:
    """
    A unified Zarr array class with both synchronous and asynchronous methods.

    This class combines the functionality of AsyncArray and Array into a single class.
    For each operation, there is both a synchronous method (e.g., `getitem`) and an
    asynchronous method (e.g., `getitem_async`).

    Parameters
    ----------
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The metadata of the array.
    store_path : StorePath
        The path to the Zarr store.
    config : ArrayConfigLike, optional
        The runtime configuration of the array, by default None.

    Attributes
    ----------
    metadata : ArrayV2Metadata | ArrayV3Metadata
        The metadata of the array.
    store_path : StorePath
        The path to the Zarr store.
    codec_pipeline : CodecPipeline
        The codec pipeline used for encoding and decoding chunks.
    _config : ArrayConfig
        The runtime configuration of the array.
    """

    metadata: ArrayV2Metadata | ArrayV3Metadata
    store_path: StorePath
    codec_pipeline: CodecPipeline
    config: ArrayConfig

    def __init__(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata | ArrayMetadataDict,
        *,
        codec_pipeline: CodecPipeline | None = None,
        config: ArrayConfigLike | None = None,
    ) -> None:
        metadata_parsed = parse_array_metadata(metadata)
        config_parsed = parse_array_config(config)

        if codec_pipeline is None:
            codec_pipeline = create_codec_pipeline(metadata=metadata_parsed, store=store_path.store)

        self.metadata = metadata_parsed
        self.store_path = store_path
        self.config = config_parsed
        self.codec_pipeline = codec_pipeline

    # -------------------------------------------------------------------------
    # Class methods: open
    # -------------------------------------------------------------------------

    @classmethod
    async def open_async(
        cls,
        store: StoreLike,
        *,
        config: ArrayConfigLike | None = None,
        codec_pipeline: CodecPipeline | None = None,
        zarr_format: ZarrFormat | None = 3,
    ) -> Array:
        """
        Async method to open an existing Zarr array from a given store.

        Parameters
        ----------
        store : StoreLike
            The store containing the Zarr array.
        zarr_format : ZarrFormat | None, optional
            The Zarr format version (default is 3).

        Returns
        -------
        Array
            The opened Zarr array.
        """
        store_path = await make_store_path(store)
        metadata_dict = await get_array_metadata(store_path, zarr_format=zarr_format)
        return cls(
            store_path=store_path,
            metadata=metadata_dict,
            codec_pipeline=codec_pipeline,
            config=config,
        )

    @classmethod
    def open(
        cls,
        store: StoreLike,
        *,
        config: ArrayConfigLike | None = None,
        codec_pipeline: CodecPipeline | None = None,
        zarr_format: ZarrFormat | None = 3,
    ) -> Array:
        """
        Open an existing Zarr array from a given store.

        Parameters
        ----------
        store : StoreLike
            The store containing the Zarr array.
        zarr_format : ZarrFormat | None, optional
            The Zarr format version (default is 3).

        Returns
        -------
        Array
            The opened Zarr array.
        """
        return sync(cls.open_async(store, zarr_format=zarr_format))

    # -------------------------------------------------------------------------
    # Properties (all synchronous, derived from metadata/store_path)
    # -------------------------------------------------------------------------

    @property
    def store(self) -> Store:
        """The store containing the array data."""
        return self.store_path.store

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions in the Array."""
        return len(self.metadata.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the Array."""
        return self.metadata.shape

    @property
    def chunks(self) -> tuple[int, ...]:
        """Returns the chunk shape of the Array."""
        return self.metadata.chunks

    @property
    def shards(self) -> tuple[int, ...] | None:
        """Returns the shard shape of the Array, or None if sharding is not used."""
        return self.metadata.shards

    @property
    def size(self) -> int:
        """Returns the total number of elements in the array."""
        return np.prod(self.metadata.shape).item()

    @property
    def filters(self) -> tuple[Numcodec, ...] | tuple[ArrayArrayCodec, ...]:
        """Filters applied to each chunk before serialization."""
        if self.metadata.zarr_format == 2:
            filters = self.metadata.filters
            if filters is None:
                return ()
            return filters
        return tuple(
            codec for codec in self.metadata.inner_codecs if isinstance(codec, ArrayArrayCodec)
        )

    @property
    def serializer(self) -> ArrayBytesCodec | None:
        """Array-to-bytes codec for serializing chunks."""
        if self.metadata.zarr_format == 2:
            return None
        return next(
            codec for codec in self.metadata.inner_codecs if isinstance(codec, ArrayBytesCodec)
        )

    @property
    def compressors(self) -> tuple[Numcodec, ...] | tuple[BytesBytesCodec, ...]:
        """Compressors applied to each chunk after serialization."""
        if self.metadata.zarr_format == 2:
            if self.metadata.compressor is not None:
                return (self.metadata.compressor,)
            return ()
        return tuple(
            codec for codec in self.metadata.inner_codecs if isinstance(codec, BytesBytesCodec)
        )

    @property
    def _zdtype(self) -> Any:
        """The zarr-specific representation of the array data type."""
        if self.metadata.zarr_format == 2:
            return self.metadata.dtype
        else:
            return self.metadata.data_type

    @property
    def dtype(self) -> np.dtype[Any]:
        """Returns the data type of the array."""
        return self._zdtype.to_native_dtype()

    @property
    def order(self) -> MemoryOrder:
        """Returns the memory order of the array."""
        if self.metadata.zarr_format == 2:
            return self.metadata.order
        else:
            return self.config.order

    @property
    def attrs(self) -> dict[str, JSON]:
        """Returns the attributes of the array."""
        return self.metadata.attributes

    @property
    def read_only(self) -> bool:
        """Returns True if the array is read-only."""
        return self.store_path.read_only

    @property
    def path(self) -> str:
        """Storage path."""
        return self.store_path.path

    @property
    def name(self) -> str:
        """Array name following h5py convention."""
        name = self.path
        if not name.startswith("/"):
            name = "/" + name
        return name

    @property
    def basename(self) -> str:
        """Final component of name."""
        return self.name.split("/")[-1]

    @property
    def cdata_shape(self) -> tuple[int, ...]:
        """The shape of the chunk grid for this array."""
        return self._chunk_grid_shape

    @property
    def _chunk_grid_shape(self) -> tuple[int, ...]:
        """The shape of the chunk grid for this array."""
        return tuple(starmap(ceildiv, zip(self.shape, self.chunks, strict=True)))

    @property
    def _shard_grid_shape(self) -> tuple[int, ...]:
        """The shape of the shard grid for this array."""
        if self.shards is None:
            shard_shape = self.chunks
        else:
            shard_shape = self.shards
        return tuple(starmap(ceildiv, zip(self.shape, shard_shape, strict=True)))

    @property
    def nchunks(self) -> int:
        """The number of chunks in this array."""
        return product(self._chunk_grid_shape)

    @property
    def _nshards(self) -> int:
        """The number of shards in this array."""
        return product(self._shard_grid_shape)

    @property
    def nbytes(self) -> int:
        """The total number of bytes that would be stored if all chunks were initialized."""
        return self.size * self.dtype.itemsize

    @property
    def info(self) -> ArrayInfo:
        """Return the statically known information for an array."""
        return self._info()

    def _info(
        self, count_chunks_initialized: int | None = None, count_bytes_stored: int | None = None
    ) -> ArrayInfo:
        return ArrayInfo(
            _zarr_format=self.metadata.zarr_format,
            _data_type=self._zdtype,
            _fill_value=self.metadata.fill_value,
            _shape=self.shape,
            _order=self.order,
            _shard_shape=self.shards,
            _chunk_shape=self.chunks,
            _read_only=self.read_only,
            _compressors=self.compressors,
            _filters=self.filters,
            _serializer=self.serializer,
            _store_type=type(self.store_path.store).__name__,
            _count_bytes=self.nbytes,
            _count_bytes_stored=count_bytes_stored,
            _count_chunks_initialized=count_chunks_initialized,
        )

    # -------------------------------------------------------------------------
    # Iteration methods (synchronous)
    # -------------------------------------------------------------------------

    def _iter_chunk_coords(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[int, ...]]:
        """Iterate over chunk coordinates in chunk grid space."""
        return _iter_chunk_coords(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_shard_coords(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[int, ...]]:
        """Iterate over shard coordinates in shard grid space."""
        return _iter_shard_coords(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_shard_keys(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[str]:
        """Iterate over the keys of stored objects supporting this array."""
        return _iter_shard_keys(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_chunk_regions(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[slice, ...]]:
        """Iterate over chunk regions in array index space."""
        return _iter_chunk_regions(array=self, origin=origin, selection_shape=selection_shape)

    def _iter_shard_regions(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[slice, ...]]:
        """Iterate over shard regions in array index space."""
        return _iter_shard_regions(array=self, origin=origin, selection_shape=selection_shape)

    # -------------------------------------------------------------------------
    # nchunks_initialized: async and sync
    # -------------------------------------------------------------------------

    async def nchunks_initialized_async(self) -> int:
        """
        Asynchronously calculate the number of chunks that have been initialized.

        Returns
        -------
        int
            The number of chunks that have been initialized.
        """
        return await _nchunks_initialized(self)

    def nchunks_initialized(self) -> int:
        """
        Calculate the number of chunks that have been initialized.

        Returns
        -------
        int
            The number of chunks that have been initialized.
        """
        return sync(self.nchunks_initialized_async())

    # -------------------------------------------------------------------------
    # _nshards_initialized: async and sync
    # -------------------------------------------------------------------------

    async def _nshards_initialized_async(self) -> int:
        """
        Asynchronously calculate the number of shards that have been initialized.

        Returns
        -------
        int
            The number of shards that have been initialized.
        """
        return await _nshards_initialized(self)

    def _nshards_initialized(self) -> int:
        """
        Calculate the number of shards that have been initialized.

        Returns
        -------
        int
            The number of shards that have been initialized.
        """
        return sync(self._nshards_initialized_async())

    # -------------------------------------------------------------------------
    # nbytes_stored: async and sync
    # -------------------------------------------------------------------------

    async def nbytes_stored_async(self) -> int:
        """
        Asynchronously calculate the number of bytes stored for this array.

        Returns
        -------
        int
            The number of bytes stored.
        """
        return await _nbytes_stored(self.store_path)

    def nbytes_stored(self) -> int:
        """
        Calculate the number of bytes stored for this array.

        Returns
        -------
        int
            The number of bytes stored.
        """
        return sync(self.nbytes_stored_async())

    # -------------------------------------------------------------------------
    # getitem: async and sync
    # -------------------------------------------------------------------------

    async def getitem_async(
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously retrieve a subset of the array's data based on the provided selection.

        Parameters
        ----------
        selection : BasicSelection
            A selection object specifying the subset of data to retrieve.
        prototype : BufferPrototype, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The retrieved subset of the array's data.
        """
        return await _getitem(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            prototype=prototype,
        )

    def getitem(
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Retrieve a subset of the array's data based on the provided selection.

        Parameters
        ----------
        selection : BasicSelection
            A selection object specifying the subset of data to retrieve.
        prototype : BufferPrototype, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The retrieved subset of the array's data.
        """
        return sync(self.getitem_async(selection, prototype=prototype))

    def __getitem__(self, selection: BasicSelection) -> NDArrayLikeOrScalar:
        """Retrieve data using indexing syntax."""
        return self.getitem(selection)

    # -------------------------------------------------------------------------
    # setitem: async and sync
    # -------------------------------------------------------------------------

    async def setitem_async(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        prototype: BufferPrototype | None = None,
    ) -> None:
        """
        Asynchronously set values in the array using basic indexing.

        Parameters
        ----------
        selection : BasicSelection
            The selection defining the region of the array to set.
        value : npt.ArrayLike
            The values to be written into the selected region.
        prototype : BufferPrototype, optional
            A buffer prototype to use.
        """
        return await _setitem(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            value,
            prototype=prototype,
        )

    def setitem(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        prototype: BufferPrototype | None = None,
    ) -> None:
        """
        Set values in the array using basic indexing.

        Parameters
        ----------
        selection : BasicSelection
            The selection defining the region of the array to set.
        value : npt.ArrayLike
            The values to be written into the selected region.
        prototype : BufferPrototype, optional
            A buffer prototype to use.
        """
        sync(self.setitem_async(selection, value, prototype=prototype))

    def __setitem__(self, selection: BasicSelection, value: npt.ArrayLike) -> None:
        """Set data using indexing syntax."""
        self.setitem(selection, value)

    # -------------------------------------------------------------------------
    # get_orthogonal_selection: async and sync
    # -------------------------------------------------------------------------

    async def get_orthogonal_selection_async(
        self,
        selection: OrthogonalSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously get an orthogonal selection from the array.

        Parameters
        ----------
        selection : OrthogonalSelection
            The orthogonal selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return await _get_orthogonal_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            out=out,
            fields=fields,
            prototype=prototype,
        )

    def get_orthogonal_selection(
        self,
        selection: OrthogonalSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Get an orthogonal selection from the array.

        Parameters
        ----------
        selection : OrthogonalSelection
            The orthogonal selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return sync(
            self.get_orthogonal_selection_async(
                selection, out=out, fields=fields, prototype=prototype
            )
        )

    # -------------------------------------------------------------------------
    # get_mask_selection: async and sync
    # -------------------------------------------------------------------------

    async def get_mask_selection_async(
        self,
        mask: MaskSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously get a mask selection from the array.

        Parameters
        ----------
        mask : MaskSelection
            The boolean mask specifying the selection.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return await _get_mask_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            mask,
            out=out,
            fields=fields,
            prototype=prototype,
        )

    def get_mask_selection(
        self,
        mask: MaskSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Get a mask selection from the array.

        Parameters
        ----------
        mask : MaskSelection
            The boolean mask specifying the selection.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return sync(
            self.get_mask_selection_async(mask, out=out, fields=fields, prototype=prototype)
        )

    # -------------------------------------------------------------------------
    # get_coordinate_selection: async and sync
    # -------------------------------------------------------------------------

    async def get_coordinate_selection_async(
        self,
        selection: CoordinateSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Asynchronously get a coordinate selection from the array.

        Parameters
        ----------
        selection : CoordinateSelection
            The coordinate selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return await _get_coordinate_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            selection,
            out=out,
            fields=fields,
            prototype=prototype,
        )

    def get_coordinate_selection(
        self,
        selection: CoordinateSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLikeOrScalar:
        """
        Get a coordinate selection from the array.

        Parameters
        ----------
        selection : CoordinateSelection
            The coordinate selection specification.
        out : NDBuffer | None, optional
            An output buffer to write the data to.
        fields : Fields | None, optional
            Fields to select from structured arrays.
        prototype : BufferPrototype | None, optional
            A buffer prototype to use for the retrieved data.

        Returns
        -------
        NDArrayLikeOrScalar
            The selected data.
        """
        return sync(
            self.get_coordinate_selection_async(
                selection, out=out, fields=fields, prototype=prototype
            )
        )

    # -------------------------------------------------------------------------
    # resize: async and sync
    # -------------------------------------------------------------------------

    async def resize_async(self, new_shape: ShapeLike, delete_outside_chunks: bool = True) -> None:
        """
        Asynchronously resize the array to a new shape.

        Parameters
        ----------
        new_shape : ShapeLike
            The desired new shape of the array.
        delete_outside_chunks : bool, optional
            If True (default), chunks that fall outside the new shape will be deleted.
        """
        return await _resize(self, new_shape, delete_outside_chunks)

    def resize(self, new_shape: ShapeLike, delete_outside_chunks: bool = True) -> None:
        """
        Resize the array to a new shape.

        Parameters
        ----------
        new_shape : ShapeLike
            The desired new shape of the array.
        delete_outside_chunks : bool, optional
            If True (default), chunks that fall outside the new shape will be deleted.
        """
        sync(self.resize_async(new_shape, delete_outside_chunks))

    # -------------------------------------------------------------------------
    # append: async and sync
    # -------------------------------------------------------------------------

    async def append_async(self, data: npt.ArrayLike, axis: int = 0) -> tuple[int, ...]:
        """
        Asynchronously append data to the array along the specified axis.

        Parameters
        ----------
        data : npt.ArrayLike
            Data to be appended.
        axis : int
            Axis along which to append.

        Returns
        -------
        tuple[int, ...]
            The new shape of the array after appending.
        """
        return await _append(self, data, axis)

    def append(self, data: npt.ArrayLike, axis: int = 0) -> tuple[int, ...]:
        """
        Append data to the array along the specified axis.

        Parameters
        ----------
        data : npt.ArrayLike
            Data to be appended.
        axis : int
            Axis along which to append.

        Returns
        -------
        tuple[int, ...]
            The new shape of the array after appending.
        """
        return sync(self.append_async(data, axis))

    # -------------------------------------------------------------------------
    # update_attributes: async and sync
    # -------------------------------------------------------------------------

    async def update_attributes_async(self, new_attributes: dict[str, JSON]) -> Self:
        """
        Asynchronously update the array's attributes.

        Parameters
        ----------
        new_attributes : dict[str, JSON]
            A dictionary of new attributes to update or add.

        Returns
        -------
        Array
            The array with the updated attributes.
        """
        await _update_attributes(self, new_attributes)
        return self

    def update_attributes(self, new_attributes: dict[str, JSON]) -> Self:
        """
        Update the array's attributes.

        Parameters
        ----------
        new_attributes : dict[str, JSON]
            A dictionary of new attributes to update or add.

        Returns
        -------
        Array
            The array with the updated attributes.
        """
        return sync(self.update_attributes_async(new_attributes))

    # -------------------------------------------------------------------------
    # info_complete: async and sync
    # -------------------------------------------------------------------------

    async def info_complete_async(self) -> ArrayInfo:
        """
        Asynchronously return all the information for an array, including dynamic information.

        Returns
        -------
        ArrayInfo
            Complete information about the array including chunks initialized and bytes stored.
        """
        return await _info_complete(self)

    def info_complete(self) -> ArrayInfo:
        """
        Return all the information for an array, including dynamic information.

        Returns
        -------
        ArrayInfo
            Complete information about the array including chunks initialized and bytes stored.
        """
        return sync(self.info_complete_async())

    # -------------------------------------------------------------------------
    # __repr__
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Array {self.store_path} shape={self.shape} dtype={self.dtype}>"
