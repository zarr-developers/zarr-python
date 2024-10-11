from __future__ import annotations

import json
from asyncio import gather
from dataclasses import dataclass, field, replace
from logging import getLogger
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, overload

import numpy as np
import numpy.typing as npt

from zarr._compat import _deprecate_positional_args
from zarr.abc.store import Store, set_or_delete
from zarr.codecs import _get_default_array_bytes_codec
from zarr.codecs._v2 import V2Compressor, V2Filters
from zarr.core.attributes import Attributes
from zarr.core.buffer import (
    BufferPrototype,
    NDArrayLike,
    NDBuffer,
    default_buffer_prototype,
)
from zarr.core.chunk_grids import RegularChunkGrid, normalize_chunks
from zarr.core.chunk_key_encodings import (
    ChunkKeyEncoding,
    DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding,
)
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ChunkCoords,
    ShapeLike,
    ZarrFormat,
    concurrent_map,
    parse_dtype,
    parse_shapelike,
    product,
)
from zarr.core.config import config, parse_indexing_order
from zarr.core.indexing import (
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
    _iter_grid,
    ceildiv,
    check_fields,
    check_no_multi_fields,
    is_pure_fancy_indexing,
    is_pure_orthogonal_indexing,
    is_scalar,
    pop_fields,
)
from zarr.core.metadata import (
    ArrayMetadata,
    ArrayMetadataDict,
    ArrayV2Metadata,
    ArrayV2MetadataDict,
    ArrayV3Metadata,
    ArrayV3MetadataDict,
    T_ArrayMetadata,
)
from zarr.core.metadata.v3 import parse_node_type_array
from zarr.core.sync import collect_aiterator, sync
from zarr.errors import MetadataValidationError
from zarr.registry import get_pipeline_class
from zarr.storage import StoreLike, make_store_path
from zarr.storage.common import StorePath, ensure_no_existing_node

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import Self

    from zarr.abc.codec import Codec, CodecPipeline
    from zarr.core.group import AsyncGroup

# Array and AsyncArray are defined in the base ``zarr`` namespace
__all__ = ["create_codec_pipeline", "parse_array_metadata"]

logger = getLogger(__name__)


def parse_array_metadata(data: Any) -> ArrayMetadata:
    if isinstance(data, ArrayMetadata):
        return data
    elif isinstance(data, dict):
        if data["zarr_format"] == 3:
            meta_out = ArrayV3Metadata.from_dict(data)
            if len(meta_out.storage_transformers) > 0:
                msg = (
                    f"Array metadata contains storage transformers: {meta_out.storage_transformers}."
                    "Arrays with storage transformers are not supported in zarr-python at this time."
                )
                raise ValueError(msg)
            return meta_out
        elif data["zarr_format"] == 2:
            return ArrayV2Metadata.from_dict(data)
    raise TypeError


def create_codec_pipeline(metadata: ArrayMetadata) -> CodecPipeline:
    if isinstance(metadata, ArrayV3Metadata):
        return get_pipeline_class().from_codecs(metadata.codecs)
    elif isinstance(metadata, ArrayV2Metadata):
        return get_pipeline_class().from_codecs(
            [V2Filters(metadata.filters), V2Compressor(metadata.compressor)]
        )
    else:
        raise TypeError


async def get_array_metadata(
    store_path: StorePath, zarr_format: ZarrFormat | None = 3
) -> dict[str, JSON]:
    if zarr_format == 2:
        zarray_bytes, zattrs_bytes = await gather(
            (store_path / ZARRAY_JSON).get(), (store_path / ZATTRS_JSON).get()
        )
        if zarray_bytes is None:
            raise FileNotFoundError(store_path)
    elif zarr_format == 3:
        zarr_json_bytes = await (store_path / ZARR_JSON).get()
        if zarr_json_bytes is None:
            raise FileNotFoundError(store_path)
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
            raise FileNotFoundError(store_path)
        # set zarr_format based on which keys were found
        if zarr_json_bytes is not None:
            zarr_format = 3
        else:
            zarr_format = 2
    else:
        raise MetadataValidationError("zarr_format", "2, 3, or None", zarr_format)

    metadata_dict: dict[str, JSON]
    if zarr_format == 2:
        # V2 arrays are comprised of a .zarray and .zattrs objects
        assert zarray_bytes is not None
        metadata_dict = json.loads(zarray_bytes.to_bytes())
        zattrs_dict = json.loads(zattrs_bytes.to_bytes()) if zattrs_bytes is not None else {}
        metadata_dict["attributes"] = zattrs_dict
    else:
        # V3 arrays are comprised of a zarr.json object
        assert zarr_json_bytes is not None
        metadata_dict = json.loads(zarr_json_bytes.to_bytes())

        parse_node_type_array(metadata_dict.get("node_type"))

    return metadata_dict


@dataclass(frozen=True)
class AsyncArray(Generic[T_ArrayMetadata]):
    metadata: T_ArrayMetadata
    store_path: StorePath
    codec_pipeline: CodecPipeline = field(init=False)
    order: Literal["C", "F"]

    @overload
    def __init__(
        self: AsyncArray[ArrayV2Metadata],
        metadata: ArrayV2Metadata | ArrayV2MetadataDict,
        store_path: StorePath,
        order: Literal["C", "F"] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: AsyncArray[ArrayV3Metadata],
        metadata: ArrayV3Metadata | ArrayV3MetadataDict,
        store_path: StorePath,
        order: Literal["C", "F"] | None = None,
    ) -> None: ...

    def __init__(
        self,
        metadata: ArrayMetadata | ArrayMetadataDict,
        store_path: StorePath,
        order: Literal["C", "F"] | None = None,
    ) -> None:
        if isinstance(metadata, dict):
            zarr_format = metadata["zarr_format"]
            # TODO: remove this when we extensively type the dict representation of metadata
            _metadata = cast(dict[str, JSON], metadata)
            if zarr_format == 2:
                metadata = ArrayV2Metadata.from_dict(_metadata)
            elif zarr_format == 3:
                metadata = ArrayV3Metadata.from_dict(_metadata)
            else:
                raise ValueError(f"Invalid zarr_format: {zarr_format}. Expected 2 or 3")

        metadata_parsed = parse_array_metadata(metadata)
        order_parsed = parse_indexing_order(order or config.get("array.order"))

        object.__setattr__(self, "metadata", metadata_parsed)
        object.__setattr__(self, "store_path", store_path)
        object.__setattr__(self, "order", order_parsed)
        object.__setattr__(self, "codec_pipeline", create_codec_pipeline(metadata=metadata_parsed))

    # this overload defines the function signature when zarr_format is 2
    @overload
    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        # v2 and v3
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        zarr_format: Literal[2],
        fill_value: Any | None = None,
        attributes: dict[str, JSON] | None = None,
        chunks: ShapeLike | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> AsyncArray[ArrayV2Metadata]: ...

    # this overload defines the function signature when zarr_format is 3
    @overload
    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        # v2 and v3
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        zarr_format: Literal[3],
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
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> AsyncArray[ArrayV3Metadata]: ...

    # this overload is necessary to handle the case where the `zarr_format` kwarg is unspecified
    @overload
    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        # v2 and v3
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        zarr_format: Literal[3] = 3,
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
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> AsyncArray[ArrayV3Metadata]: ...

    @overload
    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        # v2 and v3
        shape: ShapeLike,
        dtype: npt.DTypeLike,
        zarr_format: ZarrFormat,
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
        chunks: ShapeLike | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]: ...

    @classmethod
    async def create(
        cls,
        store: StoreLike,
        *,
        # v2 and v3
        shape: ShapeLike,
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
        chunks: ShapeLike | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        exists_ok: bool = False,
        data: npt.ArrayLike | None = None,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        store_path = await make_store_path(store)

        dtype_parsed = parse_dtype(dtype, zarr_format)
        shape = parse_shapelike(shape)

        if chunks is not None and chunk_shape is not None:
            raise ValueError("Only one of chunk_shape or chunks can be provided.")

        if chunks:
            _chunks = normalize_chunks(chunks, shape, dtype_parsed.itemsize)
        else:
            _chunks = normalize_chunks(chunk_shape, shape, dtype_parsed.itemsize)

        result: AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]
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
            result = await cls._create_v3(
                store_path,
                shape=shape,
                dtype=dtype_parsed,
                chunk_shape=_chunks,
                fill_value=fill_value,
                chunk_key_encoding=chunk_key_encoding,
                codecs=codecs,
                dimension_names=dimension_names,
                attributes=attributes,
                exists_ok=exists_ok,
            )
        elif zarr_format == 2:
            if dtype is str or dtype == "str":
                # another special case: zarr v2 added the vlen-utf8 codec
                vlen_codec: dict[str, JSON] = {"id": "vlen-utf8"}
                if filters and not any(x["id"] == "vlen-utf8" for x in filters):
                    filters = list(filters) + [vlen_codec]
                else:
                    filters = [vlen_codec]

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
            result = await cls._create_v2(
                store_path,
                shape=shape,
                dtype=dtype_parsed,
                chunks=_chunks,
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

        if data is not None:
            # insert user-provided data
            await result.setitem(..., data)

        return result

    @classmethod
    async def _create_v3(
        cls,
        store_path: StorePath,
        *,
        shape: ShapeLike,
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
    ) -> AsyncArray[ArrayV3Metadata]:
        if not exists_ok:
            await ensure_no_existing_node(store_path, zarr_format=3)

        shape = parse_shapelike(shape)
        codecs = (
            list(codecs)
            if codecs is not None
            else [_get_default_array_bytes_codec(np.dtype(dtype))]
        )

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
        await array._save_metadata(metadata, ensure_parents=True)
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
        fill_value: None | float = None,
        order: Literal["C", "F"] | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        attributes: dict[str, JSON] | None = None,
        exists_ok: bool = False,
    ) -> AsyncArray[ArrayV2Metadata]:
        if not exists_ok:
            await ensure_no_existing_node(store_path, zarr_format=2)
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
            fill_value=fill_value,
            compressor=compressor,
            filters=filters,
            attributes=attributes,
        )
        array = cls(metadata=metadata, store_path=store_path)
        await array._save_metadata(metadata, ensure_parents=True)
        return array

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, JSON],
    ) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]:
        metadata = parse_array_metadata(data)
        return cls(metadata=metadata, store_path=store_path)

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        zarr_format: ZarrFormat | None = 3,
    ) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]:
        store_path = await make_store_path(store)
        metadata_dict = await get_array_metadata(store_path, zarr_format=zarr_format)
        # TODO: remove this cast when we have better type hints
        _metadata_dict = cast(ArrayV3MetadataDict, metadata_dict)
        return cls(store_path=store_path, metadata=_metadata_dict)

    @property
    def store(self) -> Store:
        return self.store_path.store

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

        msg = (
            f"The `chunks` attribute is only defined for arrays using `RegularChunkGrid`."
            f"This array has a {self.metadata.chunk_grid} instead."
        )
        raise NotImplementedError(msg)

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
        return self.store_path.store.mode.readonly

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

    @property
    def cdata_shape(self) -> ChunkCoords:
        """
        The shape of the chunk grid for this array.
        """
        return tuple(ceildiv(s, c) for s, c in zip(self.shape, self.chunks, strict=False))

    @property
    def nchunks(self) -> int:
        """
        The number of chunks in the stored representation of this array.
        """
        return product(self.cdata_shape)

    @property
    def nchunks_initialized(self) -> int:
        """
        The number of chunks that have been persisted in storage.
        """
        return nchunks_initialized(self)

    def _iter_chunk_coords(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[ChunkCoords]:
        """
        Create an iterator over the coordinates of chunks in chunk grid space. If the `origin`
        keyword is used, iteration will start at the chunk index specified by `origin`.
        The default behavior is to start at the origin of the grid coordinate space.
        If the `selection_shape` keyword is used, iteration will be bounded over a contiguous region
        ranging from `[origin, origin selection_shape]`, where the upper bound is exclusive as
        per python indexing conventions.

        Parameters
        ----------
        origin: Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape: Sequence[int] | None, default=None
            The shape of the selection in chunk grid coordinates.

        Yields
        ------
        chunk_coords: ChunkCoords
            The coordinates of each chunk in the selection.
        """
        return _iter_grid(self.cdata_shape, origin=origin, selection_shape=selection_shape)

    def _iter_chunk_keys(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[str]:
        """
        Iterate over the storage keys of each chunk, relative to an optional origin, and optionally
        limited to a contiguous region in chunk grid coordinates.

        Parameters
        ----------
        origin: Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape: Sequence[int] | None, default=None
            The shape of the selection in chunk grid coordinates.

        Yields
        ------
        key: str
            The storage key of each chunk in the selection.
        """
        # Iterate over the coordinates of chunks in chunk grid space.
        for k in self._iter_chunk_coords(origin=origin, selection_shape=selection_shape):
            # Encode the chunk key from the chunk coordinates.
            yield self.metadata.encode_chunk_key(k)

    def _iter_chunk_regions(
        self, *, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[slice, ...]]:
        """
        Iterate over the regions spanned by each chunk.

        Parameters
        ----------
        origin: Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape: Sequence[int] | None, default=None
            The shape of the selection in chunk grid coordinates.

        Yields
        ------
        region: tuple[slice, ...]
            A tuple of slice objects representing the region spanned by each chunk in the selection.
        """
        for cgrid_position in self._iter_chunk_coords(
            origin=origin, selection_shape=selection_shape
        ):
            out: tuple[slice, ...] = ()
            for c_pos, c_shape in zip(cgrid_position, self.chunks, strict=False):
                start = c_pos * c_shape
                stop = start + c_shape
                out += (slice(start, stop, 1),)
            yield out

    @property
    def nbytes(self) -> int:
        """
        The number of bytes that can be stored in this array.
        """
        return self.nchunks * self.dtype.itemsize

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
        self,
        selection: BasicSelection,
        *,
        prototype: BufferPrototype | None = None,
    ) -> NDArrayLike:
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
        )
        return await self._get_selection(indexer, prototype=prototype)

    async def _save_metadata(self, metadata: ArrayMetadata, ensure_parents: bool = False) -> None:
        to_save = metadata.to_buffer_dict(default_buffer_prototype())
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]

        if ensure_parents:
            # To enable zarr.create(store, path="a/b/c"), we need to create all the intermediate groups.
            parents = _build_parents(self)

            for parent in parents:
                awaitables.extend(
                    [
                        (parent.store_path / key).set_if_not_exists(value)
                        for key, value in parent.metadata.to_buffer_dict(
                            default_buffer_prototype()
                        ).items()
                    ]
                )

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
            array_like = prototype.buffer.create_zero_length().as_array_like()
            if isinstance(array_like, np._typing._SupportsArrayFunc):
                # TODO: need to handle array types that don't support __array_function__
                # like PyTorch and JAX
                array_like_ = cast(np._typing._SupportsArrayFunc, array_like)
            value = np.asanyarray(value, dtype=self.metadata.dtype, like=array_like_)
        else:
            if not hasattr(value, "shape"):
                value = np.asarray(value, self.metadata.dtype)
            # assert (
            #     value.shape == indexer.shape
            # ), f"shape of value doesn't match indexer shape. Expected {indexer.shape}, got {value.shape}"
            if not hasattr(value, "dtype") or value.dtype.name != self.metadata.dtype.name:
                if hasattr(value, "astype"):
                    # Handle things that are already NDArrayLike more efficiently
                    value = value.astype(dtype=self.metadata.dtype, order="A")
                else:
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
        prototype: BufferPrototype | None = None,
    ) -> None:
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
        )
        return await self._set_selection(indexer, value, prototype=prototype)

    async def resize(self, new_shape: ChunkCoords, delete_outside_chunks: bool = True) -> Self:
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

    async def update_attributes(self, new_attributes: dict[str, JSON]) -> Self:
        # metadata.attributes is "frozen" so we simply clear and update the dict
        self.metadata.attributes.clear()
        self.metadata.attributes.update(new_attributes)

        # Write new metadata
        await self._save_metadata(self.metadata)

        return self

    def __repr__(self) -> str:
        return f"<AsyncArray {self.store_path} shape={self.shape} dtype={self.dtype}>"

    async def info(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class Array:
    _async_array: AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]

    @classmethod
    @_deprecate_positional_args
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
    def store(self) -> Store:
        return self._async_array.store

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

    @property
    def fill_value(self) -> Any:
        return self.metadata.fill_value

    @property
    def cdata_shape(self) -> ChunkCoords:
        """
        The shape of the chunk grid for this array.
        """
        return tuple(ceildiv(s, c) for s, c in zip(self.shape, self.chunks, strict=False))

    @property
    def nchunks(self) -> int:
        """
        The number of chunks in the stored representation of this array.
        """
        return self._async_array.nchunks

    def _iter_chunk_coords(
        self, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[ChunkCoords]:
        """
        Create an iterator over the coordinates of chunks in chunk grid space. If the `origin`
        keyword is used, iteration will start at the chunk index specified by `origin`.
        The default behavior is to start at the origin of the grid coordinate space.
        If the `selection_shape` keyword is used, iteration will be bounded over a contiguous region
        ranging from `[origin, origin + selection_shape]`, where the upper bound is exclusive as
        per python indexing conventions.

        Parameters
        ----------
        origin: Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape: Sequence[int] | None, default=None
            The shape of the selection in chunk grid coordinates.

        Yields
        ------
        chunk_coords: ChunkCoords
            The coordinates of each chunk in the selection.
        """
        yield from self._async_array._iter_chunk_coords(
            origin=origin, selection_shape=selection_shape
        )

    @property
    def nbytes(self) -> int:
        """
        The number of bytes that can be stored in this array.
        """
        return self._async_array.nbytes

    @property
    def nchunks_initialized(self) -> int:
        """
        The number of chunks that have been initialized in the stored representation of this array.
        """
        return self._async_array.nchunks_initialized

    def _iter_chunk_keys(
        self, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[str]:
        """
        Iterate over the storage keys of each chunk, relative to an optional origin, and optionally
        limited to a contiguous region in chunk grid coordinates.

        Parameters
        ----------
        origin: Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape: Sequence[int] | None, default=None
            The shape of the selection in chunk grid coordinates.

        Yields
        ------
        key: str
            The storage key of each chunk in the selection.
        """
        yield from self._async_array._iter_chunk_keys(
            origin=origin, selection_shape=selection_shape
        )

    def _iter_chunk_regions(
        self, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[tuple[slice, ...]]:
        """
        Iterate over the regions spanned by each chunk.

        Parameters
        ----------
        origin: Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape: Sequence[int] | None, default=None
            The shape of the selection in chunk grid coordinates.

        Yields
        ------
        region: tuple[slice, ...]
            A tuple of slice objects representing the region spanned by each chunk in the selection.
        """
        yield from self._async_array._iter_chunk_regions(
            origin=origin, selection_shape=selection_shape
        )

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

    @_deprecate_positional_args
    def get_basic_selection(
        self,
        selection: BasicSelection = Ellipsis,
        *,
        out: NDBuffer | None = None,
        prototype: BufferPrototype | None = None,
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
        prototype : BufferPrototype, optional
            The prototype of the buffer to use for the output data. If not provided, the default buffer prototype is used.
        fields : str or sequence of str, optional
            For arrays with a structured dtype, one or more fields can be specified to
            extract data for.

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

        if prototype is None:
            prototype = default_buffer_prototype()
        return sync(
            self._async_array._get_selection(
                BasicIndexer(selection, self.shape, self.metadata.chunk_grid),
                out=out,
                fields=fields,
                prototype=prototype,
            )
        )

    @_deprecate_positional_args
    def set_basic_selection(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BasicIndexer(selection, self.shape, self.metadata.chunk_grid)
        sync(self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype))

    @_deprecate_positional_args
    def get_orthogonal_selection(
        self,
        selection: OrthogonalSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = OrthogonalIndexer(selection, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._get_selection(
                indexer=indexer, out=out, fields=fields, prototype=prototype
            )
        )

    @_deprecate_positional_args
    def set_orthogonal_selection(
        self,
        selection: OrthogonalSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = OrthogonalIndexer(selection, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype)
        )

    @_deprecate_positional_args
    def get_mask_selection(
        self,
        mask: MaskSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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

        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = MaskIndexer(mask, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._get_selection(
                indexer=indexer, out=out, fields=fields, prototype=prototype
            )
        )

    @_deprecate_positional_args
    def set_mask_selection(
        self,
        mask: MaskSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = MaskIndexer(mask, self.shape, self.metadata.chunk_grid)
        sync(self._async_array._set_selection(indexer, value, fields=fields, prototype=prototype))

    @_deprecate_positional_args
    def get_coordinate_selection(
        self,
        selection: CoordinateSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
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

    @_deprecate_positional_args
    def set_coordinate_selection(
        self,
        selection: CoordinateSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
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

    @_deprecate_positional_args
    def get_block_selection(
        self,
        selection: BasicSelection,
        *,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BlockIndexer(selection, self.shape, self.metadata.chunk_grid)
        return sync(
            self._async_array._get_selection(
                indexer=indexer, out=out, fields=fields, prototype=prototype
            )
        )

    @_deprecate_positional_args
    def set_block_selection(
        self,
        selection: BasicSelection,
        value: npt.ArrayLike,
        *,
        fields: Fields | None = None,
        prototype: BufferPrototype | None = None,
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
        if prototype is None:
            prototype = default_buffer_prototype()
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

        Notes
        -----
        When resizing an array, the data are not rearranged in any way.

        If one or more dimensions are shrunk, any chunks falling outside the
        new array shape will be deleted from the underlying store.
        However, it is noteworthy that the chunks partially falling inside the new array
        (i.e. boundary chunks) will remain intact, and therefore,
        the data falling outside the new array but inside the boundary chunks
        would be restored by a subsequent resize operation that grows the array size.

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
        """
        resized = sync(self._async_array.resize(new_shape))
        # TODO: remove this cast when type inference improves
        _resized = cast(AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata], resized)
        return type(self)(_resized)

    def update_attributes(self, new_attributes: dict[str, JSON]) -> Array:
        # TODO: remove this cast when type inference improves
        new_array = sync(self._async_array.update_attributes(new_attributes))
        # TODO: remove this cast when type inference improves
        _new_array = cast(AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata], new_array)
        return type(self)(_new_array)

    def __repr__(self) -> str:
        return f"<Array {self.store_path} shape={self.shape} dtype={self.dtype}>"

    def info(self) -> None:
        return sync(
            self._async_array.info(),
        )


def nchunks_initialized(
    array: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | Array,
) -> int:
    """
    Calculate the number of chunks that have been initialized, i.e. the number of chunks that have
    been persisted to the storage backend.

    Parameters
    ----------
    array : Array
        The array to inspect.

    Returns
    -------
    nchunks_initialized : int
        The number of chunks that have been initialized.

    See Also
    --------
    chunks_initialized
    """
    return len(chunks_initialized(array))


def chunks_initialized(
    array: Array | AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata],
) -> tuple[str, ...]:
    """
    Return the keys of the chunks that have been persisted to the storage backend.

    Parameters
    ----------
    array : Array
        The array to inspect.

    Returns
    -------
    chunks_initialized : tuple[str, ...]
        The keys of the chunks that have been initialized.

    See Also
    --------
    nchunks_initialized

    """
    # TODO: make this compose with the underlying async iterator
    store_contents = list(
        collect_aiterator(array.store_path.store.list_prefix(prefix=array.store_path.path))
    )
    out: list[str] = []

    for chunk_key in array._iter_chunk_keys():
        if chunk_key in store_contents:
            out.append(chunk_key)

    return tuple(out)


def _build_parents(
    node: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata] | AsyncGroup,
) -> list[AsyncGroup]:
    from zarr.core.group import AsyncGroup, GroupMetadata

    store = node.store_path.store
    path = node.store_path.path
    if not path:
        return []

    required_parts = path.split("/")[:-1]
    parents = [
        # the root group
        AsyncGroup(
            metadata=GroupMetadata(zarr_format=node.metadata.zarr_format),
            store_path=StorePath(store=store, path=""),
        )
    ]

    for i, part in enumerate(required_parts):
        p = "/".join(required_parts[:i] + [part])
        parents.append(
            AsyncGroup(
                metadata=GroupMetadata(zarr_format=node.metadata.zarr_format),
                store_path=StorePath(store=store, path=p),
            )
        )

    return parents
