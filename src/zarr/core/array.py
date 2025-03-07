from __future__ import annotations

import json
import warnings
from asyncio import gather
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import starmap
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypedDict,
    cast,
    overload,
)
from warnings import warn

import numcodecs
import numcodecs.abc
import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated

from zarr._compat import _deprecate_positional_args
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec, Codec
from zarr.abc.store import Store, set_or_delete
from zarr.codecs._v2 import V2Codec
from zarr.core._info import ArrayInfo
from zarr.core.array_spec import ArrayConfig, ArrayConfigLike, parse_array_config
from zarr.core.attributes import Attributes
from zarr.core.buffer import (
    BufferPrototype,
    NDArrayLike,
    NDBuffer,
    default_buffer_prototype,
)
from zarr.core.buffer.cpu import buffer_prototype as cpu_buffer_prototype
from zarr.core.chunk_grids import RegularChunkGrid, _auto_partition, normalize_chunks
from zarr.core.chunk_key_encodings import (
    ChunkKeyEncoding,
    ChunkKeyEncodingLike,
    DefaultChunkKeyEncoding,
    V2ChunkKeyEncoding,
)
from zarr.core.common import (
    JSON,
    ZARR_JSON,
    ZARRAY_JSON,
    ZATTRS_JSON,
    ChunkCoords,
    MemoryOrder,
    ShapeLike,
    ZarrFormat,
    _default_zarr_format,
    _warn_order_kwarg,
    concurrent_map,
    parse_dtype,
    parse_order,
    parse_shapelike,
    product,
)
from zarr.core.config import config as zarr_config
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
from zarr.core.metadata.v2 import (
    _default_compressor,
    _default_filters,
    parse_compressor,
    parse_filters,
)
from zarr.core.metadata.v3 import DataType, parse_node_type_array
from zarr.core.sync import sync
from zarr.errors import MetadataValidationError
from zarr.registry import (
    _parse_array_array_codec,
    _parse_array_bytes_codec,
    _parse_bytes_bytes_codec,
    get_pipeline_class,
)
from zarr.storage._common import StorePath, ensure_no_existing_node, make_store_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Self

    from zarr.abc.codec import CodecPipeline
    from zarr.codecs.sharding import ShardingCodecIndexLocation
    from zarr.core.group import AsyncGroup
    from zarr.storage import StoreLike


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
        v2_codec = V2Codec(filters=metadata.filters, compressor=metadata.compressor)
        return get_pipeline_class().from_codecs([v2_codec])
    else:
        raise TypeError


async def get_array_metadata(
    store_path: StorePath, zarr_format: ZarrFormat | None = 3
) -> dict[str, JSON]:
    if zarr_format == 2:
        zarray_bytes, zattrs_bytes = await gather(
            (store_path / ZARRAY_JSON).get(prototype=cpu_buffer_prototype),
            (store_path / ZATTRS_JSON).get(prototype=cpu_buffer_prototype),
        )
        if zarray_bytes is None:
            raise FileNotFoundError(store_path)
    elif zarr_format == 3:
        zarr_json_bytes = await (store_path / ZARR_JSON).get(prototype=cpu_buffer_prototype)
        if zarr_json_bytes is None:
            raise FileNotFoundError(store_path)
    elif zarr_format is None:
        zarr_json_bytes, zarray_bytes, zattrs_bytes = await gather(
            (store_path / ZARR_JSON).get(prototype=cpu_buffer_prototype),
            (store_path / ZARRAY_JSON).get(prototype=cpu_buffer_prototype),
            (store_path / ZATTRS_JSON).get(prototype=cpu_buffer_prototype),
        )
        if zarr_json_bytes is not None and zarray_bytes is not None:
            # warn and favor v3
            msg = f"Both zarr.json (Zarr format 3) and .zarray (Zarr format 2) metadata objects exist at {store_path}. Zarr v3 will be used."
            warnings.warn(msg, stacklevel=1)
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
    """
    An asynchronous array class representing a chunked array stored in a Zarr store.

    Parameters
    ----------
    metadata : ArrayMetadata
        The metadata of the array.
    store_path : StorePath
        The path to the Zarr store.
    config : ArrayConfigLike, optional
        The runtime configuration of the array, by default None.

    Attributes
    ----------
    metadata : ArrayMetadata
        The metadata of the array.
    store_path : StorePath
        The path to the Zarr store.
    codec_pipeline : CodecPipeline
        The codec pipeline used for encoding and decoding chunks.
    _config : ArrayConfig
        The runtime configuration of the array.
    """

    metadata: T_ArrayMetadata
    store_path: StorePath
    codec_pipeline: CodecPipeline = field(init=False)
    _config: ArrayConfig

    @overload
    def __init__(
        self: AsyncArray[ArrayV2Metadata],
        metadata: ArrayV2Metadata | ArrayV2MetadataDict,
        store_path: StorePath,
        config: ArrayConfigLike | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: AsyncArray[ArrayV3Metadata],
        metadata: ArrayV3Metadata | ArrayV3MetadataDict,
        store_path: StorePath,
        config: ArrayConfigLike | None = None,
    ) -> None: ...

    def __init__(
        self,
        metadata: ArrayMetadata | ArrayMetadataDict,
        store_path: StorePath,
        config: ArrayConfigLike | None = None,
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
        config_parsed = parse_array_config(config)

        object.__setattr__(self, "metadata", metadata_parsed)
        object.__setattr__(self, "store_path", store_path)
        object.__setattr__(self, "_config", config_parsed)
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
        order: MemoryOrder | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        overwrite: bool = False,
        data: npt.ArrayLike | None = None,
        config: ArrayConfigLike | None = None,
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
        chunk_shape: ShapeLike | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # runtime
        overwrite: bool = False,
        data: npt.ArrayLike | None = None,
        config: ArrayConfigLike | None = None,
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
        zarr_format: Literal[3] = 3,
        fill_value: Any | None = None,
        attributes: dict[str, JSON] | None = None,
        # v3 only
        chunk_shape: ShapeLike | None = None,
        chunk_key_encoding: (
            ChunkKeyEncoding
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # runtime
        overwrite: bool = False,
        data: npt.ArrayLike | None = None,
        config: ArrayConfigLike | None = None,
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
        chunk_shape: ShapeLike | None = None,
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
        order: MemoryOrder | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        overwrite: bool = False,
        data: npt.ArrayLike | None = None,
        config: ArrayConfigLike | None = None,
    ) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]: ...

    @classmethod
    @deprecated("Use zarr.api.asynchronous.create_array instead.")
    @_deprecate_positional_args
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
        chunk_shape: ShapeLike | None = None,
        chunk_key_encoding: (
            ChunkKeyEncodingLike
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # v2 only
        chunks: ShapeLike | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: MemoryOrder | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        overwrite: bool = False,
        data: npt.ArrayLike | None = None,
        config: ArrayConfigLike | None = None,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Method to create a new asynchronous array instance.

        .. deprecated:: 3.0.0
            Deprecated in favor of :func:`zarr.api.asynchronous.create_array`.

        Parameters
        ----------
        store : StoreLike
            The store where the array will be created.
        shape : ShapeLike
            The shape of the array.
        dtype : npt.DTypeLike
            The data type of the array.
        zarr_format : ZarrFormat, optional
            The Zarr format version (default is 3).
        fill_value : Any, optional
            The fill value of the array (default is None).
        attributes : dict[str, JSON], optional
            The attributes of the array (default is None).
        chunk_shape : ChunkCoords, optional
            The shape of the array's chunks
            Zarr format 3 only. Zarr format 2 arrays should use `chunks` instead.
            If not specified, default are guessed based on the shape and dtype.
        chunk_key_encoding : ChunkKeyEncodingLike, optional
            A specification of how the chunk keys are represented in storage.
            Zarr format 3 only. Zarr format 2 arrays should use `dimension_separator` instead.
            Default is ``("default", "/")``.
        codecs : Sequence of Codecs or dicts, optional
            An iterable of Codec or dict serializations of Codecs. The elements of
            this collection specify the transformation from array values to stored bytes.
            Zarr format 3 only. Zarr format 2 arrays should use ``filters`` and ``compressor`` instead.

            If no codecs are provided, default codecs will be used:

            - For numeric arrays, the default is ``BytesCodec`` and ``ZstdCodec``.
            - For Unicode strings, the default is ``VLenUTF8Codec`` and ``ZstdCodec``.
            - For bytes or objects, the default is ``VLenBytesCodec`` and ``ZstdCodec``.

            These defaults can be changed by modifying the value of ``array.v3_default_filters``,
            ``array.v3_default_serializer`` and ``array.v3_default_compressors`` in :mod:`zarr.core.config`.
        dimension_names : Iterable[str], optional
            The names of the dimensions (default is None).
            Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
        chunks : ShapeLike, optional
            The shape of the array's chunks.
            Zarr format 2 only. Zarr format 3 arrays should use ``chunk_shape`` instead.
            If not specified, default are guessed based on the shape and dtype.
        dimension_separator : Literal[".", "/"], optional
            The dimension separator (default is ".").
            Zarr format 2 only. Zarr format 3 arrays should use ``chunk_key_encoding`` instead.
        order : Literal["C", "F"], optional
            The memory of the array (default is "C").
            If ``zarr_format`` is 2, this parameter sets the memory order of the array.
            If `zarr_format`` is 3, then this parameter is deprecated, because memory order
            is a runtime parameter for Zarr 3 arrays. The recommended way to specify the memory
            order for Zarr 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
        filters : list[dict[str, JSON]], optional
            Sequence of filters to use to encode chunk data prior to compression.
            Zarr format 2 only. Zarr format 3 arrays should use ``codecs`` instead. If no ``filters``
            are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v2_default_filters`` in :mod:`zarr.core.config`.
        compressor : dict[str, JSON], optional
            The compressor used to compress the data (default is None).
            Zarr format 2 only. Zarr format 3 arrays should use ``codecs`` instead.

            If no ``compressor`` is provided, a default compressor will be used:

            - For numeric arrays, the default is ``ZstdCodec``.
            - For Unicode strings, the default is ``VLenUTF8Codec``.
            - For bytes or objects, the default is ``VLenBytesCodec``.

            These defaults can be changed by modifying the value of ``array.v2_default_compressor`` in :mod:`zarr.core.config`.
        overwrite : bool, optional
            Whether to raise an error if the store already exists (default is False).
        data : npt.ArrayLike, optional
            The data to be inserted into the array (default is None).
        config : ArrayConfigLike, optional
            Runtime configuration for the array.

        Returns
        -------
        AsyncArray
            The created asynchronous array instance.
        """
        return await cls._create(
            store,
            # v2 and v3
            shape=shape,
            dtype=dtype,
            zarr_format=zarr_format,
            fill_value=fill_value,
            attributes=attributes,
            # v3 only
            chunk_shape=chunk_shape,
            chunk_key_encoding=chunk_key_encoding,
            codecs=codecs,
            dimension_names=dimension_names,
            # v2 only
            chunks=chunks,
            dimension_separator=dimension_separator,
            order=order,
            filters=filters,
            compressor=compressor,
            # runtime
            overwrite=overwrite,
            data=data,
            config=config,
        )

    @classmethod
    async def _create(
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
        chunk_shape: ShapeLike | None = None,
        chunk_key_encoding: (
            ChunkKeyEncodingLike
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        # v2 only
        chunks: ShapeLike | None = None,
        dimension_separator: Literal[".", "/"] | None = None,
        order: MemoryOrder | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        overwrite: bool = False,
        data: npt.ArrayLike | None = None,
        config: ArrayConfigLike | None = None,
    ) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
        """Method to create a new asynchronous array instance.
        See :func:`AsyncArray.create` for more details.
        Deprecated in favor of :func:`zarr.api.asynchronous.create_array`.
        """
        store_path = await make_store_path(store)

        dtype_parsed = parse_dtype(dtype, zarr_format)
        shape = parse_shapelike(shape)

        if chunks is not None and chunk_shape is not None:
            raise ValueError("Only one of chunk_shape or chunks can be provided.")

        if chunks:
            _chunks = normalize_chunks(chunks, shape, dtype_parsed.itemsize)
        else:
            _chunks = normalize_chunks(chunk_shape, shape, dtype_parsed.itemsize)
        config_parsed = parse_array_config(config)

        result: AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]
        if zarr_format == 3:
            if dimension_separator is not None:
                raise ValueError(
                    "dimension_separator cannot be used for arrays with zarr_format 3. Use chunk_key_encoding instead."
                )
            if filters is not None:
                raise ValueError(
                    "filters cannot be used for arrays with zarr_format 3. Use array-to-array codecs instead."
                )
            if compressor is not None:
                raise ValueError(
                    "compressor cannot be used for arrays with zarr_format 3. Use bytes-to-bytes codecs instead."
                )

            if order is not None:
                _warn_order_kwarg()

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
                overwrite=overwrite,
                config=config_parsed,
            )
        elif zarr_format == 2:
            if codecs is not None:
                raise ValueError(
                    "codecs cannot be used for arrays with zarr_format 2. Use filters and compressor instead."
                )
            if chunk_key_encoding is not None:
                raise ValueError(
                    "chunk_key_encoding cannot be used for arrays with zarr_format 2. Use dimension_separator instead."
                )
            if dimension_names is not None:
                raise ValueError("dimension_names cannot be used for arrays with zarr_format 2.")

            if order is None:
                order_parsed = parse_order(zarr_config.get("array.order"))
            else:
                order_parsed = order

            result = await cls._create_v2(
                store_path,
                shape=shape,
                dtype=dtype_parsed,
                chunks=_chunks,
                dimension_separator=dimension_separator,
                fill_value=fill_value,
                order=order_parsed,
                config=config_parsed,
                filters=filters,
                compressor=compressor,
                attributes=attributes,
                overwrite=overwrite,
            )
        else:
            raise ValueError(f"Insupported zarr_format. Got: {zarr_format}")

        if data is not None:
            # insert user-provided data
            await result.setitem(..., data)

        return result

    @staticmethod
    def _create_metadata_v3(
        shape: ShapeLike,
        dtype: np.dtype[Any],
        chunk_shape: ChunkCoords,
        fill_value: Any | None = None,
        chunk_key_encoding: ChunkKeyEncodingLike | None = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        attributes: dict[str, JSON] | None = None,
    ) -> ArrayV3Metadata:
        """
        Create an instance of ArrayV3Metadata.
        """

        shape = parse_shapelike(shape)
        codecs = list(codecs) if codecs is not None else _get_default_codecs(np.dtype(dtype))
        chunk_key_encoding_parsed: ChunkKeyEncodingLike
        if chunk_key_encoding is None:
            chunk_key_encoding_parsed = {"name": "default", "separator": "/"}
        else:
            chunk_key_encoding_parsed = chunk_key_encoding

        if dtype.kind in "UTS":
            warn(
                f"The dtype `{dtype}` is currently not part in the Zarr format 3 specification. It "
                "may not be supported by other zarr implementations and may change in the future.",
                category=UserWarning,
                stacklevel=2,
            )
        chunk_grid_parsed = RegularChunkGrid(chunk_shape=chunk_shape)
        return ArrayV3Metadata(
            shape=shape,
            data_type=dtype,
            chunk_grid=chunk_grid_parsed,
            chunk_key_encoding=chunk_key_encoding_parsed,
            fill_value=fill_value,
            codecs=codecs,
            dimension_names=tuple(dimension_names) if dimension_names else None,
            attributes=attributes or {},
        )

    @classmethod
    async def _create_v3(
        cls,
        store_path: StorePath,
        *,
        shape: ShapeLike,
        dtype: np.dtype[Any],
        chunk_shape: ChunkCoords,
        config: ArrayConfig,
        fill_value: Any | None = None,
        chunk_key_encoding: (
            ChunkKeyEncodingLike
            | tuple[Literal["default"], Literal[".", "/"]]
            | tuple[Literal["v2"], Literal[".", "/"]]
            | None
        ) = None,
        codecs: Iterable[Codec | dict[str, JSON]] | None = None,
        dimension_names: Iterable[str] | None = None,
        attributes: dict[str, JSON] | None = None,
        overwrite: bool = False,
    ) -> AsyncArray[ArrayV3Metadata]:
        if overwrite:
            if store_path.store.supports_deletes:
                await store_path.delete_dir()
            else:
                await ensure_no_existing_node(store_path, zarr_format=3)
        else:
            await ensure_no_existing_node(store_path, zarr_format=3)

        if isinstance(chunk_key_encoding, tuple):
            chunk_key_encoding = (
                V2ChunkKeyEncoding(separator=chunk_key_encoding[1])
                if chunk_key_encoding[0] == "v2"
                else DefaultChunkKeyEncoding(separator=chunk_key_encoding[1])
            )

        metadata = cls._create_metadata_v3(
            shape=shape,
            dtype=dtype,
            chunk_shape=chunk_shape,
            fill_value=fill_value,
            chunk_key_encoding=chunk_key_encoding,
            codecs=codecs,
            dimension_names=dimension_names,
            attributes=attributes,
        )

        array = cls(metadata=metadata, store_path=store_path, config=config)
        await array._save_metadata(metadata, ensure_parents=True)
        return array

    @staticmethod
    def _create_metadata_v2(
        shape: ChunkCoords,
        dtype: np.dtype[Any],
        chunks: ChunkCoords,
        order: MemoryOrder,
        dimension_separator: Literal[".", "/"] | None = None,
        fill_value: float | None = None,
        filters: Iterable[dict[str, JSON] | numcodecs.abc.Codec] | None = None,
        compressor: dict[str, JSON] | numcodecs.abc.Codec | None = None,
        attributes: dict[str, JSON] | None = None,
    ) -> ArrayV2Metadata:
        if dimension_separator is None:
            dimension_separator = "."

        dtype = parse_dtype(dtype, zarr_format=2)

        # inject VLenUTF8 for str dtype if not already present
        if np.issubdtype(dtype, np.str_):
            filters = filters or []
            from numcodecs.vlen import VLenUTF8

            if not any(isinstance(x, VLenUTF8) or x["id"] == "vlen-utf8" for x in filters):
                filters = list(filters) + [VLenUTF8()]

        return ArrayV2Metadata(
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

    @classmethod
    async def _create_v2(
        cls,
        store_path: StorePath,
        *,
        shape: ChunkCoords,
        dtype: np.dtype[Any],
        chunks: ChunkCoords,
        order: MemoryOrder,
        config: ArrayConfig,
        dimension_separator: Literal[".", "/"] | None = None,
        fill_value: float | None = None,
        filters: Iterable[dict[str, JSON] | numcodecs.abc.Codec] | None = None,
        compressor: dict[str, JSON] | numcodecs.abc.Codec | None = None,
        attributes: dict[str, JSON] | None = None,
        overwrite: bool = False,
    ) -> AsyncArray[ArrayV2Metadata]:
        if overwrite:
            if store_path.store.supports_deletes:
                await store_path.delete_dir()
            else:
                await ensure_no_existing_node(store_path, zarr_format=2)
        else:
            await ensure_no_existing_node(store_path, zarr_format=2)

        metadata = cls._create_metadata_v2(
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            order=order,
            dimension_separator=dimension_separator,
            fill_value=fill_value,
            filters=filters,
            compressor=compressor,
            attributes=attributes,
        )

        array = cls(metadata=metadata, store_path=store_path, config=config)
        await array._save_metadata(metadata, ensure_parents=True)
        return array

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, JSON],
    ) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]:
        """
        Create a Zarr array from a dictionary, with support for both Zarr format 2 and 3 metadata.

        Parameters
        ----------
        store_path : StorePath
            The path within the store where the array should be created.

        data : dict
            A dictionary representing the array data. This dictionary should include necessary metadata
            for the array, such as shape, dtype, and other attributes. The format of the metadata
            will determine whether a Zarr format 2 or 3 array is created.

        Returns
        -------
        AsyncArray[ArrayV3Metadata] or AsyncArray[ArrayV2Metadata]
            The created Zarr array, either using Zarr format 2 or 3 metadata based on the provided data.

        Raises
        ------
        ValueError
            If the dictionary data is invalid or incompatible with either Zarr format 2 or 3 array creation.
        """
        metadata = parse_array_metadata(data)
        return cls(metadata=metadata, store_path=store_path)

    @classmethod
    async def open(
        cls,
        store: StoreLike,
        zarr_format: ZarrFormat | None = 3,
    ) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]:
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
        AsyncArray
            The opened Zarr array.

        Examples
        --------
        >>> import zarr
        >>>  store = zarr.storage.MemoryStore(mode='w')
        >>>  async_arr = await AsyncArray.open(store) # doctest: +ELLIPSIS
        <AsyncArray memory://... shape=(100, 100) dtype=int32>
        """
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
        """Returns the number of dimensions in the Array.

        Returns
        -------
        int
            The number of dimensions in the Array.
        """
        return len(self.metadata.shape)

    @property
    def shape(self) -> ChunkCoords:
        """Returns the shape of the Array.

        Returns
        -------
        tuple
            The shape of the Array.
        """
        return self.metadata.shape

    @property
    def chunks(self) -> ChunkCoords:
        """Returns the chunk shape of the Array.
        If sharding is used the inner chunk shape is returned.

        Only defined for arrays using using `RegularChunkGrid`.
        If array doesn't use `RegularChunkGrid`, `NotImplementedError` is raised.

        Returns
        -------
        ChunkCoords:
            The chunk shape of the Array.
        """
        return self.metadata.chunks

    @property
    def shards(self) -> ChunkCoords | None:
        """Returns the shard shape of the Array.
        Returns None if sharding is not used.

        Only defined for arrays using using `RegularChunkGrid`.
        If array doesn't use `RegularChunkGrid`, `NotImplementedError` is raised.

        Returns
        -------
        ChunkCoords:
            The shard shape of the Array.
        """
        return self.metadata.shards

    @property
    def size(self) -> int:
        """Returns the total number of elements in the array

        Returns
        -------
        int
            Total number of elements in the array
        """
        return np.prod(self.metadata.shape).item()

    @property
    def filters(self) -> tuple[numcodecs.abc.Codec, ...] | tuple[ArrayArrayCodec, ...]:
        """
        Filters that are applied to each chunk of the array, in order, before serializing that
        chunk to bytes.
        """
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
        """
        Array-to-bytes codec to use for serializing the chunks into bytes.
        """
        if self.metadata.zarr_format == 2:
            return None

        return next(
            codec for codec in self.metadata.inner_codecs if isinstance(codec, ArrayBytesCodec)
        )

    @property
    @deprecated("Use AsyncArray.compressors instead.")
    def compressor(self) -> numcodecs.abc.Codec | None:
        """
        Compressor that is applied to each chunk of the array.

        .. deprecated:: 3.0.0
            `array.compressor` is deprecated and will be removed in a future release.
            Use `array.compressors` instead.
        """
        if self.metadata.zarr_format == 2:
            return self.metadata.compressor
        raise TypeError("`compressor` is not available for Zarr format 3 arrays.")

    @property
    def compressors(self) -> tuple[numcodecs.abc.Codec, ...] | tuple[BytesBytesCodec, ...]:
        """
        Compressors that are applied to each chunk of the array. Compressors are applied in order, and after any
        filters are applied (if any are specified) and the data is serialized into bytes.
        """
        if self.metadata.zarr_format == 2:
            if self.metadata.compressor is not None:
                return (self.metadata.compressor,)
            return ()

        return tuple(
            codec for codec in self.metadata.inner_codecs if isinstance(codec, BytesBytesCodec)
        )

    @property
    def dtype(self) -> np.dtype[Any]:
        """Returns the data type of the array.

        Returns
        -------
        np.dtype
            Data type of the array
        """
        return self.metadata.dtype

    @property
    def order(self) -> MemoryOrder:
        """Returns the memory order of the array.

        Returns
        -------
        bool
            Memory order of the array
        """
        return self._config.order

    @property
    def attrs(self) -> dict[str, JSON]:
        """Returns the attributes of the array.

        Returns
        -------
        dict
            Attributes of the array
        """
        return self.metadata.attributes

    @property
    def read_only(self) -> bool:
        """Returns True if the array is read-only.

        Returns
        -------
        bool
            True if the array is read-only
        """
        # Backwards compatibility for 2.x
        return self.store_path.read_only

    @property
    def path(self) -> str:
        """Storage path.

        Returns
        -------
        str
            The path to the array in the Zarr store.
        """
        return self.store_path.path

    @property
    def name(self) -> str:
        """Array name following h5py convention.

        Returns
        -------
        str
            The name of the array.
        """
        # follow h5py convention: add leading slash
        name = self.path
        if not name.startswith("/"):
            name = "/" + name
        return name

    @property
    def basename(self) -> str:
        """Final component of name.

        Returns
        -------
        str
            The basename or final component of the array name.
        """
        return self.name.split("/")[-1]

    @property
    def cdata_shape(self) -> ChunkCoords:
        """
        The shape of the chunk grid for this array.

        Returns
        -------
        Tuple[int]
            The shape of the chunk grid for this array.
        """
        return tuple(starmap(ceildiv, zip(self.shape, self.chunks, strict=False)))

    @property
    def nchunks(self) -> int:
        """
        The number of chunks in the stored representation of this array.

        Returns
        -------
        int
            The total number of chunks in the array.
        """
        return product(self.cdata_shape)

    async def nchunks_initialized(self) -> int:
        """
        Calculate the number of chunks that have been initialized, i.e. the number of chunks that have
        been persisted to the storage backend.

        Returns
        -------
        nchunks_initialized : int
            The number of chunks that have been initialized.

        Notes
        -----
        On :class:`AsyncArray` this is an asynchronous method, unlike the (synchronous)
        property :attr:`Array.nchunks_initialized`.

        Examples
        --------
        >>> arr = await zarr.api.asynchronous.create(shape=(10,), chunks=(2,))
        >>> await arr.nchunks_initialized()
        0
        >>> await arr.setitem(slice(5), 1)
        >>> await arr.nchunks_initialized()
        3
        """
        return len(await chunks_initialized(self))

    async def nbytes_stored(self) -> int:
        return await self.store_path.store.getsize_prefix(self.store_path.path)

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
        origin : Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape : Sequence[int] | None, default=None
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
        origin : Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape : Sequence[int] | None, default=None
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
        origin : Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape : Sequence[int] | None, default=None
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
        The total number of bytes that can be stored in the chunks of this array.

        Notes
        -----
        This value is calculated by multiplying the number of elements in the array and the size
        of each element, the latter of which is determined by the dtype of the array.
        For this reason, ``nbytes`` will likely be inaccurate for arrays with variable-length
        dtypes. It is not possible to determine the size of an array with variable-length elements
        from the shape and dtype alone.
        """
        return self.size * self.dtype.itemsize

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
                order=self._config.order,
                fill_value=self.metadata.fill_value,
            )
        if product(indexer.shape) > 0:
            # need to use the order from the metadata for v2
            _config = self._config
            if self.metadata.zarr_format == 2:
                _config = replace(_config, order=self.metadata.order)

            # reading chunks and decoding them
            await self.codec_pipeline.read(
                [
                    (
                        self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                        self.metadata.get_chunk_spec(chunk_coords, _config, prototype=prototype),
                        chunk_selection,
                        out_selection,
                        is_complete_chunk,
                    )
                    for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
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
        """
        Asynchronous function that retrieves a subset of the array's data based on the provided selection.

        Parameters
        ----------
        selection : BasicSelection
            A selection object specifying the subset of data to retrieve.
        prototype : BufferPrototype, optional
            A buffer prototype to use for the retrieved data (default is None).

        Returns
        -------
        NDArrayLike
            The retrieved subset of the array's data.

        Examples
        --------
        >>> import zarr
        >>>  store = zarr.storage.MemoryStore(mode='w')
        >>>  async_arr = await zarr.api.asynchronous.create_array(
        ...      store=store,
        ...      shape=(100,100),
        ...      chunks=(10,10),
        ...      dtype='i4',
        ...      fill_value=0)
        <AsyncArray memory://... shape=(100, 100) dtype=int32>
        >>> await async_arr.getitem((0,1)) # doctest: +ELLIPSIS
        array(0, dtype=int32)

        """
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
        )
        return await self._get_selection(indexer, prototype=prototype)

    async def _save_metadata(self, metadata: ArrayMetadata, ensure_parents: bool = False) -> None:
        """
        Asynchronously save the array metadata.
        """
        to_save = metadata.to_buffer_dict(cpu_buffer_prototype)
        awaitables = [set_or_delete(self.store_path / key, value) for key, value in to_save.items()]

        if ensure_parents:
            # To enable zarr.create(store, path="a/b/c"), we need to create all the intermediate groups.
            parents = _build_parents(self)

            for parent in parents:
                awaitables.extend(
                    [
                        (parent.store_path / key).set_if_not_exists(value)
                        for key, value in parent.metadata.to_buffer_dict(
                            cpu_buffer_prototype
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

        # need to use the order from the metadata for v2
        _config = self._config
        if self.metadata.zarr_format == 2:
            _config = replace(_config, order=self.metadata.order)

        # merging with existing data and encoding chunks
        await self.codec_pipeline.write(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    self.metadata.get_chunk_spec(chunk_coords, _config, prototype),
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
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
        """
        Asynchronously set values in the array using basic indexing.

        Parameters
        ----------
        selection : BasicSelection
            The selection defining the region of the array to set.

        value : numpy.typing.ArrayLike
            The values to be written into the selected region of the array.

        prototype : BufferPrototype or None, optional
            A prototype buffer that defines the structure and properties of the array chunks being modified.
            If None, the default buffer prototype is used. Default is None.

        Returns
        -------
        None
            This method does not return any value.

        Raises
        ------
        IndexError
            If the selection is out of bounds for the array.

        ValueError
            If the values are not compatible with the array's dtype or shape.

        Notes
        -----
        - This method is asynchronous and should be awaited.
        - Supports basic indexing, where the selection is contiguous and does not involve advanced indexing.
        """
        if prototype is None:
            prototype = default_buffer_prototype()
        indexer = BasicIndexer(
            selection,
            shape=self.metadata.shape,
            chunk_grid=self.metadata.chunk_grid,
        )
        return await self._set_selection(indexer, value, prototype=prototype)

    async def resize(self, new_shape: ShapeLike, delete_outside_chunks: bool = True) -> None:
        """
        Asynchronously resize the array to a new shape.

        Parameters
        ----------
        new_shape : ChunkCoords
            The desired new shape of the array.

        delete_outside_chunks : bool, optional
            If True (default), chunks that fall outside the new shape will be deleted. If False,
            the data in those chunks will be preserved.

        Returns
        -------
        AsyncArray
            The resized array.

        Raises
        ------
        ValueError
            If the new shape is incompatible with the current array's chunking configuration.

        Notes
        -----
        - This method is asynchronous and should be awaited.
        """
        new_shape = parse_shapelike(new_shape)
        assert len(new_shape) == len(self.metadata.shape)
        new_metadata = self.metadata.update_shape(new_shape)

        if delete_outside_chunks:
            # Remove all chunks outside of the new shape
            old_chunk_coords = set(self.metadata.chunk_grid.all_chunk_coords(self.metadata.shape))
            new_chunk_coords = set(self.metadata.chunk_grid.all_chunk_coords(new_shape))

            async def _delete_key(key: str) -> None:
                await (self.store_path / key).delete()

            await concurrent_map(
                [
                    (self.metadata.encode_chunk_key(chunk_coords),)
                    for chunk_coords in old_chunk_coords.difference(new_chunk_coords)
                ],
                _delete_key,
                zarr_config.get("async.concurrency"),
            )

        # Write new metadata
        await self._save_metadata(new_metadata)

        # Update metadata (in place)
        object.__setattr__(self, "metadata", new_metadata)

    async def append(self, data: npt.ArrayLike, axis: int = 0) -> ChunkCoords:
        """Append `data` to `axis`.

        Parameters
        ----------
        data : array-like
            Data to be appended.
        axis : int
            Axis along which to append.

        Returns
        -------
        new_shape : tuple

        Notes
        -----
        The size of all dimensions other than `axis` must match between this
        array and `data`.
        """
        # ensure data is array-like
        if not hasattr(data, "shape"):
            data = np.asanyarray(data)

        self_shape_preserved = tuple(s for i, s in enumerate(self.shape) if i != axis)
        data_shape_preserved = tuple(s for i, s in enumerate(data.shape) if i != axis)
        if self_shape_preserved != data_shape_preserved:
            raise ValueError(
                f"shape of data to append is not compatible with the array. "
                f"The shape of the data is ({data_shape_preserved})"
                f"and the shape of the array is ({self_shape_preserved})."
                "All dimensions must match except for the dimension being "
                "appended."
            )
        # remember old shape
        old_shape = self.shape

        # determine new shape
        new_shape = tuple(
            self.shape[i] if i != axis else self.shape[i] + data.shape[i]
            for i in range(len(self.shape))
        )

        # resize
        await self.resize(new_shape)

        # store data
        append_selection = tuple(
            slice(None) if i != axis else slice(old_shape[i], new_shape[i])
            for i in range(len(self.shape))
        )
        await self.setitem(append_selection, data)

        return new_shape

    async def update_attributes(self, new_attributes: dict[str, JSON]) -> Self:
        """
        Asynchronously update the array's attributes.

        Parameters
        ----------
        new_attributes : dict of str to JSON
            A dictionary of new attributes to update or add to the array. The keys represent attribute
            names, and the values must be JSON-compatible.

        Returns
        -------
        AsyncArray
            The array with the updated attributes.

        Raises
        ------
        ValueError
            If the attributes are invalid or incompatible with the array's metadata.

        Notes
        -----
        - This method is asynchronous and should be awaited.
        - The updated attributes will be merged with existing attributes, and any conflicts will be
          overwritten by the new values.
        """
        self.metadata.attributes.update(new_attributes)

        # Write new metadata
        await self._save_metadata(self.metadata)

        return self

    def __repr__(self) -> str:
        return f"<AsyncArray {self.store_path} shape={self.shape} dtype={self.dtype}>"

    @property
    def info(self) -> Any:
        """
        Return the statically known information for an array.

        Returns
        -------
        ArrayInfo

        See Also
        --------
        AsyncArray.info_complete
            All information about a group, including dynamic information
            like the number of bytes and chunks written.

        Examples
        --------

        >>> arr = await zarr.api.asynchronous.create(
        ...     path="array", shape=(3, 4, 5), chunks=(2, 2, 2))
        ... )
        >>> arr.info
        Type               : Array
        Zarr format        : 3
        Data type          : DataType.float64
        Shape              : (3, 4, 5)
        Chunk shape        : (2, 2, 2)
        Order              : C
        Read-only          : False
        Store type         : MemoryStore
        Codecs             : [{'endian': <Endian.little: 'little'>}]
        No. bytes          : 480
        """
        return self._info()

    async def info_complete(self) -> Any:
        """
        Return all the information for an array, including dynamic information like a storage size.

        In addition to the static information, this provides

        - The count of chunks initialized
        - The sum of the bytes written

        Returns
        -------
        ArrayInfo

        See Also
        --------
        AsyncArray.info
            A property giving just the statically known information about an array.
        """
        return self._info(
            await self.nchunks_initialized(),
            await self.store_path.store.getsize_prefix(self.store_path.path),
        )

    def _info(
        self, count_chunks_initialized: int | None = None, count_bytes_stored: int | None = None
    ) -> Any:
        _data_type: np.dtype[Any] | DataType
        if isinstance(self.metadata, ArrayV2Metadata):
            _data_type = self.metadata.dtype
        else:
            _data_type = self.metadata.data_type

        return ArrayInfo(
            _zarr_format=self.metadata.zarr_format,
            _data_type=_data_type,
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


# TODO: Array can be a frozen data class again once property setters (e.g. shape) are removed
@dataclass(frozen=False)
class Array:
    """Instantiate an array from an initialized store."""

    _async_array: AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]

    @classmethod
    @deprecated("Use zarr.create_array instead.")
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
        order: MemoryOrder | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        overwrite: bool = False,
        config: ArrayConfigLike | None = None,
    ) -> Array:
        """Creates a new Array instance from an initialized store.

        .. deprecated:: 3.0.0
            Deprecated in favor of :func:`zarr.create_array`.

        Parameters
        ----------
        store : StoreLike
            The array store that has already been initialized.
        shape : ChunkCoords
            The shape of the array.
        dtype : npt.DTypeLike
            The data type of the array.
        chunk_shape : ChunkCoords, optional
            The shape of the Array's chunks.
            Zarr format 3 only. Zarr format 2 arrays should use `chunks` instead.
            If not specified, default are guessed based on the shape and dtype.
        chunk_key_encoding : ChunkKeyEncodingLike, optional
            A specification of how the chunk keys are represented in storage.
            Zarr format 3 only. Zarr format 2 arrays should use `dimension_separator` instead.
            Default is ``("default", "/")``.
        codecs : Sequence of Codecs or dicts, optional
            An iterable of Codec or dict serializations of Codecs. The elements of
            this collection specify the transformation from array values to stored bytes.
            Zarr format 3 only. Zarr format 2 arrays should use ``filters`` and ``compressor`` instead.

            If no codecs are provided, default codecs will be used:

            - For numeric arrays, the default is ``BytesCodec`` and ``ZstdCodec``.
            - For Unicode strings, the default is ``VLenUTF8Codec`` and ``ZstdCodec``.
            - For bytes or objects, the default is ``VLenBytesCodec`` and ``ZstdCodec``.

            These defaults can be changed by modifying the value of ``array.v3_default_filters``,
            ``array.v3_default_serializer`` and ``array.v3_default_compressors`` in :mod:`zarr.core.config`.
        dimension_names : Iterable[str], optional
            The names of the dimensions (default is None).
            Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
        chunks : ChunkCoords, optional
            The shape of the array's chunks.
            Zarr format 2 only. Zarr format 3 arrays should use ``chunk_shape`` instead.
            If not specified, default are guessed based on the shape and dtype.
        dimension_separator : Literal[".", "/"], optional
            The dimension separator (default is ".").
            Zarr format 2 only. Zarr format 3 arrays should use ``chunk_key_encoding`` instead.
        order : Literal["C", "F"], optional
            The memory of the array (default is "C").
            If ``zarr_format`` is 2, this parameter sets the memory order of the array.
            If `zarr_format`` is 3, then this parameter is deprecated, because memory order
            is a runtime parameter for Zarr 3 arrays. The recommended way to specify the memory
            order for Zarr 3 arrays is via the ``config`` parameter, e.g. ``{'order': 'C'}``.
        filters : list[dict[str, JSON]], optional
            Sequence of filters to use to encode chunk data prior to compression.
            Zarr format 2 only. Zarr format 3 arrays should use ``codecs`` instead. If no ``filters``
            are provided, a default set of filters will be used.
            These defaults can be changed by modifying the value of ``array.v2_default_filters`` in :mod:`zarr.core.config`.
        compressor : dict[str, JSON], optional
            Primary compressor to compress chunk data.
            Zarr format 2 only. Zarr format 3 arrays should use ``codecs`` instead.

            If no ``compressor`` is provided, a default compressor will be used:

            - For numeric arrays, the default is ``ZstdCodec``.
            - For Unicode strings, the default is ``VLenUTF8Codec``.
            - For bytes or objects, the default is ``VLenBytesCodec``.

            These defaults can be changed by modifying the value of ``array.v2_default_compressor`` in :mod:`zarr.core.config`.
        overwrite : bool, optional
            Whether to raise an error if the store already exists (default is False).

        Returns
        -------
        Array
            Array created from the store.
        """
        return cls._create(
            store,
            # v2 and v3
            shape=shape,
            dtype=dtype,
            zarr_format=zarr_format,
            attributes=attributes,
            fill_value=fill_value,
            # v3 only
            chunk_shape=chunk_shape,
            chunk_key_encoding=chunk_key_encoding,
            codecs=codecs,
            dimension_names=dimension_names,
            # v2 only
            chunks=chunks,
            dimension_separator=dimension_separator,
            order=order,
            filters=filters,
            compressor=compressor,
            # runtime
            overwrite=overwrite,
            config=config,
        )

    @classmethod
    def _create(
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
        order: MemoryOrder | None = None,
        filters: list[dict[str, JSON]] | None = None,
        compressor: dict[str, JSON] | None = None,
        # runtime
        overwrite: bool = False,
        config: ArrayConfigLike | None = None,
    ) -> Array:
        """Creates a new Array instance from an initialized store.
        See :func:`Array.create` for more details.
        Deprecated in favor of :func:`zarr.create_array`.
        """
        async_array = sync(
            AsyncArray._create(
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
                overwrite=overwrite,
                config=config,
            ),
        )
        return cls(async_array)

    @classmethod
    def from_dict(
        cls,
        store_path: StorePath,
        data: dict[str, JSON],
    ) -> Array:
        """
        Create a Zarr array from a dictionary.

        Parameters
        ----------
        store_path : StorePath
            The path within the store where the array should be created.

        data : dict
            A dictionary representing the array data. This dictionary should include necessary metadata
            for the array, such as shape, dtype, fill value, and attributes.

        Returns
        -------
        Array
            The created Zarr array.

        Raises
        ------
        ValueError
            If the dictionary data is invalid or missing required fields for array creation.
        """
        async_array = AsyncArray.from_dict(store_path=store_path, data=data)
        return cls(async_array)

    @classmethod
    def open(
        cls,
        store: StoreLike,
    ) -> Array:
        """Opens an existing Array from a store.

        Parameters
        ----------
        store : Store
            Store containing the Array.

        Returns
        -------
        Array
            Array opened from the store.
        """
        async_array = sync(AsyncArray.open(store))
        return cls(async_array)

    @property
    def store(self) -> Store:
        return self._async_array.store

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions in the array.

        Returns
        -------
        int
            The number of dimensions in the array.
        """
        return self._async_array.ndim

    @property
    def shape(self) -> ChunkCoords:
        """Returns the shape of the array.

        Returns
        -------
        ChunkCoords
            The shape of the array.
        """
        return self._async_array.shape

    @shape.setter
    def shape(self, value: ChunkCoords) -> None:
        """Sets the shape of the array by calling resize."""
        self.resize(value)

    @property
    def chunks(self) -> ChunkCoords:
        """Returns a tuple of integers describing the length of each dimension of a chunk of the array.
        If sharding is used the inner chunk shape is returned.

        Only defined for arrays using using `RegularChunkGrid`.
        If array doesn't use `RegularChunkGrid`, `NotImplementedError` is raised.

        Returns
        -------
        tuple
            A tuple of integers representing the length of each dimension of a chunk.
        """
        return self._async_array.chunks

    @property
    def shards(self) -> ChunkCoords | None:
        """Returns a tuple of integers describing the length of each dimension of a shard of the array.
        Returns None if sharding is not used.

        Only defined for arrays using using `RegularChunkGrid`.
        If array doesn't use `RegularChunkGrid`, `NotImplementedError` is raised.

        Returns
        -------
        tuple | None
            A tuple of integers representing the length of each dimension of a shard or None if sharding is not used.
        """
        return self._async_array.shards

    @property
    def size(self) -> int:
        """Returns the total number of elements in the array.

        Returns
        -------
        int
            Total number of elements in the array.
        """
        return self._async_array.size

    @property
    def dtype(self) -> np.dtype[Any]:
        """Returns the NumPy data type.

        Returns
        -------
        np.dtype
            The NumPy data type.
        """
        return self._async_array.dtype

    @property
    def attrs(self) -> Attributes:
        """Returns a MutableMapping containing user-defined attributes.

        Returns
        -------
        attrs : MutableMapping
            A MutableMapping object containing user-defined attributes.

        Notes
        -----
        Note that attribute values must be JSON serializable.
        """
        return Attributes(self)

    @property
    def path(self) -> str:
        """Storage path."""
        return self._async_array.path

    @property
    def name(self) -> str:
        """Array name following h5py convention."""
        return self._async_array.name

    @property
    def basename(self) -> str:
        """Final component of name."""
        return self._async_array.basename

    @property
    def metadata(self) -> ArrayMetadata:
        return self._async_array.metadata

    @property
    def store_path(self) -> StorePath:
        return self._async_array.store_path

    @property
    def order(self) -> MemoryOrder:
        return self._async_array.order

    @property
    def read_only(self) -> bool:
        return self._async_array.read_only

    @property
    def fill_value(self) -> Any:
        return self.metadata.fill_value

    @property
    def filters(self) -> tuple[numcodecs.abc.Codec, ...] | tuple[ArrayArrayCodec, ...]:
        """
        Filters that are applied to each chunk of the array, in order, before serializing that
        chunk to bytes.
        """
        return self._async_array.filters

    @property
    def serializer(self) -> None | ArrayBytesCodec:
        """
        Array-to-bytes codec to use for serializing the chunks into bytes.
        """
        return self._async_array.serializer

    @property
    @deprecated("Use Array.compressors instead.")
    def compressor(self) -> numcodecs.abc.Codec | None:
        """
        Compressor that is applied to each chunk of the array.

        .. deprecated:: 3.0.0
            `array.compressor` is deprecated and will be removed in a future release.
            Use `array.compressors` instead.
        """
        return self._async_array.compressor

    @property
    def compressors(self) -> tuple[numcodecs.abc.Codec, ...] | tuple[BytesBytesCodec, ...]:
        """
        Compressors that are applied to each chunk of the array. Compressors are applied in order, and after any
        filters are applied (if any are specified) and the data is serialized into bytes.
        """
        return self._async_array.compressors

    @property
    def cdata_shape(self) -> ChunkCoords:
        """
        The shape of the chunk grid for this array.
        """
        return tuple(starmap(ceildiv, zip(self.shape, self.chunks, strict=False)))

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
        origin : Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape : Sequence[int] | None, default=None
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
        The total number of bytes that can be stored in the chunks of this array.

        Notes
        -----
        This value is calculated by multiplying the number of elements in the array and the size
        of each element, the latter of which is determined by the dtype of the array.
        For this reason, ``nbytes`` will likely be inaccurate for arrays with variable-length
        dtypes. It is not possible to determine the size of an array with variable-length elements
        from the shape and dtype alone.
        """
        return self._async_array.nbytes

    @property
    def nchunks_initialized(self) -> int:
        """
        Calculate the number of chunks that have been initialized, i.e. the number of chunks that have
        been persisted to the storage backend.

        Returns
        -------
        nchunks_initialized : int
            The number of chunks that have been initialized.

        Notes
        -----
        On :class:`Array` this is a (synchronous) property, unlike asynchronous function
        :meth:`AsyncArray.nchunks_initialized`.

        Examples
        --------
        >>> arr = await zarr.create(shape=(10,), chunks=(2,))
        >>> arr.nchunks_initialized
        0
        >>> arr[:5] = 1
        >>> arr.nchunks_initialized
        3
        """
        return sync(self._async_array.nchunks_initialized())

    def nbytes_stored(self) -> int:
        """
        Determine the size, in bytes, of the array actually written to the store.

        Returns
        -------
        size : int
        """
        return sync(self._async_array.nbytes_stored())

    def _iter_chunk_keys(
        self, origin: Sequence[int] | None = None, selection_shape: Sequence[int] | None = None
    ) -> Iterator[str]:
        """
        Iterate over the storage keys of each chunk, relative to an optional origin, and optionally
        limited to a contiguous region in chunk grid coordinates.

        Parameters
        ----------
        origin : Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape : Sequence[int] | None, default=None
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
        origin : Sequence[int] | None, default=None
            The origin of the selection relative to the array's chunk grid.
        selection_shape : Sequence[int] | None, default=None
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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=(10,),
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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=(10, 10),
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

        For arrays with a structured dtype, see Zarr format 2 for examples of how to use
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

        For arrays with a structured dtype, see Zarr format 2 for examples of how to use
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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=(3,),
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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=(5, 5, 5),
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

        For arrays with a structured dtype, see Zarr format 2 for examples of how to use
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
        For arrays with a structured dtype, see Zarr format 2 for examples of how to use
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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=data.shape,
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
        mask : ndarray, bool
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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=data.shape,
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
        mask : ndarray, bool
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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=(3, 3),
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

        if not is_scalar(value, self.dtype) and (
            isinstance(value, NDArrayLike) and indexer.shape != value.shape
        ):
            raise ValueError(
                f"Attempting to set a selection of {indexer.sel_shape[0]} "
                f"elements with an array of {value.shape[0]} elements."
            )

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
            >>> z = zarr.create_array(
            >>>        StorePath(MemoryStore(mode="w")),
            >>>        shape=data.shape,
            >>>        chunks=(3, 3),
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

    def resize(self, new_shape: ShapeLike) -> None:
        """
        Change the shape of the array by growing or shrinking one or more
        dimensions.

        Parameters
        ----------
        new_shape : tuple
            New shape of the array.

        Notes
        -----
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
        sync(self._async_array.resize(new_shape))

    def append(self, data: npt.ArrayLike, axis: int = 0) -> ChunkCoords:
        """Append `data` to `axis`.

        Parameters
        ----------
        data : array-like
            Data to be appended.
        axis : int
            Axis along which to append.

        Returns
        -------
        new_shape : tuple

        Notes
        -----
        The size of all dimensions other than `axis` must match between this
        array and `data`.

        Examples
        --------
        >>> import numpy as np
        >>> import zarr
        >>> a = np.arange(10000000, dtype='i4').reshape(10000, 1000)
        >>> z = zarr.array(a, chunks=(1000, 100))
        >>> z.shape
        (10000, 1000)
        >>> z.append(a)
        (20000, 1000)
        >>> z.append(np.vstack([a, a]), axis=1)
        (20000, 2000)
        >>> z.shape
        (20000, 2000)
        """
        return sync(self._async_array.append(data, axis=axis))

    def update_attributes(self, new_attributes: dict[str, JSON]) -> Array:
        """
        Update the array's attributes.

        Parameters
        ----------
        new_attributes : dict
            A dictionary of new attributes to update or add to the array. The keys represent attribute
            names, and the values must be JSON-compatible.

        Returns
        -------
        Array
            The array with the updated attributes.

        Raises
        ------
        ValueError
            If the attributes are invalid or incompatible with the array's metadata.

        Notes
        -----
        - The updated attributes will be merged with existing attributes, and any conflicts will be
          overwritten by the new values.
        """
        # TODO: remove this cast when type inference improves
        new_array = sync(self._async_array.update_attributes(new_attributes))
        # TODO: remove this cast when type inference improves
        _new_array = cast(AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata], new_array)
        return type(self)(_new_array)

    def __repr__(self) -> str:
        return f"<Array {self.store_path} shape={self.shape} dtype={self.dtype}>"

    @property
    def info(self) -> Any:
        """
        Return the statically known information for an array.

        Returns
        -------
        ArrayInfo

        See Also
        --------
        Array.info_complete
            All information about a group, including dynamic information
            like the number of bytes and chunks written.

        Examples
        --------
        >>> arr = zarr.create(shape=(10,), chunks=(2,), dtype="float32")
        >>> arr.info
        Type               : Array
        Zarr format        : 3
        Data type          : DataType.float32
        Shape              : (10,)
        Chunk shape        : (2,)
        Order              : C
        Read-only          : False
        Store type         : MemoryStore
        Codecs             : [BytesCodec(endian=<Endian.little: 'little'>)]
        No. bytes          : 40
        """
        return self._async_array.info

    def info_complete(self) -> Any:
        """
        Returns all the information about an array, including information from the Store.

        In addition to the statically known information like ``name`` and ``zarr_format``,
        this includes additional information like the size of the array in bytes and
        the number of chunks written.

        Note that this method will need to read metadata from the store.

        Returns
        -------
        ArrayInfo

        See Also
        --------
        Array.info
            The statically known subset of metadata about an array.
        """
        return sync(self._async_array.info_complete())


async def chunks_initialized(
    array: AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata],
) -> tuple[str, ...]:
    """
    Return the keys of the chunks that have been persisted to the storage backend.

    Parameters
    ----------
    array : AsyncArray
        The array to inspect.

    Returns
    -------
    chunks_initialized : tuple[str, ...]
        The keys of the chunks that have been initialized.

    See Also
    --------
    nchunks_initialized

    """
    store_contents = [
        x async for x in array.store_path.store.list_prefix(prefix=array.store_path.path)
    ]
    return tuple(chunk_key for chunk_key in array._iter_chunk_keys() if chunk_key in store_contents)


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


def _get_default_codecs(
    np_dtype: np.dtype[Any],
) -> tuple[Codec, ...]:
    filters, serializer, compressors = _get_default_chunk_encoding_v3(np_dtype)
    return filters + (serializer,) + compressors


FiltersLike: TypeAlias = (
    Iterable[dict[str, JSON] | ArrayArrayCodec | numcodecs.abc.Codec]
    | ArrayArrayCodec
    | Iterable[numcodecs.abc.Codec]
    | numcodecs.abc.Codec
    | Literal["auto"]
    | None
)
CompressorLike: TypeAlias = dict[str, JSON] | BytesBytesCodec | numcodecs.abc.Codec | None
CompressorsLike: TypeAlias = (
    Iterable[dict[str, JSON] | BytesBytesCodec | numcodecs.abc.Codec]
    | dict[str, JSON]
    | BytesBytesCodec
    | numcodecs.abc.Codec
    | Literal["auto"]
    | None
)
SerializerLike: TypeAlias = dict[str, JSON] | ArrayBytesCodec | Literal["auto"]


class ShardsConfigParam(TypedDict):
    shape: ChunkCoords
    index_location: ShardingCodecIndexLocation | None


ShardsLike: TypeAlias = ChunkCoords | ShardsConfigParam | Literal["auto"]


async def init_array(
    *,
    store_path: StorePath,
    shape: ShapeLike,
    dtype: npt.DTypeLike,
    chunks: ChunkCoords | Literal["auto"] = "auto",
    shards: ShardsLike | None = None,
    filters: FiltersLike = "auto",
    compressors: CompressorsLike = "auto",
    serializer: SerializerLike = "auto",
    fill_value: Any | None = None,
    order: MemoryOrder | None = None,
    zarr_format: ZarrFormat | None = 3,
    attributes: dict[str, JSON] | None = None,
    chunk_key_encoding: ChunkKeyEncodingLike | None = None,
    dimension_names: Iterable[str] | None = None,
    overwrite: bool = False,
    config: ArrayConfigLike | None,
) -> AsyncArray[ArrayV3Metadata] | AsyncArray[ArrayV2Metadata]:
    """Create and persist an array metadata document.

    Parameters
    ----------
    store_path : StorePath
        StorePath instance. The path attribute is the name of the array to initialize.
    shape : ChunkCoords
        Shape of the array.
    dtype : npt.DTypeLike
        Data type of the array.
    chunks : ChunkCoords, optional
        Chunk shape of the array.
        If not specified, default are guessed based on the shape and dtype.
    shards : ChunkCoords, optional
        Shard shape of the array. The default value of ``None`` results in no sharding at all.
    filters : Iterable[Codec], optional
        Iterable of filters to apply to each chunk of the array, in order, before serializing that
        chunk to bytes.

        For Zarr format 3, a "filter" is a codec that takes an array and returns an array,
        and these values must be instances of ``ArrayArrayCodec``, or dict representations
        of ``ArrayArrayCodec``.
        If no ``filters`` are provided, a default set of filters will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_filters``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default filters.

        For Zarr format 2, a "filter" can be any numcodecs codec; you should ensure that the
        the order if your filters is consistent with the behavior of each filter.
        If no ``filters`` are provided, a default set of filters will be used.
        These defaults can be changed by modifying the value of ``array.v2_default_filters``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default filters.
    compressors : Iterable[Codec], optional
        List of compressors to apply to the array. Compressors are applied in order, and after any
        filters are applied (if any are specified) and the data is serialized into bytes.

        For Zarr format 3, a "compressor" is a codec that takes a bytestream, and
        returns another bytestream. Multiple compressors my be provided for Zarr format 3.
        If no ``compressors`` are provided, a default set of compressors will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_compressors``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default compressors.

        For Zarr format 2, a "compressor" can be any numcodecs codec. Only a single compressor may
        be provided for Zarr format 2.
        If no ``compressor`` is provided, a default compressor will be used.
        in :mod:`zarr.core.config`.
        Use ``None`` to omit the default compressor.
    serializer : dict[str, JSON] | ArrayBytesCodec, optional
        Array-to-bytes codec to use for encoding the array data.
        Zarr format 3 only. Zarr format 2 arrays use implicit array-to-bytes conversion.
        If no ``serializer`` is provided, a default serializer will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_serializer``
        in :mod:`zarr.core.config`.
    fill_value : Any, optional
        Fill value for the array.
    order : {"C", "F"}, optional
        The memory of the array (default is "C").
        For Zarr format 2, this parameter sets the memory order of the array.
        For Zarr format 3, this parameter is deprecated, because memory order
        is a runtime parameter for Zarr format 3 arrays. The recommended way to specify the memory
        order for Zarr format 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
        If no ``order`` is provided, a default order will be used.
        This default can be changed by modifying the value of ``array.order`` in :mod:`zarr.core.config`.
    zarr_format : {2, 3}, optional
        The zarr format to use when saving.
    attributes : dict, optional
        Attributes for the array.
    chunk_key_encoding : ChunkKeyEncodingLike, optional
        A specification of how the chunk keys are represented in storage.
        For Zarr format 3, the default is ``{"name": "default", "separator": "/"}}``.
        For Zarr format 2, the default is ``{"name": "v2", "separator": "."}}``.
    dimension_names : Iterable[str], optional
        The names of the dimensions (default is None).
        Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
    overwrite : bool, default False
        Whether to overwrite an array with the same name in the store, if one exists.
    config : ArrayConfigLike or None, optional
        Configuration for this array.

    Returns
    -------
    AsyncArray
        The AsyncArray.
    """

    if zarr_format is None:
        zarr_format = _default_zarr_format()

    from zarr.codecs.sharding import ShardingCodec, ShardingCodecIndexLocation

    dtype_parsed = parse_dtype(dtype, zarr_format=zarr_format)
    shape_parsed = parse_shapelike(shape)
    chunk_key_encoding_parsed = _parse_chunk_key_encoding(
        chunk_key_encoding, zarr_format=zarr_format
    )

    if overwrite:
        if store_path.store.supports_deletes:
            await store_path.delete_dir()
        else:
            await ensure_no_existing_node(store_path, zarr_format=zarr_format)
    else:
        await ensure_no_existing_node(store_path, zarr_format=zarr_format)

    shard_shape_parsed, chunk_shape_parsed = _auto_partition(
        array_shape=shape_parsed, shard_shape=shards, chunk_shape=chunks, dtype=dtype_parsed
    )
    chunks_out: tuple[int, ...]
    meta: ArrayV2Metadata | ArrayV3Metadata
    if zarr_format == 2:
        if shard_shape_parsed is not None:
            msg = (
                "Zarr format 2 arrays can only be created with `shard_shape` set to `None`. "
                f"Got `shard_shape={shards}` instead."
            )

            raise ValueError(msg)
        if serializer != "auto":
            raise ValueError("Zarr format 2 arrays do not support `serializer`.")

        filters_parsed, compressor_parsed = _parse_chunk_encoding_v2(
            compressor=compressors, filters=filters, dtype=np.dtype(dtype)
        )

        if dimension_names is not None:
            raise ValueError("Zarr format 2 arrays do not support dimension names.")
        if order is None:
            order_parsed = zarr_config.get("array.order")
        else:
            order_parsed = order

        meta = AsyncArray._create_metadata_v2(
            shape=shape_parsed,
            dtype=dtype_parsed,
            chunks=chunk_shape_parsed,
            dimension_separator=chunk_key_encoding_parsed.separator,
            fill_value=fill_value,
            order=order_parsed,
            filters=filters_parsed,
            compressor=compressor_parsed,
            attributes=attributes,
        )
    else:
        array_array, array_bytes, bytes_bytes = _parse_chunk_encoding_v3(
            compressors=compressors,
            filters=filters,
            serializer=serializer,
            dtype=dtype_parsed,
        )
        sub_codecs = cast(tuple[Codec, ...], (*array_array, array_bytes, *bytes_bytes))
        codecs_out: tuple[Codec, ...]
        if shard_shape_parsed is not None:
            index_location = None
            if isinstance(shards, dict):
                index_location = ShardingCodecIndexLocation(shards.get("index_location", None))
            if index_location is None:
                index_location = ShardingCodecIndexLocation.end
            sharding_codec = ShardingCodec(
                chunk_shape=chunk_shape_parsed, codecs=sub_codecs, index_location=index_location
            )
            sharding_codec.validate(
                shape=chunk_shape_parsed,
                dtype=dtype_parsed,
                chunk_grid=RegularChunkGrid(chunk_shape=shard_shape_parsed),
            )
            codecs_out = (sharding_codec,)
            chunks_out = shard_shape_parsed
        else:
            chunks_out = chunk_shape_parsed
            codecs_out = sub_codecs

        meta = AsyncArray._create_metadata_v3(
            shape=shape_parsed,
            dtype=dtype_parsed,
            fill_value=fill_value,
            chunk_shape=chunks_out,
            chunk_key_encoding=chunk_key_encoding_parsed,
            codecs=codecs_out,
            dimension_names=dimension_names,
            attributes=attributes,
        )

    arr = AsyncArray(metadata=meta, store_path=store_path, config=config)
    await arr._save_metadata(meta, ensure_parents=True)
    return arr


async def create_array(
    store: str | StoreLike,
    *,
    name: str | None = None,
    shape: ShapeLike | None = None,
    dtype: npt.DTypeLike | None = None,
    data: np.ndarray[Any, np.dtype[Any]] | None = None,
    chunks: ChunkCoords | Literal["auto"] = "auto",
    shards: ShardsLike | None = None,
    filters: FiltersLike = "auto",
    compressors: CompressorsLike = "auto",
    serializer: SerializerLike = "auto",
    fill_value: Any | None = None,
    order: MemoryOrder | None = None,
    zarr_format: ZarrFormat | None = 3,
    attributes: dict[str, JSON] | None = None,
    chunk_key_encoding: ChunkKeyEncodingLike | None = None,
    dimension_names: Iterable[str] | None = None,
    storage_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    config: ArrayConfigLike | None = None,
    write_data: bool = True,
) -> AsyncArray[ArrayV2Metadata] | AsyncArray[ArrayV3Metadata]:
    """Create an array.

    Parameters
    ----------
    store : str or Store
        Store or path to directory in file system or name of zip file.
    name : str or None, optional
        The name of the array within the store. If ``name`` is ``None``, the array will be located
        at the root of the store.
    shape : ChunkCoords, optional
        Shape of the array. Can be ``None`` if ``data`` is provided.
    dtype : npt.DTypeLike | None
        Data type of the array. Can be ``None`` if ``data`` is provided.
    data : Array-like data to use for initializing the array. If this parameter is provided, the
        ``shape`` and ``dtype`` parameters must be identical to ``data.shape`` and ``data.dtype``,
        or ``None``.
    chunks : ChunkCoords, optional
        Chunk shape of the array.
        If not specified, default are guessed based on the shape and dtype.
    shards : ChunkCoords, optional
        Shard shape of the array. The default value of ``None`` results in no sharding at all.
    filters : Iterable[Codec], optional
        Iterable of filters to apply to each chunk of the array, in order, before serializing that
        chunk to bytes.

        For Zarr format 3, a "filter" is a codec that takes an array and returns an array,
        and these values must be instances of ``ArrayArrayCodec``, or dict representations
        of ``ArrayArrayCodec``.
        If no ``filters`` are provided, a default set of filters will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_filters``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default filters.

        For Zarr format 2, a "filter" can be any numcodecs codec; you should ensure that the
        the order if your filters is consistent with the behavior of each filter.
        If no ``filters`` are provided, a default set of filters will be used.
        These defaults can be changed by modifying the value of ``array.v2_default_filters``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default filters.
    compressors : Iterable[Codec], optional
        List of compressors to apply to the array. Compressors are applied in order, and after any
        filters are applied (if any are specified) and the data is serialized into bytes.

        For Zarr format 3, a "compressor" is a codec that takes a bytestream, and
        returns another bytestream. Multiple compressors my be provided for Zarr format 3.
        If no ``compressors`` are provided, a default set of compressors will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_compressors``
        in :mod:`zarr.core.config`.
        Use ``None`` to omit default compressors.

        For Zarr format 2, a "compressor" can be any numcodecs codec. Only a single compressor may
        be provided for Zarr format 2.
        If no ``compressor`` is provided, a default compressor will be used.
        in :mod:`zarr.core.config`.
        Use ``None`` to omit the default compressor.
    serializer : dict[str, JSON] | ArrayBytesCodec, optional
        Array-to-bytes codec to use for encoding the array data.
        Zarr format 3 only. Zarr format 2 arrays use implicit array-to-bytes conversion.
        If no ``serializer`` is provided, a default serializer will be used.
        These defaults can be changed by modifying the value of ``array.v3_default_serializer``
        in :mod:`zarr.core.config`.
    fill_value : Any, optional
        Fill value for the array.
    order : {"C", "F"}, optional
        The memory of the array (default is "C").
        For Zarr format 2, this parameter sets the memory order of the array.
        For Zarr format 3, this parameter is deprecated, because memory order
        is a runtime parameter for Zarr format 3 arrays. The recommended way to specify the memory
        order for Zarr format 3 arrays is via the ``config`` parameter, e.g. ``{'config': 'C'}``.
        If no ``order`` is provided, a default order will be used.
        This default can be changed by modifying the value of ``array.order`` in :mod:`zarr.core.config`.
    zarr_format : {2, 3}, optional
        The zarr format to use when saving.
    attributes : dict, optional
        Attributes for the array.
    chunk_key_encoding : ChunkKeyEncodingLike, optional
        A specification of how the chunk keys are represented in storage.
        For Zarr format 3, the default is ``{"name": "default", "separator": "/"}}``.
        For Zarr format 2, the default is ``{"name": "v2", "separator": "."}}``.
    dimension_names : Iterable[str], optional
        The names of the dimensions (default is None).
        Zarr format 3 only. Zarr format 2 arrays should not use this parameter.
    storage_options : dict, optional
        If using an fsspec URL to create the store, these will be passed to the backend implementation.
        Ignored otherwise.
    overwrite : bool, default False
        Whether to overwrite an array with the same name in the store, if one exists.
    config : ArrayConfigLike, optional
        Runtime configuration for the array.
    write_data : bool
        If a pre-existing array-like object was provided to this function via the ``data`` parameter
        then ``write_data`` determines whether the values in that array-like object should be
        written to the Zarr array created by this function. If ``write_data`` is ``False``, then the
        array will be left empty.

    Returns
    -------
    AsyncArray
        The array.

    Examples
    --------
    >>> import zarr
    >>> store = zarr.storage.MemoryStore(mode='w')
    >>> async_arr = await zarr.api.asynchronous.create_array(
    >>>     store=store,
    >>>     shape=(100,100),
    >>>     chunks=(10,10),
    >>>     dtype='i4',
    >>>     fill_value=0)
    <AsyncArray memory://140349042942400 shape=(100, 100) dtype=int32>
    """
    mode: Literal["a"] = "a"
    store_path = await make_store_path(store, path=name, mode=mode, storage_options=storage_options)

    data_parsed, shape_parsed, dtype_parsed = _parse_data_params(
        data=data, shape=shape, dtype=dtype
    )
    result = await init_array(
        store_path=store_path,
        shape=shape_parsed,
        dtype=dtype_parsed,
        chunks=chunks,
        shards=shards,
        filters=filters,
        compressors=compressors,
        serializer=serializer,
        fill_value=fill_value,
        order=order,
        zarr_format=zarr_format,
        attributes=attributes,
        chunk_key_encoding=chunk_key_encoding,
        dimension_names=dimension_names,
        overwrite=overwrite,
        config=config,
    )

    if write_data is True and data_parsed is not None:
        await result._set_selection(
            BasicIndexer(..., shape=result.shape, chunk_grid=result.metadata.chunk_grid),
            data_parsed,
            prototype=default_buffer_prototype(),
        )
    return result


def _parse_chunk_key_encoding(
    data: ChunkKeyEncodingLike | None, zarr_format: ZarrFormat
) -> ChunkKeyEncoding:
    """
    Take an implicit specification of a chunk key encoding and parse it into a ChunkKeyEncoding object.
    """
    if data is None:
        if zarr_format == 2:
            result = ChunkKeyEncoding.from_dict({"name": "v2", "separator": "."})
        else:
            result = ChunkKeyEncoding.from_dict({"name": "default", "separator": "/"})
    elif isinstance(data, ChunkKeyEncoding):
        result = data
    else:
        result = ChunkKeyEncoding.from_dict(data)
    if zarr_format == 2 and result.name != "v2":
        msg = (
            "Invalid chunk key encoding. For Zarr format 2 arrays, the `name` field of the "
            f"chunk key encoding must be 'v2'. Got `name` = {result.name} instead."
        )
        raise ValueError(msg)
    return result


def _get_default_chunk_encoding_v3(
    np_dtype: np.dtype[Any],
) -> tuple[tuple[ArrayArrayCodec, ...], ArrayBytesCodec, tuple[BytesBytesCodec, ...]]:
    """
    Get the default ArrayArrayCodecs, ArrayBytesCodec, and BytesBytesCodec for a given dtype.
    """
    dtype = DataType.from_numpy(np_dtype)
    if dtype == DataType.string:
        dtype_key = "string"
    elif dtype == DataType.bytes:
        dtype_key = "bytes"
    else:
        dtype_key = "numeric"

    default_filters = zarr_config.get("array.v3_default_filters").get(dtype_key)
    default_serializer = zarr_config.get("array.v3_default_serializer").get(dtype_key)
    default_compressors = zarr_config.get("array.v3_default_compressors").get(dtype_key)

    filters = tuple(_parse_array_array_codec(codec_dict) for codec_dict in default_filters)
    serializer = _parse_array_bytes_codec(default_serializer)
    compressors = tuple(_parse_bytes_bytes_codec(codec_dict) for codec_dict in default_compressors)

    return filters, serializer, compressors


def _get_default_chunk_encoding_v2(
    np_dtype: np.dtype[Any],
) -> tuple[tuple[numcodecs.abc.Codec, ...] | None, numcodecs.abc.Codec | None]:
    """
    Get the default chunk encoding for Zarr format 2 arrays, given a dtype
    """

    compressor_dict = _default_compressor(np_dtype)
    filter_dicts = _default_filters(np_dtype)

    compressor = None
    if compressor_dict is not None:
        compressor = numcodecs.get_codec(compressor_dict)

    filters = None
    if filter_dicts is not None:
        filters = tuple(numcodecs.get_codec(f) for f in filter_dicts)

    return filters, compressor


def _parse_chunk_encoding_v2(
    *,
    compressor: CompressorsLike,
    filters: FiltersLike,
    dtype: np.dtype[Any],
) -> tuple[tuple[numcodecs.abc.Codec, ...] | None, numcodecs.abc.Codec | None]:
    """
    Generate chunk encoding classes for Zarr format 2 arrays with optional defaults.
    """
    default_filters, default_compressor = _get_default_chunk_encoding_v2(dtype)

    _filters: tuple[numcodecs.abc.Codec, ...] | None
    _compressor: numcodecs.abc.Codec | None

    if compressor is None or compressor == ():
        _compressor = None
    elif compressor == "auto":
        _compressor = default_compressor
    elif isinstance(compressor, tuple | list) and len(compressor) == 1:
        _compressor = parse_compressor(compressor[0])
    else:
        if isinstance(compressor, Iterable) and not isinstance(compressor, dict):
            msg = f"For Zarr format 2 arrays, the `compressor` must be a single codec. Got an iterable with type {type(compressor)} instead."
            raise TypeError(msg)
        _compressor = parse_compressor(compressor)

    if filters is None:
        _filters = None
    elif filters == "auto":
        _filters = default_filters
    else:
        if isinstance(filters, Iterable):
            for idx, f in enumerate(filters):
                if not isinstance(f, numcodecs.abc.Codec):
                    msg = (
                        "For Zarr format 2 arrays, all elements of `filters` must be numcodecs codecs. "
                        f"Element at index {idx} has type {type(f)}, which is not a numcodecs codec."
                    )
                    raise TypeError(msg)
        _filters = parse_filters(filters)

    return _filters, _compressor


def _parse_chunk_encoding_v3(
    *,
    compressors: CompressorsLike,
    filters: FiltersLike,
    serializer: SerializerLike,
    dtype: np.dtype[Any],
) -> tuple[tuple[ArrayArrayCodec, ...], ArrayBytesCodec, tuple[BytesBytesCodec, ...]]:
    """
    Generate chunk encoding classes for v3 arrays with optional defaults.
    """
    default_array_array, default_array_bytes, default_bytes_bytes = _get_default_chunk_encoding_v3(
        dtype
    )

    if filters is None:
        out_array_array: tuple[ArrayArrayCodec, ...] = ()
    elif filters == "auto":
        out_array_array = default_array_array
    else:
        maybe_array_array: Iterable[Codec | dict[str, JSON]]
        if isinstance(filters, dict | Codec):
            maybe_array_array = (filters,)
        else:
            maybe_array_array = cast(Iterable[Codec | dict[str, JSON]], filters)
        out_array_array = tuple(_parse_array_array_codec(c) for c in maybe_array_array)

    if serializer == "auto":
        out_array_bytes = default_array_bytes
    else:
        out_array_bytes = _parse_array_bytes_codec(serializer)

    if compressors is None:
        out_bytes_bytes: tuple[BytesBytesCodec, ...] = ()
    elif compressors == "auto":
        out_bytes_bytes = default_bytes_bytes
    else:
        maybe_bytes_bytes: Iterable[Codec | dict[str, JSON]]
        if isinstance(compressors, dict | Codec):
            maybe_bytes_bytes = (compressors,)
        else:
            maybe_bytes_bytes = cast(Iterable[Codec | dict[str, JSON]], compressors)

        out_bytes_bytes = tuple(_parse_bytes_bytes_codec(c) for c in maybe_bytes_bytes)

    return out_array_array, out_array_bytes, out_bytes_bytes


def _parse_deprecated_compressor(
    compressor: CompressorLike | None, compressors: CompressorsLike, zarr_format: int = 3
) -> CompressorsLike | None:
    if compressor != "auto":
        if compressors != "auto":
            raise ValueError("Cannot specify both `compressor` and `compressors`.")
        if zarr_format == 3:
            warn(
                "The `compressor` argument is deprecated. Use `compressors` instead.",
                category=UserWarning,
                stacklevel=2,
            )
        if compressor is None:
            # "no compression"
            compressors = ()
        else:
            compressors = (compressor,)
    elif zarr_format == 2 and compressor == compressors == "auto":
        compressors = ({"id": "blosc"},)
    return compressors


def _parse_data_params(
    *,
    data: np.ndarray[Any, np.dtype[Any]] | None,
    shape: ShapeLike | None,
    dtype: npt.DTypeLike | None,
) -> tuple[np.ndarray[Any, np.dtype[Any]] | None, ShapeLike, npt.DTypeLike]:
    """
    Ensure an array-like ``data`` parameter is consistent with the ``dtype`` and ``shape``
    parameters.
    """
    if data is None:
        if shape is None:
            msg = (
                "The data parameter was set to None, but shape was not specified. "
                "Either provide a value for data, or specify shape."
            )
            raise ValueError(msg)
        shape_out = shape
        if dtype is None:
            msg = (
                "The data parameter was set to None, but dtype was not specified."
                "Either provide an array-like value for data, or specify dtype."
            )
            raise ValueError(msg)
        dtype_out = dtype
    else:
        if shape is not None:
            msg = (
                "The data parameter was used, but the shape parameter was also "
                "used. This is an error. Either use the data parameter, or the shape parameter, "
                "but not both."
            )
            raise ValueError(msg)
        shape_out = data.shape
        if dtype is not None:
            msg = (
                "The data parameter was used, but the dtype parameter was also "
                "used. This is an error. Either use the data parameter, or the dtype parameter, "
                "but not both."
            )
            raise ValueError(msg)
        dtype_out = data.dtype
    return data, shape_out, dtype_out
