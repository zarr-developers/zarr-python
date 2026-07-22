"""The default engines: today's codec-pipeline machinery behind the protocol."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from zarr.core.array import _get_chunk_spec, create_codec_pipeline
from zarr.core.array_spec import ArraySpec, parse_array_config
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.common import product
from zarr.core.indexing import BasicIndexer
from zarr.core.sync import sync
from zarr.errors import ChunkNotFoundError
from zarr.storage._common import StorePath

if TYPE_CHECKING:
    from zarr.abc.engine import Region
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfig
    from zarr.core.buffer import BufferPrototype, NDArrayLike, NDBuffer
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "DefaultArrayEngine",
    "DefaultAsyncArrayEngine",
    "DefaultAsyncHierarchyEngine",
    "DefaultHierarchyEngine",
]


def _region_to_slices(region: Region) -> tuple[slice, ...]:
    return tuple(slice(s, e) for s, e in zip(region.start, region.end_exclusive, strict=True))


class DefaultAsyncArrayEngine:
    """Codec-pipeline-backed engine. Any store, Zarr v2 and v3."""

    def __init__(self, store_path: StorePath, metadata: ArrayMetadata, config: ArrayConfig) -> None:
        self.store_path = store_path
        self.metadata = metadata
        self.config = config
        self._chunk_grid = ChunkGrid.from_metadata(metadata)
        self._codec_pipeline = create_codec_pipeline(metadata=metadata, store=store_path.store)

    def with_metadata(self, metadata: ArrayMetadata) -> DefaultAsyncArrayEngine:
        return DefaultAsyncArrayEngine(
            store_path=self.store_path, metadata=metadata, config=self.config
        )

    def _indexer(self, region: Region) -> BasicIndexer:
        return BasicIndexer(
            _region_to_slices(region),
            shape=self.metadata.shape,
            chunk_grid=self._chunk_grid,
        )

    def _regular_chunk_spec(
        self, config: ArrayConfig, prototype: BufferPrototype
    ) -> ArraySpec | None:
        """Same optimization as `_get_selection`/`_set_selection`: build one shared
        `ArraySpec` for regular chunk grids instead of a per-chunk lookup.
        """
        if not self._chunk_grid.is_regular:
            return None
        return ArraySpec(
            shape=self._chunk_grid.chunk_shape,
            dtype=self.metadata.dtype,
            fill_value=self.metadata.fill_value,
            config=config,
            prototype=prototype,
        )

    def _v2_order_config(self) -> ArrayConfig:
        # need to use the order from the metadata for v2 (mirrors _get_selection/_set_selection)
        if self.metadata.zarr_format == 2:
            return replace(self.config, order=self.metadata.order)
        return self.config

    async def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> NDArrayLike:
        indexer = self._indexer(selection)
        if self.metadata.zarr_format == 2:
            dtype = self.metadata.dtype.to_native_dtype()
            order = self.metadata.order
        else:
            dtype = self.metadata.data_type.to_native_dtype()
            order = self.config.order
        out_buffer = prototype.nd_buffer.empty(shape=indexer.shape, dtype=dtype, order=order)
        if product(indexer.shape) > 0:
            _config = self._v2_order_config()
            regular_chunk_spec = self._regular_chunk_spec(_config, prototype)
            indexed_chunks = list(indexer)
            results = await self._codec_pipeline.read(
                [
                    (
                        self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                        regular_chunk_spec
                        if regular_chunk_spec is not None
                        else _get_chunk_spec(
                            self.metadata, self._chunk_grid, chunk_coords, _config, prototype
                        ),
                        chunk_selection,
                        out_selection,
                        is_complete_chunk,
                    )
                    for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexed_chunks
                ],
                out_buffer,
                drop_axes=indexer.drop_axes,
            )
            if _config.read_missing_chunks is False:
                missing_info = []
                for i, result in enumerate(results):
                    if result["status"] == "missing":
                        coords = indexed_chunks[i][0]
                        key = self.metadata.encode_chunk_key(coords)
                        missing_info.append(f"  chunk '{key}' (grid position {coords})")
                if missing_info:
                    chunks_str = "\n".join(missing_info)
                    raise ChunkNotFoundError(
                        f"{len(missing_info)} chunk(s) not found in store '{self.store_path}'.\n"
                        f"Set the 'array.read_missing_chunks' config to True to fill "
                        f"missing chunks with the fill value.\n"
                        f"Missing chunks:\n{chunks_str}"
                    )
        return out_buffer.as_ndarray_like()

    async def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        indexer = self._indexer(selection)
        _config = self._v2_order_config()
        regular_chunk_spec = self._regular_chunk_spec(_config, prototype)
        await self._codec_pipeline.write(
            [
                (
                    self.store_path / self.metadata.encode_chunk_key(chunk_coords),
                    regular_chunk_spec
                    if regular_chunk_spec is not None
                    else _get_chunk_spec(
                        self.metadata, self._chunk_grid, chunk_coords, _config, prototype
                    ),
                    chunk_selection,
                    out_selection,
                    is_complete_chunk,
                )
                for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
            ],
            value,
            drop_axes=indexer.drop_axes,
        )


class DefaultArrayEngine:
    """Sync adapter over `DefaultAsyncArrayEngine` via `sync()`."""

    def __init__(self, async_engine: DefaultAsyncArrayEngine) -> None:
        self._async = async_engine

    def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> NDArrayLike:
        return sync(self._async.read_selection(selection, prototype=prototype))

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        sync(self._async.write_selection(selection, value, prototype=prototype))

    def with_metadata(self, metadata: ArrayMetadata) -> DefaultArrayEngine:
        return DefaultArrayEngine(self._async.with_metadata(metadata))


class DefaultAsyncHierarchyEngine:
    """Store-bound factory for default async engines."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> DefaultAsyncArrayEngine:
        return DefaultAsyncArrayEngine(
            store_path=StorePath(self.store, path),
            metadata=metadata,
            config=config if config is not None else parse_array_config(None),
        )


class DefaultHierarchyEngine:
    """Store-bound factory for default sync engines."""

    def __init__(self, store: Store) -> None:
        self._async = DefaultAsyncHierarchyEngine(store)

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> DefaultArrayEngine:
        return DefaultArrayEngine(self._async.array_engine(path, metadata, config))
