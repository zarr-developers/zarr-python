"""The built-in engine: zarr-python's own data path, behind the protocol.

This engine owns no I/O logic. It forwards to the module-level
`_get_selection` / `_set_selection` in `zarr.core.array` — the same functions
the non-engine path calls — so the built-in engine is the identity engine by
construction, and improvements to that path (codec-pipeline changes, chunk-spec
caching, dtype handling) reach it without being mirrored here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.core.array_spec import parse_array_config
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.sync import sync
from zarr.storage._common import StorePath

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.abc.codec import CodecPipeline
    from zarr.abc.engine import SelectionRequest
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfig
    from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar, NDBuffer
    from zarr.core.indexing import Fields
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "DefaultArrayEngine",
    "DefaultAsyncArrayEngine",
    "DefaultAsyncHierarchyEngine",
    "DefaultHierarchyEngine",
    "adopt_codec_pipeline",
]


def adopt_codec_pipeline(engine: object, codec_pipeline: CodecPipeline) -> None:
    """Give a built-in engine a codec pipeline the caller has already built.

    `Array` and `AsyncArray` each build one, and each resolves its own engine.
    Letting the engine build another would duplicate that work on every array
    and would re-emit the codec chain's advisory warnings (e.g. sharding's
    "disables partial reads"), which are meant to fire once per user-facing
    chain rather than once per construction.

    A no-op for any other engine, and for a built-in engine that already has a
    pipeline: an engine that does not use zarr-python's codec pipeline has no
    use for one.
    """
    if isinstance(engine, DefaultArrayEngine):
        engine = engine._async
    if isinstance(engine, DefaultAsyncArrayEngine) and engine._pipeline is None:
        engine._pipeline = codec_pipeline


class DefaultAsyncArrayEngine:
    """Codec-pipeline-backed engine. Any store, Zarr v2 and v3."""

    def __init__(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata,
        config: ArrayConfig,
        codec_pipeline: CodecPipeline | None = None,
    ) -> None:
        self.store_path = store_path
        self.metadata = metadata
        self.config = config
        self._chunk_grid = ChunkGrid.from_metadata(metadata)
        self._pipeline = codec_pipeline

    @property
    def codec_pipeline(self) -> CodecPipeline:
        """The codec pipeline, built on first use if one was not supplied.

        `AsyncArray` passes in the pipeline it has already built. Building a
        second one here would not just duplicate the work — `create_codec_pipeline`
        emits the codec chain's advisory warnings (e.g. sharding's "disables
        partial reads"), which are meant to fire once per user-facing chain, so
        a second construction would also double the warnings.
        """
        if self._pipeline is None:
            # Imported lazily: `zarr.core.array` imports the engine package to
            # resolve `engine=`, while this engine calls back into it.
            from zarr.core.array import create_codec_pipeline

            self._pipeline = create_codec_pipeline(
                metadata=self.metadata, store=self.store_path.store
            )
        return self._pipeline

    def with_metadata(self, metadata: ArrayMetadata) -> DefaultAsyncArrayEngine:
        # The pipeline is metadata-derived, so a new one is built for the new
        # metadata rather than carried over.
        return DefaultAsyncArrayEngine(
            store_path=self.store_path, metadata=metadata, config=self.config
        )

    async def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        from zarr.core.array import _get_selection

        return await _get_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            self._chunk_grid,
            request.indexer,
            prototype=prototype,
            out=out,
            fields=fields,
        )

    async def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        from zarr.core.array import _set_selection

        await _set_selection(
            self.store_path,
            self.metadata,
            self.codec_pipeline,
            self.config,
            self._chunk_grid,
            request.indexer,
            value,
            prototype=prototype,
            fields=fields,
        )


class DefaultArrayEngine:
    """Sync adapter over `DefaultAsyncArrayEngine` via `sync()`."""

    def __init__(self, async_engine: DefaultAsyncArrayEngine) -> None:
        self._async = async_engine

    def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        return sync(
            self._async.read_selection(request, prototype=prototype, out=out, fields=fields)
        )

    def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        sync(self._async.write_selection(request, value, prototype=prototype, fields=fields))

    def with_metadata(self, metadata: ArrayMetadata) -> DefaultArrayEngine:
        return DefaultArrayEngine(self._async.with_metadata(metadata))


class DefaultAsyncHierarchyEngine:
    """Store-bound factory for built-in async engines."""

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
    """Store-bound factory for built-in sync engines."""

    def __init__(self, store: Store) -> None:
        self._async = DefaultAsyncHierarchyEngine(store)

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> DefaultArrayEngine:
        return DefaultArrayEngine(self._async.array_engine(path, metadata, config))
