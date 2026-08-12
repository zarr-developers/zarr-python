"""Zarrista-backed array engines."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from zarr.errors import UnsupportedEngineError
from zarr.zarrista._resolve import dense_forward_box, lazy_view, scatter_points
from zarr.zarrista._source import (
    AsyncZarristaSource,
    ZarristaSource,
    await_on,
    run_off_loop,
    tensor_to_numpy,
)
from zarr.zarrista._translate import translate_store_async, translate_store_sync

if TYPE_CHECKING:
    import numpy.typing as npt
    from zarr_metadata import ZarrV3ArrayMetadataJSON

    from zarr.abc.engine import SelectionRequest
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfig
    from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar, NDBuffer
    from zarr.core.indexing import Fields
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "ZarristaAsyncEngine",
    "ZarristaAsyncHierarchyEngine",
    "ZarristaEngine",
    "ZarristaHierarchyEngine",
]


def _require_v3(metadata: ArrayMetadata) -> ZarrV3ArrayMetadataJSON:
    if metadata.zarr_format != 3:
        raise UnsupportedEngineError(
            "the zarrista engine supports Zarr v3 only; this array is "
            f"format v{metadata.zarr_format}"
        )
    # `metadata.to_dict()` is a plain `dict[str, JSON]`; zarrista's stubs type
    # `from_metadata`'s argument as the `ZarrV3ArrayMetadataJSON` TypedDict,
    # which the dict's runtime shape matches by construction.
    return cast("ZarrV3ArrayMetadataJSON", metadata.to_dict())


def _reject_unenforceable_config(config: ArrayConfig | None) -> None:
    """Reject an `ArrayConfig` the zarrista engine cannot honour.

    Most of `ArrayConfig` only affects the in-memory layout of the result,
    which this engine normalizes anyway. Two fields change *semantics*, in
    opposite directions, and per the project's fail-loud rule are refused
    rather than silently downgraded:

    - `read_missing_chunks=False` asks for a `ChunkNotFoundError` on a missing
      chunk. zarrista fills it with the fill value and cannot be made to raise.
    - `write_empty_chunks=True` asks for all-fill chunks to be written out.
      zarrista elides them and cannot be made to keep them.

    Both rejected values are the non-default, so an array using zarr's
    defaults is served.
    """
    if config is None:
        return
    if not config.read_missing_chunks:
        raise UnsupportedEngineError(
            "the zarrista engine cannot enforce read_missing_chunks=False "
            "(it fills missing chunks with the fill value instead of raising); "
            "use the default engine to enforce this setting"
        )
    if config.write_empty_chunks:
        raise UnsupportedEngineError(
            "the zarrista engine cannot enforce write_empty_chunks=True "
            "(it elides chunks that hold only the fill value); "
            "use the default engine to enforce this setting"
        )


def _as_contiguous(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Return `array` as memory zarrista will accept.

    `np.ascontiguousarray` is not enough on its own: a broadcast array (which
    is what a scalar write widens to) carries 0-strides, and NumPy still
    reports it as C-contiguous when it holds at most one element per axis. The
    buffer zarrista then sees is not the dense block it expects, so force a
    real copy whenever a stride is 0.
    """
    if any(stride == 0 for stride in array.strides):
        return np.ascontiguousarray(array.copy())
    return np.ascontiguousarray(array)


def _reject_fields(fields: Fields | None) -> None:
    # Truthiness, not `is not None`: `pop_fields` returns an empty list, not
    # `None`, when a tuple selection names no fields.
    if fields:
        raise UnsupportedEngineError(
            "the zarrista engine does not support structured-dtype field selection; "
            "use the default engine"
        )


def _finish_read(
    result: npt.NDArray[Any],
    request: SelectionRequest,
    out: NDBuffer | None,
) -> NDArrayLikeOrScalar:
    """Match zarr-python's result conventions for this selection kind."""
    # A coordinate or mask selection is flat at this boundary: zarr's own
    # `_get_selection` returns the flattened points and the caller reshapes to
    # `sel_shape`. `LazyArray` returns the un-flattened broadcast shape, whose
    # C-order ravel is the same point order.
    if request.kind in ("coordinate", "mask"):
        result = result.reshape(-1)
    if out is not None:
        # `NDArrayLike` only types slice-key indexing, so the Ellipsis fill is
        # written through an `Any`-typed local.
        out_array: Any = out.as_ndarray_like()
        out_array[...] = result
        return cast("NDArrayLikeOrScalar", out_array)
    # A rank-0 basic selection is a scalar, as in `_get_selection`.
    if request.kind == "basic" and result.shape == ():
        return cast("NDArrayLikeOrScalar", result[()])
    return result


class _ZarristaEngineBase:
    """Shared selection resolution for the sync and async zarrista engines."""

    _metadata: ArrayMetadata
    _config: ArrayConfig | None

    def _chunk_shape(self, chunk_grid: Any) -> tuple[int, ...] | None:
        """The partition size for reads: the outer chunk shape, if regular."""
        if chunk_grid.is_regular:
            return cast("tuple[int, ...]", tuple(chunk_grid.chunk_shape))
        return None

    def _read(
        self,
        source: Any,
        request: SelectionRequest,
        *,
        out: NDBuffer | None,
        fields: Fields | None,
    ) -> NDArrayLikeOrScalar:
        _reject_fields(fields)
        view = lazy_view(source, request, self._chunk_shape(request.chunk_grid))
        return _finish_read(np.asarray(view.result()), request, out)

    def _write(
        self,
        source: Any,
        arr: Any,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        fields: Fields | None,
        store_subset: Any,
        read_subset: Any,
    ) -> None:
        _reject_fields(fields)
        dtype = self._metadata.dtype.to_native_dtype()
        view = lazy_view(source, request, None)
        transform = view.transform
        # `_set_selection` accepts scalars and broadcastable values, so widen
        # to the selection's shape before laying anything out.
        values = np.broadcast_to(np.asarray(value, dtype=dtype), transform.domain.shape)

        box = dense_forward_box(transform)
        if box is not None:
            key, ndim_shape = box
            store_subset(key, values.reshape(ndim_shape))
            return

        scatter_points(
            read_subset,
            store_subset,
            transform,
            values,
            request.chunk_grid,
            request.shape,
        )


class ZarristaEngine(_ZarristaEngineBase):
    """Sync engine over `zarrista.Array`. No event loop involved."""

    def __init__(
        self, zarrista_array: Any, metadata: ArrayMetadata, config: ArrayConfig | None
    ) -> None:
        self._arr = zarrista_array
        self._metadata = metadata
        self._config = config

    def with_metadata(self, metadata: ArrayMetadata) -> ZarristaEngine:
        import zarrista

        return ZarristaEngine(
            zarrista.Array.from_metadata(_require_v3(metadata), self._arr.storage, self._arr.path),
            metadata,
            self._config,
        )

    def _source(self) -> ZarristaSource:
        return ZarristaSource(self._arr, self._metadata.dtype.to_native_dtype())

    def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        return self._read(self._source(), request, out=out, fields=fields)

    def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        arr = self._arr

        def read_subset(key: Any) -> npt.NDArray[Any]:
            # `np.array`, not `asarray`: zarrista hands back read-only
            # Rust-owned memory, and the scatter patches this in place.
            return np.array(tensor_to_numpy(arr.retrieve_array_subset(key)))

        def store_subset(key: Any, data: npt.NDArray[Any]) -> None:
            arr.store_array_subset(key, _as_contiguous(data))

        self._write(
            self._source(),
            arr,
            request,
            value,
            fields=fields,
            store_subset=store_subset,
            read_subset=read_subset,
        )


class ZarristaAsyncEngine(_ZarristaEngineBase):
    """Async engine over `zarrista.AsyncArray`.

    Store translation and construction of the underlying `zarrista.AsyncArray`
    are deferred to the first read or write rather than done at construction.
    `AsyncArray.__init__` always eagerly resolves *an* async engine — even for
    a plain sync `Array`, which never touches it, since only the sync engine is
    lazily resolved. Without this deferral, `engine="zarrista"` over a
    sync-only store (e.g. `LocalStore`) would raise `UnsupportedEngineError`
    just from opening the array, even when only ever accessed synchronously.
    """

    def __init__(
        self, store: Store, path: str, metadata: ArrayMetadata, config: ArrayConfig | None
    ) -> None:
        self._store = store
        self._path = path
        self._metadata = metadata
        self._config = config
        self._arr: Any | None = None

    def with_metadata(self, metadata: ArrayMetadata) -> ZarristaAsyncEngine:
        _require_v3(metadata)
        return ZarristaAsyncEngine(self._store, self._path, metadata, self._config)

    def _ensure_arr(self) -> Any:
        if self._arr is None:
            import zarrista

            self._arr = zarrista.AsyncArray.from_metadata(
                _require_v3(self._metadata), translate_store_async(self._store), self._path
            )
        return self._arr

    def _source(self, loop: asyncio.AbstractEventLoop) -> AsyncZarristaSource:
        return AsyncZarristaSource(self._ensure_arr(), self._metadata.dtype.to_native_dtype(), loop)

    async def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar:
        loop = asyncio.get_running_loop()
        source = self._source(loop)
        # `LazyArray` is synchronous and issues its reads back to this loop, so
        # it must not resolve on the loop thread itself.
        return await run_off_loop(lambda: self._read(source, request, out=out, fields=fields))

    async def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        arr = self._ensure_arr()
        source = self._source(loop)

        def read_subset(key: Any) -> npt.NDArray[Any]:
            return np.array(tensor_to_numpy(await_on(loop, lambda: arr.retrieve_array_subset(key))))

        def store_subset(key: Any, data: npt.NDArray[Any]) -> None:
            await_on(loop, lambda: arr.store_array_subset(key, _as_contiguous(data)))

        await run_off_loop(
            lambda: self._write(
                source,
                arr,
                request,
                value,
                fields=fields,
                store_subset=store_subset,
                read_subset=read_subset,
            )
        )


class ZarristaHierarchyEngine:
    """Store-bound factory for sync zarrista engines (translates the store once)."""

    def __init__(self, store: Store) -> None:
        self._zarr_store = store
        self._zstore = translate_store_sync(store)

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> ZarristaEngine:
        """Mint a sync array engine bound to `path`/`metadata`."""
        import zarrista

        _reject_unenforceable_config(config)
        return ZarristaEngine(
            zarrista.Array.from_metadata(
                _require_v3(metadata), self._zstore, "/" + path.strip("/")
            ),
            metadata,
            config,
        )


class ZarristaAsyncHierarchyEngine:
    """Store-bound factory for async zarrista engines.

    Unlike `ZarristaHierarchyEngine`, this does *not* translate the store at
    construction time — see `ZarristaAsyncEngine` for why.
    """

    def __init__(self, store: Store) -> None:
        self._zarr_store = store

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> ZarristaAsyncEngine:
        """Mint an async array engine bound to `path`/`metadata`.

        `metadata` is validated as Zarr v3 eagerly (cheap, and lets an
        unsupported-format error surface immediately); the store itself is only
        translated lazily, on first I/O.
        """
        _reject_unenforceable_config(config)
        _require_v3(metadata)
        return ZarristaAsyncEngine(self._zarr_store, "/" + path.strip("/"), metadata, config)
