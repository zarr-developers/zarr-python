"""Zarrista-backed array engines."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from zarr.errors import UnsupportedEngineError
from zarr.zarrista._translate import translate_store_async, translate_store_sync

if TYPE_CHECKING:
    from zarr_metadata import ArrayMetadataV3

    from zarr.abc.engine import Region
    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfig
    from zarr.core.buffer import BufferPrototype, NDBuffer
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "ZarristaAsyncEngine",
    "ZarristaAsyncHierarchyEngine",
    "ZarristaEngine",
    "ZarristaHierarchyEngine",
]


def _require_v3(metadata: ArrayMetadata) -> ArrayMetadataV3:
    if metadata.zarr_format != 3:
        raise UnsupportedEngineError(
            "the zarrista engine supports Zarr v3 only; this array is "
            f"format v{metadata.zarr_format}"
        )
    # `metadata.to_dict()` is a plain `dict[str, JSON]`; zarrista's stubs type
    # `from_metadata`'s argument as the `ArrayMetadataV3` TypedDict, which the
    # dict's runtime shape matches by construction.
    return cast("ArrayMetadataV3", metadata.to_dict())


def _reject_unenforceable_config(config: ArrayConfig | None) -> None:
    """Reject an `ArrayConfig` the zarrista engine cannot honour.

    The zarrista engine owns its own codec options and does not consult
    zarr-python's `ArrayConfig`. Most fields (e.g. `order`) only affect the
    in-memory layout of the returned array, which the facade normalizes, so
    ignoring them is safe. `read_missing_chunks=False`, however, changes
    *semantics*: the default engine raises `ChunkNotFoundError` for a missing
    chunk, whereas zarrista silently fills it with the fill value. Per the
    project's fail-loud rule, refuse rather than silently downgrade to
    fill-value reads.
    """
    if config is not None and not config.read_missing_chunks:
        raise UnsupportedEngineError(
            "the zarrista engine cannot enforce read_missing_chunks=False "
            "(it fills missing chunks with the fill value instead of raising); "
            "use the default engine to enforce this setting"
        )


def _region_to_selection(region: Region) -> tuple[slice, ...]:
    return tuple(slice(s, e) for s, e in zip(region.start, region.end_exclusive, strict=True))


def _decoded_to_numpy(decoded: Any) -> np.ndarray[Any, Any]:
    # Only `Tensor` (fixed-width, unmasked) implements `__array__`/the buffer
    # protocol in a way that produces a correct zarr-python array. `VariableArray`
    # (vlen data) only exposes the Arrow C Data interface, so `np.asarray` on it
    # does *not* raise -- it silently wraps the opaque object in a 0-d `object`
    # array -- which is why this is a type-name check rather than a try/except
    # around `np.asarray`. `MaskedTensor`/`MaskedVariableArray` do convert (the
    # former to a `numpy.ma.MaskedArray`), but masked layouts have no
    # zarr-python equivalent to hold that result.
    type_name = type(decoded).__name__
    if type_name != "Tensor":
        raise NotImplementedError(
            f"zarrista returned a {type_name}; only fixed-width, unmasked reads "
            "(zarrista `Tensor`) are supported by this engine. vlen reads "
            "(`VariableArray`) are pending zarrista numpy export; masked layouts "
            "(`MaskedTensor`/`MaskedVariableArray`) have no zarr-python equivalent."
        )
    return np.asarray(decoded)


def _chunks_overlapping(
    region: Region, chunk_shape: tuple[int, ...]
) -> itertools.product[tuple[int, ...]]:
    ranges = [
        range(s // c, (e + c - 1) // c) if e > s else range(0)
        for s, e, c in zip(region.start, region.end_exclusive, chunk_shape, strict=True)
    ]
    return itertools.product(*ranges)


class ZarristaEngine:
    """Sync engine over `zarrista.Array`. No event loop involved."""

    def __init__(self, zarrista_array: Any) -> None:
        self._arr = zarrista_array

    def with_metadata(self, metadata: ArrayMetadata) -> ZarristaEngine:
        import zarrista

        return ZarristaEngine(
            zarrista.Array.from_metadata(_require_v3(metadata), self._arr.store, self._arr.path)
        )

    def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> Any:
        return _decoded_to_numpy(self._arr.retrieve_array_subset(_region_to_selection(selection)))

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        import zarrista

        value_np = np.ascontiguousarray(value.as_ndarray_like())
        # regular grids: the origin chunk has the full (unclipped) chunk shape
        chunk_shape = tuple(self._arr.chunk_shape([0] * len(selection.start)))
        for chunk_idx_tuple in _chunks_overlapping(selection, chunk_shape):
            chunk_idx = list(chunk_idx_tuple)
            chunk_slices = self._arr.chunk_subset(chunk_idx)
            # overlap of the write region with this chunk, in array coords
            lo = tuple(
                max(cs.start, s) for cs, s in zip(chunk_slices, selection.start, strict=True)
            )
            hi = tuple(
                min(cs.stop, e) for cs, e in zip(chunk_slices, selection.end_exclusive, strict=True)
            )
            in_value = tuple(
                slice(a - s, b - s) for a, b, s in zip(lo, hi, selection.start, strict=True)
            )
            in_chunk = tuple(
                slice(a - cs.start, b - cs.start)
                for a, b, cs in zip(lo, hi, chunk_slices, strict=True)
            )
            full_chunk = all(
                sl.start == 0 and sl.stop == (cs.stop - cs.start) and (cs.stop - cs.start) == c
                for sl, cs, c in zip(in_chunk, chunk_slices, chunk_shape, strict=True)
            )
            if full_chunk:
                chunk_np = np.ascontiguousarray(value_np[in_value])
            else:
                chunk_np = np.array(_decoded_to_numpy(self._arr.retrieve_chunk(chunk_idx)))
                chunk_np[in_chunk] = value_np[in_value]
                chunk_np = np.ascontiguousarray(chunk_np)
            self._arr.store_chunk(
                chunk_idx, zarrista.ArrayBytes(chunk_np.reshape(-1).view(np.uint8))
            )


class ZarristaHierarchyEngine:
    """Store-bound factory for sync zarrista engines (translates the store once)."""

    def __init__(self, store: Store) -> None:
        self._zarr_store = store
        self._zstore = translate_store_sync(store)

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> ZarristaEngine:
        """Mint a sync array engine bound to `path`/`metadata`.

        `config` is accepted for protocol conformance with `HierarchyEngine`
        and is otherwise unused -- zarrista owns its own codec options and does
        not read zarr-python's `ArrayConfig` (e.g. `order`) -- with one
        exception: `config.read_missing_chunks=False` is rejected with an
        `UnsupportedEngineError`, since zarrista cannot enforce it and would
        silently fill missing chunks instead of raising.
        """
        import zarrista

        _reject_unenforceable_config(config)
        return ZarristaEngine(
            zarrista.Array.from_metadata(_require_v3(metadata), self._zstore, "/" + path.strip("/"))
        )


class ZarristaAsyncEngine:
    """Async engine over `zarrista.AsyncArray`.

    Store translation, and construction of the underlying `zarrista.AsyncArray`,
    are deferred to the first `read_selection`/`write_selection` call rather
    than done at construction time. `AsyncArray.__init__` always eagerly
    resolves *an* async engine -- even for a plain sync `Array` that never
    touches it, since only the sync engine is lazily resolved there (see
    `Array.engine`). Without this deferral, `engine="zarrista"` over a
    sync-only store (e.g. `LocalStore`) would raise `UnsupportedEngineError`
    just from opening the array, even when only ever accessed synchronously.
    """

    def __init__(self, store: Store, path: str, metadata: ArrayMetadata) -> None:
        self._store = store
        self._path = path
        self._metadata = metadata
        self._arr: Any | None = None

    def with_metadata(self, metadata: ArrayMetadata) -> ZarristaAsyncEngine:
        _require_v3(metadata)
        return ZarristaAsyncEngine(self._store, self._path, metadata)

    async def _ensure_arr(self) -> Any:
        if self._arr is None:
            import zarrista

            meta_dict = _require_v3(self._metadata)
            zstore = translate_store_async(self._store)
            self._arr = zarrista.AsyncArray.from_metadata(meta_dict, zstore, self._path)
        return self._arr

    async def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> Any:
        arr = await self._ensure_arr()
        return _decoded_to_numpy(await arr.retrieve_array_subset(_region_to_selection(selection)))

    async def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        import zarrista

        arr = await self._ensure_arr()
        value_np = np.ascontiguousarray(value.as_ndarray_like())
        # regular grids: the origin chunk has the full (unclipped) chunk shape
        chunk_shape = tuple(arr.chunk_shape([0] * len(selection.start)))
        for chunk_idx_tuple in _chunks_overlapping(selection, chunk_shape):
            chunk_idx = list(chunk_idx_tuple)
            chunk_slices = arr.chunk_subset(chunk_idx)
            lo = tuple(
                max(cs.start, s) for cs, s in zip(chunk_slices, selection.start, strict=True)
            )
            hi = tuple(
                min(cs.stop, e) for cs, e in zip(chunk_slices, selection.end_exclusive, strict=True)
            )
            in_value = tuple(
                slice(a - s, b - s) for a, b, s in zip(lo, hi, selection.start, strict=True)
            )
            in_chunk = tuple(
                slice(a - cs.start, b - cs.start)
                for a, b, cs in zip(lo, hi, chunk_slices, strict=True)
            )
            full_chunk = all(
                sl.start == 0 and sl.stop == (cs.stop - cs.start) and (cs.stop - cs.start) == c
                for sl, cs, c in zip(in_chunk, chunk_slices, chunk_shape, strict=True)
            )
            if full_chunk:
                chunk_np = np.ascontiguousarray(value_np[in_value])
            else:
                chunk_np = np.array(_decoded_to_numpy(await arr.retrieve_chunk(chunk_idx)))
                chunk_np[in_chunk] = value_np[in_value]
                chunk_np = np.ascontiguousarray(chunk_np)
            await arr.store_chunk(
                chunk_idx, zarrista.ArrayBytes(chunk_np.reshape(-1).view(np.uint8))
            )


class ZarristaAsyncHierarchyEngine:
    """Store-bound factory for async zarrista engines.

    Unlike `ZarristaHierarchyEngine`, this does *not* translate the store at
    construction time -- see `ZarristaAsyncEngine` for why.
    """

    def __init__(self, store: Store) -> None:
        self._zarr_store = store

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> ZarristaAsyncEngine:
        """Mint an async array engine bound to `path`/`metadata`.

        `config` is accepted for protocol conformance with
        `AsyncHierarchyEngine` and is otherwise unused -- zarrista owns its own
        codec options and does not read zarr-python's `ArrayConfig` (e.g.
        `order`) -- with one exception: `config.read_missing_chunks=False` is
        rejected with an `UnsupportedEngineError`, since zarrista cannot enforce
        it and would silently fill missing chunks instead of raising. `metadata`
        is validated as Zarr v3 eagerly (cheap, and lets an unsupported-format
        error surface immediately); the store itself is only translated lazily,
        on first I/O.
        """
        _reject_unenforceable_config(config)
        _require_v3(metadata)
        return ZarristaAsyncEngine(self._zarr_store, "/" + path.strip("/"), metadata)
