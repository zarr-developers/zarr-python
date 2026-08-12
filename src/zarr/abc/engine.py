"""Array engine protocols.

An *array engine* owns the data path of one open array: reading and writing
decoded data for a selection. `AsyncArray` wraps an object satisfying
`AsyncArrayEngine`; `Array` wraps an object satisfying `ArrayEngine`. A
*hierarchy engine* is bound to a store and mints array engines that share
resources.

What crosses the boundary is a `SelectionRequest`: the user's selection,
unresolved, tagged with the dialect it is written in. Both descriptions an
engine might want are derived from it, losslessly:

- `SelectionRequest.indexer` is zarr-python's native `Indexer`, a chunk-gather
  plan. The built-in engine consumes this and so drives the existing codec
  pipeline with no round-trip.
- The raw `selection` and `kind` are what a backend that reads *boxes* rather
  than chunks (an FFI binding, an HTTP range endpoint) needs, typically to
  drive a `zarr_indexing.LazyArray`.

The request deliberately carries the *unresolved* selection. An `Indexer` has
already thrown away what a box-reading backend needs — `CoordinateIndexer`
stores its coordinates in chunk-sorted order, not the user's — so an engine
that reconstructed a selection from an indexer would silently reorder results.
Deriving both views from the same raw selection avoids that entirely, and
keeps `zarr_indexing` out of this module: only a backend that wants a
transform imports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy.typing as npt

    from zarr.core.array_spec import ArrayConfig
    from zarr.core.buffer import BufferPrototype, NDArrayLikeOrScalar, NDBuffer
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.indexing import Fields, Indexer
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "ArrayEngine",
    "AsyncArrayEngine",
    "AsyncHierarchyEngine",
    "HierarchyEngine",
    "SelectionKind",
    "SelectionRequest",
]

SelectionKind = Literal["basic", "orthogonal", "coordinate", "mask", "block"]
"""Which indexing dialect a `SelectionRequest.selection` is written in."""


@dataclass(frozen=True)
class SelectionRequest:
    """One read or write request: a selection, its dialect, and its context.

    See the module docstring for why this carries the *unresolved* selection.
    """

    kind: SelectionKind
    selection: Any
    shape: tuple[int, ...]
    chunk_grid: ChunkGrid

    @cached_property
    def indexer(self) -> Indexer:
        """This request as a zarr-python `Indexer`.

        Resolved on first access and then cached, so an engine that never asks
        for it does not pay to build it.
        """
        from zarr.core.indexing import (
            BasicIndexer,
            BlockIndexer,
            CoordinateIndexer,
            MaskIndexer,
            OrthogonalIndexer,
        )

        indexer_for: dict[str, Any] = {
            "basic": BasicIndexer,
            "orthogonal": OrthogonalIndexer,
            "coordinate": CoordinateIndexer,
            "mask": MaskIndexer,
            "block": BlockIndexer,
        }
        return indexer_for[self.kind](self.selection, self.shape, self.chunk_grid)  # type: ignore[no-any-return]


@runtime_checkable
class AsyncArrayEngine(Protocol):
    """The asynchronous data path of one open array.

    Bound to `(store, path, metadata, config)` at construction. The signatures
    mirror zarr-python's own `_get_selection` / `_set_selection` — including
    the scalar return for a rank-0 basic selection — so that the built-in
    engine is a pure delegation.

    Note: `runtime_checkable` isinstance checks only verify method names;
    mypy is the authoritative conformance check.
    """

    async def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar: ...

    async def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None: ...

    def with_metadata(self, metadata: ArrayMetadata) -> AsyncArrayEngine: ...


@runtime_checkable
class ArrayEngine(Protocol):
    """The synchronous data path of one open array.

    Methods must not require a running event loop. See `AsyncArrayEngine`.
    """

    def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> NDArrayLikeOrScalar: ...

    def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None: ...

    def with_metadata(self, metadata: ArrayMetadata) -> ArrayEngine: ...


@runtime_checkable
class AsyncHierarchyEngine(Protocol):
    """A store-bound factory for asynchronous array engines."""

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> AsyncArrayEngine: ...


@runtime_checkable
class HierarchyEngine(Protocol):
    """A store-bound factory for synchronous array engines."""

    def array_engine(
        self, path: str, metadata: ArrayMetadata, config: ArrayConfig | None = None
    ) -> ArrayEngine: ...
