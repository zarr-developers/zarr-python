"""Array engine protocols.

An *array engine* owns the data path of one open array: reading and writing
decoded data for contiguous regions. `Array` wraps an object satisfying
`ArrayEngine`; `AsyncArray` wraps an object satisfying `AsyncArrayEngine`.
A *hierarchy engine* is bound to a store and mints array engines that share
resources. See `docs/superpowers/specs/2026-07-22-array-engine-protocol-design.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from zarr.core.buffer import BufferPrototype, NDArrayLike, NDBuffer
    from zarr.core.metadata import ArrayMetadata

__all__ = [
    "ArrayEngine",
    "AsyncArrayEngine",
    "AsyncHierarchyEngine",
    "HierarchyEngine",
    "Region",
]


class Region(NamedTuple):
    """A contiguous, step-1 box in array-element coordinates.

    One entry per dimension; `start` is inclusive, `end_exclusive` exclusive.
    Callers pass normalized values: non-negative and clipped to the array
    shape. This is the only selection type that crosses the engine boundary.
    """

    start: tuple[int, ...]
    end_exclusive: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """The ndim-preserving shape of the box."""
        return tuple(e - s for s, e in zip(self.start, self.end_exclusive, strict=True))


@runtime_checkable
class ArrayEngine(Protocol):
    """The synchronous data path of one open array.

    Bound to `(store, path, metadata)` at construction. Methods must not
    require a running event loop. Read results are ndim-preserving with
    shape `selection.shape` and need only implement `__array__`.

    Note: `runtime_checkable` isinstance checks only verify method names;
    mypy is the authoritative conformance check.
    """

    def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> NDArrayLike: ...

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None: ...

    def with_metadata(self, metadata: ArrayMetadata) -> ArrayEngine: ...


@runtime_checkable
class AsyncArrayEngine(Protocol):
    """The asynchronous data path of one open array. See `ArrayEngine`."""

    async def read_selection(
        self, selection: Region, *, prototype: BufferPrototype
    ) -> NDArrayLike: ...

    async def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None: ...

    def with_metadata(self, metadata: ArrayMetadata) -> AsyncArrayEngine: ...


@runtime_checkable
class HierarchyEngine(Protocol):
    """A store-bound factory for synchronous array engines."""

    def array_engine(self, path: str, metadata: ArrayMetadata) -> ArrayEngine: ...


@runtime_checkable
class AsyncHierarchyEngine(Protocol):
    """A store-bound factory for asynchronous array engines."""

    def array_engine(self, path: str, metadata: ArrayMetadata) -> AsyncArrayEngine: ...
