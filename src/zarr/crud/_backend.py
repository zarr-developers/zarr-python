from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.common import JSON


class NodeExistsError(ValueError):
    """Raised when a node already exists at a path and overwrite was not requested."""


@runtime_checkable
class CrudBackend(Protocol):
    """The byte/metadata-level contract a CRUD backend must implement.

    Methods take neutral types: the metadata document as a `dict`, a zarr
    `Store`, and plain zarr paths (`""`, `"foo/bar"`). They return raw bytes,
    parsed JSON documents, or `None`. The shared `zarr.crud` facade builds the
    numpy- and selection-level API on top of these.

    `create_*` raise `zarr.crud.NodeExistsError` when a node exists and
    `overwrite` is false. `read_metadata`/`delete_node`/`list_children` raise
    `zarr.errors.NodeNotFoundError` when the target is missing.

    Note: because this protocol is `runtime_checkable`, `isinstance` checks only
    verify that the method names exist, not their signatures or that they are
    async. Static type checking (mypy) is the authoritative conformance check.
    """

    async def create_array(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None: ...

    async def create_group(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None: ...

    async def read_metadata(self, store: Store, path: str) -> dict[str, JSON]: ...

    async def read_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> bytes: ...

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes: ...

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None: ...

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None: ...

    async def delete_node(self, store: Store, path: str) -> None: ...

    async def list_children(self, store: Store, path: str) -> list[tuple[str, dict[str, JSON]]]: ...
