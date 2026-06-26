from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

import _zarrs_bindings as _zb

from zarr.crud import NodeExistsError
from zarr.errors import NodeNotFoundError
from zarr.zarrs._bridge import resolve_store

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.common import JSON


def _node_path(path: str) -> str:
    """Convert a zarr path (`""`, `"foo/bar"`) to a zarrs node path (`"/"`,
    `"/foo/bar"`)."""
    return f"/{path.strip('/')}"


@contextmanager
def _translate_errors() -> Iterator[None]:
    try:
        yield
    except _zb.NodeNotFoundError as err:
        raise NodeNotFoundError(str(err)) from err
    except _zb.NodeExistsError as err:
        raise NodeExistsError(str(err)) from err


class ZarrsBackend:
    """CRUD backend backed by the Rust `zarrs` crate via `_zarrs_bindings`.

    Owns the zarrs-specific plumbing: JSON-serializing the metadata document,
    the `/`-prefixed node-path form, store resolution, offloading the blocking
    Rust calls to a worker thread, and translating binding exceptions to the
    canonical `zarr.crud` / `zarr.errors` types.

    Known limitation: creating a Zarr v2 *group* with attributes writes a
    non-standard `.zattrs` (the attributes nested under an ``"attributes"`` key)
    that zarr-python and other readers interpret incorrectly. This is a
    zarrs-crate behavior; the pure-Python reference backend writes the standard
    layout. Prefer the reference backend for writing v2 groups until the zarrs
    crate is fixed.
    """

    async def create_array(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        with _translate_errors():
            await asyncio.to_thread(
                _zb.create_array,
                resolve_store(store),
                _node_path(path),
                json.dumps(metadata),
                overwrite,
            )

    async def create_group(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        with _translate_errors():
            await asyncio.to_thread(
                _zb.create_group,
                resolve_store(store),
                _node_path(path),
                json.dumps(metadata),
                overwrite,
            )

    async def read_metadata(self, store: Store, path: str) -> dict[str, JSON]:
        with _translate_errors():
            raw = await asyncio.to_thread(_zb.read_metadata, resolve_store(store), _node_path(path))
        return cast("dict[str, JSON]", json.loads(raw))

    async def read_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> bytes:
        return await asyncio.to_thread(
            _zb.retrieve_chunk,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(coords),
        )

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes:
        return await asyncio.to_thread(
            _zb.retrieve_array_subset,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(start),
            list(shape),
        )

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None:
        await asyncio.to_thread(
            _zb.store_chunk,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(coords),
            data,
        )

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None:
        await asyncio.to_thread(
            _zb.erase_chunk,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            list(coords),
        )

    async def delete_node(self, store: Store, path: str) -> None:
        with _translate_errors():
            await asyncio.to_thread(_zb.delete_node, resolve_store(store), _node_path(path))

    async def list_children(self, store: Store, path: str) -> list[tuple[str, dict[str, JSON]]]:
        with _translate_errors():
            raw: list[tuple[str, str]] = await asyncio.to_thread(
                _zb.list_children, resolve_store(store), _node_path(path)
            )
        return [
            (child_path.lstrip("/"), cast("dict[str, JSON]", json.loads(doc)))
            for child_path, doc in raw
        ]
