from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import _zarrs_bindings as _zb

from zarr.errors import NodeNotFoundError
from zarr.zarrs._bridge import resolve_store

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from zarr.abc.store import Store
    from zarr.core.common import JSON

NodeExistsError = _zb.NodeExistsError
"""Raised by `create_new_*` when a node already exists at the target path."""


@dataclass(frozen=True, slots=True)
class ZarrsOptions:
    """Options for zarrs-backed operations.

    Currently empty: fields (concurrency limits, checksum validation) arrive in
    a later phase. Accepting it now keeps signatures stable.
    """


def _node_path(path: str) -> str:
    """Convert a zarr-python node path (`""`, `"foo/bar"`) to a zarrs node path
    (`"/"`, `"/foo/bar"`)."""
    return f"/{path.strip('/')}"


@contextmanager
def _translate_errors() -> Iterator[None]:
    try:
        yield
    except _zb.NodeNotFoundError as err:
        raise NodeNotFoundError(str(err)) from err


async def create_new_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create a group at `path` from a group metadata document.

    Raises `NodeExistsError` if any node already exists at `path`.
    Creation is not atomic with respect to concurrent writers: a concurrent
    creation at the same path can race the existence check.
    """
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_group, resolve_store(store), _node_path(path), json.dumps(metadata), False
        )


async def create_overwrite_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create a group at `path`, deleting any existing node (and its children) first.

    Creation is not atomic with respect to concurrent writers: a concurrent
    creation at the same path can race the existence check.
    """
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_group, resolve_store(store), _node_path(path), json.dumps(metadata), True
        )


async def create_new_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create an array at `path` from a v2 or v3 array metadata document.

    Raises `NodeExistsError` if any node already exists at `path`. Creation is
    not atomic with respect to concurrent writers: a concurrent creation at the
    same path can race the existence check.
    """
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_array, resolve_store(store), _node_path(path), json.dumps(metadata), False
        )


async def create_overwrite_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Create an array at `path`, deleting any existing node (and its children)
    first. The delete-then-create sequence is not atomic with respect to
    concurrent writers.
    """
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_array, resolve_store(store), _node_path(path), json.dumps(metadata), True
        )


async def read_metadata(
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> dict[str, JSON]:
    """Read the metadata document of the array or group at `path`.

    Raises `zarr.errors.NodeNotFoundError` if no node exists there.
    """
    with _translate_errors():
        raw = await asyncio.to_thread(_zb.read_metadata, resolve_store(store), _node_path(path))
    return cast("dict[str, JSON]", json.loads(raw))
