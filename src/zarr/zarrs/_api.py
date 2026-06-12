from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    """Create a group at `path`, deleting any existing node (and its children) first."""
    with _translate_errors():
        await asyncio.to_thread(
            _zb.create_group, resolve_store(store), _node_path(path), json.dumps(metadata), True
        )
