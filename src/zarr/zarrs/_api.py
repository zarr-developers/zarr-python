from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import _zarrs_bindings as _zb
import numpy as np

from zarr.errors import NodeNotFoundError
from zarr.zarrs._bridge import resolve_store

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import numpy.typing as npt

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


async def delete_node(
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Delete the node at `path`, including all keys and child nodes under it.

    Raises `zarr.errors.NodeNotFoundError` if no node exists there. Deleting
    the root node (`path=""`) clears the entire store.
    """
    with _translate_errors():
        await asyncio.to_thread(_zb.delete_node, resolve_store(store), _node_path(path))


async def list_children(
    store: Store,
    path: str,
    *,
    options: ZarrsOptions | None = None,
) -> list[tuple[str, dict[str, JSON]]]:
    """List the direct children of the group at `path` as
    `(path, metadata_document)` pairs. Paths are store-relative (no leading
    `/`).

    Raises `zarr.errors.NodeNotFoundError` if no *group* exists at `path` --
    including when `path` holds an array.
    """
    with _translate_errors():
        raw: list[tuple[str, str]] = await asyncio.to_thread(
            _zb.list_children, resolve_store(store), _node_path(path)
        )
    return [
        (child_path.lstrip("/"), cast("dict[str, JSON]", json.loads(doc)))
        for child_path, doc in raw
    ]


def _chunk_dtype_and_shape(
    metadata: Mapping[str, JSON],
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Resolve the numpy dtype and chunk shape from a metadata document, using
    zarr-python's own metadata parsing.

    The dtype is coerced to native byte order: zarrs always decodes to (and
    encodes from) the native in-memory representation, applying any byte-order
    codec itself.
    """
    from zarr.core.metadata.v2 import ArrayV2Metadata
    from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata

    if metadata.get("zarr_format") == 3:
        meta3 = ArrayV3Metadata.from_dict(dict(metadata))
        grid = meta3.chunk_grid
        if not isinstance(grid, RegularChunkGridMetadata):
            raise NotImplementedError("only regular chunk grids are supported")
        return meta3.data_type.to_native_dtype().newbyteorder("="), grid.chunk_shape
    meta2 = ArrayV2Metadata.from_dict(dict(metadata))
    return meta2.dtype.to_native_dtype().newbyteorder("="), meta2.chunks


async def decode_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    selection: tuple[slice | int, ...] | None = None,
    options: ZarrsOptions | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Read and decode the chunk at `chunk_coords` of the array described by
    `metadata`, located at `path` in `store`.

    The metadata document is authoritative: it is not read from the store.
    Missing chunks decode to the fill value. `selection` (a chunk-relative
    subset) is not implemented yet.

    The returned array is a read-only, zero-copy view over the decoded bytes;
    call `.copy()` if you need a writable array.
    """
    if selection is not None:
        raise NotImplementedError("chunk subset selection is not implemented yet")
    raw = await asyncio.to_thread(
        _zb.retrieve_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
    )
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)


async def read_encoded_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: ZarrsOptions | None = None,
) -> bytes | None:
    """Read the raw, still-encoded bytes of the chunk at `chunk_coords`, or
    `None` if the chunk does not exist. No codecs are applied."""
    result: bytes | None = await asyncio.to_thread(
        _zb.retrieve_encoded_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
    )
    return result


async def encode_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    value: npt.ArrayLike,
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Encode `value` with the codecs in `metadata` and store it as the chunk
    at `chunk_coords`. `value` must match the chunk shape exactly."""
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if arr.shape != chunk_shape:
        raise ValueError(f"value shape {arr.shape} does not match chunk shape {chunk_shape}")
    await asyncio.to_thread(
        _zb.store_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
        arr.tobytes(),
    )


async def erase_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: ZarrsOptions | None = None,
) -> None:
    """Delete the chunk at `chunk_coords`. Deleting a missing chunk is a no-op."""
    await asyncio.to_thread(
        _zb.erase_chunk,
        resolve_store(store),
        _node_path(path),
        json.dumps(metadata),
        list(chunk_coords),
    )
