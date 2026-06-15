from __future__ import annotations

import asyncio
import json
import operator
import types
from collections.abc import Sequence
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


BasicIndex = int | slice | types.EllipsisType
BasicSelection = BasicIndex | tuple[BasicIndex, ...]


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


def _array_shape(metadata: Mapping[str, JSON]) -> tuple[int, ...]:
    """Resolve the array shape from a metadata document."""
    shape = metadata.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, str):
        raise TypeError("metadata document has no valid 'shape'")
    result: list[int] = []
    for s in shape:
        if not isinstance(s, (int, float)):
            raise TypeError(f"shape element {s!r} is not a number")
        if isinstance(s, float) and not s.is_integer():
            raise TypeError(f"shape element {s!r} is not an integer")
        result.append(int(s))
    return tuple(result)


def _normalize_selection(
    selection: BasicSelection, shape: tuple[int, ...]
) -> tuple[list[int], list[int], tuple[slice | int, ...]]:
    """Normalize a numpy-style basic-indexing selection against `shape`.

    Returns `(start, bounding_shape, post_index)`: the step-1 bounding box to
    fetch (per-dimension start and length), and the numpy index to apply to
    the fetched block to produce the final result (strides, reversals, and
    integer-axis removal). Only integers, slices, and `Ellipsis` are
    supported; fancy indexing raises `TypeError`.
    """
    sel_tuple = selection if isinstance(selection, tuple) else (selection,)

    n_ellipsis = sum(1 for s in sel_tuple if s is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if n_ellipsis == 1:
        i = sel_tuple.index(Ellipsis)
        n_fill = len(shape) - (len(sel_tuple) - 1)
        if n_fill < 0:
            raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional")
        sel_tuple = sel_tuple[:i] + (slice(None),) * n_fill + sel_tuple[i + 1 :]
    if len(sel_tuple) > len(shape):
        raise IndexError(f"too many indices for array: array is {len(shape)}-dimensional")
    sel_tuple = sel_tuple + (slice(None),) * (len(shape) - len(sel_tuple))

    starts: list[int] = []
    lengths: list[int] = []
    post: list[slice | int] = []
    for dim, (sel, size) in enumerate(zip(sel_tuple, shape, strict=True)):
        if isinstance(sel, slice):
            start, stop, step = sel.indices(size)
            n = len(range(start, stop, step))
            if n == 0:
                starts.append(0)
                lengths.append(0)
                post.append(slice(None))
            elif step > 0:
                last = start + (n - 1) * step
                starts.append(start)
                lengths.append(last - start + 1)
                post.append(slice(None, None, step))
            else:
                # descending: bounding box is [last, start], ascending in store
                # order; slice(None, None, step) over the block starts at its
                # final element (global `start`) and lands exactly on index 0
                # (global `last`) because the block length is (n-1)*|step| + 1.
                last = start + (n - 1) * step
                starts.append(last)
                lengths.append(start - last + 1)
                post.append(slice(None, None, step))
        else:
            assert not isinstance(sel, types.EllipsisType), "Ellipsis already expanded above"
            try:
                idx = operator.index(sel)
            except TypeError:
                raise TypeError(
                    "unsupported selection element "
                    f"{sel!r}: only integers, slices, and Ellipsis are supported"
                ) from None
            if idx < 0:
                idx += size
            if not 0 <= idx < size:
                raise IndexError(f"index {sel} is out of bounds for axis {dim} with size {size}")
            starts.append(idx)
            lengths.append(1)
            post.append(0)
    return starts, lengths, tuple(post)


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


async def decode_region(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    selection: BasicSelection,
    *,
    options: ZarrsOptions | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Read and decode the region of the array described by `metadata` given
    by a numpy-style basic-indexing `selection` (integers, slices including
    steps, `Ellipsis`).

    The metadata document is authoritative: it is not read from the store.
    One zarrs call fetches the step-1 bounding box of the selection (decoding
    all overlapping chunks, in parallel for multi-chunk regions); strides,
    reversals, and integer-axis removal are applied as numpy views on the
    result. Missing chunks decode to the fill value. Fancy indexing (integer
    or boolean arrays) is not supported and raises `TypeError`. The returned
    array is a read-only view; call `.copy()` if you need a writable array.

    Note: zarrs fetches the step-1 bounding box of the selection. A selection
    like `slice(0, N, step)` reads `O(N)` bytes from the store even though only
    `O(N / step)` are returned; for sparse selections over large arrays, prefer
    reading per-chunk with `decode_chunk`.
    """
    dtype, _ = _chunk_dtype_and_shape(metadata)
    shape = _array_shape(metadata)
    starts, lengths, post_index = _normalize_selection(selection, shape)
    if 0 in lengths:
        block = np.empty(lengths, dtype=dtype)
        block.flags.writeable = False
    else:
        raw = await asyncio.to_thread(
            _zb.retrieve_array_subset,
            resolve_store(store),
            _node_path(path),
            json.dumps(metadata),
            starts,
            lengths,
        )
        block = np.frombuffer(raw, dtype=dtype).reshape(lengths)
    result: np.ndarray[Any, np.dtype[Any]] = block[post_index]
    return result
