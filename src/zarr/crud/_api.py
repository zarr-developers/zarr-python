from __future__ import annotations

import operator
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from zarr.core.buffer.core import default_buffer_prototype
from zarr.crud._common import parse_array_metadata
from zarr.crud._registry import get_backend

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from zarr.abc.store import Store
    from zarr.core.common import JSON
    from zarr.crud._backend import CrudBackend


@dataclass(frozen=True, slots=True)
class CrudOptions:
    """Options for CRUD operations.

    Currently empty: fields (concurrency limits, checksum validation) arrive in
    a later phase. Accepting it now keeps signatures stable.
    """


BasicIndex = int | slice | types.EllipsisType
BasicSelection = BasicIndex | tuple[BasicIndex, ...]


def _resolve_backend(backend: CrudBackend | str | None) -> CrudBackend:
    if backend is None or isinstance(backend, str):
        return get_backend(backend)
    return backend


def _chunk_dtype_and_shape(
    metadata: Mapping[str, JSON],
) -> tuple[np.dtype[Any], tuple[int, ...]]:
    """Resolve native-byte-order numpy dtype and regular chunk shape.

    Backends decode to (and encode from) the native in-memory representation,
    applying any byte-order codec themselves, so the dtype is coerced to native.
    """
    from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata

    meta_obj = parse_array_metadata(metadata)
    if isinstance(meta_obj, ArrayV3Metadata):
        grid = meta_obj.chunk_grid
        if not isinstance(grid, RegularChunkGridMetadata):
            raise NotImplementedError("only regular chunk grids are supported")
        chunk_shape = tuple(grid.chunk_shape)
    else:
        chunk_shape = tuple(meta_obj.chunks)
    return meta_obj.dtype.to_native_dtype().newbyteorder("="), chunk_shape


def _array_shape(metadata: Mapping[str, JSON]) -> tuple[int, ...]:
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


def _chunk_key(metadata: Mapping[str, JSON], path: str, coords: tuple[int, ...]) -> str:
    meta_obj = parse_array_metadata(metadata)
    rel = meta_obj.encode_chunk_key(coords)
    p = path.strip("/")
    return f"{p}/{rel}" if p else rel


def _normalize_selection(
    selection: BasicSelection, shape: tuple[int, ...]
) -> tuple[list[int], list[int], tuple[slice | int, ...]]:
    """Normalize a numpy basic-indexing selection to a step-1 bounding box.

    Returns `(start, bounding_shape, post_index)`: the box to fetch and the
    numpy index to apply to it (strides, reversals, integer-axis removal). Only
    integers, slices, and `Ellipsis` are supported; fancy indexing raises.
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


# --- node lifecycle ---


async def create_new_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Create a group from a group metadata document. Raises `NodeExistsError`
    if a node already exists at `path`. Not atomic against concurrent writers."""
    await _resolve_backend(engine).create_group(store, path, metadata, overwrite=False)


async def create_overwrite_group(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Create a group, deleting any existing node (and children) first. Not
    atomic against concurrent writers."""
    await _resolve_backend(engine).create_group(store, path, metadata, overwrite=True)


async def create_new_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Create an array from a v2 or v3 metadata document. Raises
    `NodeExistsError` if a node already exists. Not atomic against concurrent
    writers."""
    await _resolve_backend(engine).create_array(store, path, metadata, overwrite=False)


async def create_overwrite_array(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Create an array, deleting any existing node (and children) first. Not
    atomic against concurrent writers."""
    await _resolve_backend(engine).create_array(store, path, metadata, overwrite=True)


async def read_metadata(
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> dict[str, JSON]:
    """Read the metadata document of the array or group at `path`. Raises
    `zarr.errors.NodeNotFoundError` if no node exists there."""
    return await _resolve_backend(engine).read_metadata(store, path)


async def delete_node(
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Delete the node at `path` and everything under it. Raises
    `zarr.errors.NodeNotFoundError` if absent. `path=""` clears the store."""
    await _resolve_backend(engine).delete_node(store, path)


async def list_children(
    store: Store,
    path: str,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> list[tuple[str, dict[str, JSON]]]:
    """List the direct children of the group at `path` as
    `(path, metadata_document)` pairs (store-relative, no leading `/`). Raises
    `zarr.errors.NodeNotFoundError` if no group exists there."""
    return await _resolve_backend(engine).list_children(store, path)


# --- chunk I/O ---


async def read_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Read and decode the whole chunk at `chunk_coords`. The metadata document
    is authoritative; missing chunks decode to the fill value. The result is a
    read-only view (`.copy()` for a writable array)."""
    be = _resolve_backend(engine)
    raw = await be.read_chunk(store, path, metadata, tuple(chunk_coords))
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    return np.frombuffer(raw, dtype=dtype).reshape(chunk_shape)


async def read_encoded_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> bytes | None:
    """Read the raw, still-encoded bytes of the chunk at `chunk_coords`, or
    `None` if absent. Pure store I/O (`store.get` on the chunk key): the
    `engine` argument is accepted for signature uniformity but unused."""
    key = _chunk_key(metadata, path, tuple(chunk_coords))
    buf = await store.get(key, prototype=default_buffer_prototype())
    return None if buf is None else buf.to_bytes()


async def write_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    value: npt.ArrayLike,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Encode `value` with the codecs in `metadata` and store it as the chunk at
    `chunk_coords`. `value` must match the chunk shape exactly."""
    be = _resolve_backend(engine)
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    arr = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if arr.shape != chunk_shape:
        raise ValueError(f"value shape {arr.shape} does not match chunk shape {chunk_shape}")
    await be.write_chunk(store, path, metadata, tuple(chunk_coords), arr.tobytes())


async def delete_chunk(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    chunk_coords: tuple[int, ...],
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Delete the chunk at `chunk_coords`. Deleting a missing chunk is a no-op."""
    await _resolve_backend(engine).delete_chunk(store, path, metadata, tuple(chunk_coords))


# --- region I/O ---


async def read_region(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    selection: BasicSelection,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Read and decode a region given by a numpy basic-indexing `selection`
    (integers, slices with steps, `Ellipsis`). One backend call fetches the
    step-1 bounding box; strides/reversals/integer-axis removal are applied as
    numpy views. Missing chunks decode to the fill value. Fancy indexing raises
    `TypeError`. The result is a read-only view.

    Note: a `slice(0, N, step)` reads `O(N)` bytes even though `O(N / step)` are
    returned; for sparse selections over large arrays prefer `read_chunk`."""
    be = _resolve_backend(engine)
    dtype, _ = _chunk_dtype_and_shape(metadata)
    shape = _array_shape(metadata)
    starts, lengths, post_index = _normalize_selection(selection, shape)
    if 0 in lengths:
        block = np.empty(lengths, dtype=dtype)
        block.flags.writeable = False
    else:
        raw = await be.read_subset(store, path, metadata, tuple(starts), tuple(lengths))
        block = np.frombuffer(raw, dtype=dtype).reshape(lengths)
    return cast("np.ndarray[Any, np.dtype[Any]]", block[post_index])


async def write_region(
    metadata: Mapping[str, JSON],
    store: Store,
    path: str,
    selection: BasicSelection,
    value: npt.ArrayLike,
    *,
    options: CrudOptions | None = None,
    engine: CrudBackend | str | None = None,
) -> None:
    """Write `value` into the region given by a numpy basic-indexing `selection`.

    The selection is decomposed into chunks: fully-covered chunks are written
    directly, and partially-covered (boundary) chunks are read-modify-written.
    Every write goes through the backend, so the backend's codec pipeline encodes
    the data. `value` is broadcast to the selection shape. Fancy indexing raises
    `TypeError`."""
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.indexing import BasicIndexer

    be = _resolve_backend(engine)
    meta_obj = parse_array_metadata(metadata)
    dtype, chunk_shape = _chunk_dtype_and_shape(metadata)
    shape = _array_shape(metadata)
    indexer = BasicIndexer(selection, shape, ChunkGrid.from_metadata(meta_obj))
    value_arr = np.broadcast_to(np.asarray(value, dtype=dtype), indexer.shape)
    for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer:
        chunk_value = value_arr[out_selection]
        if is_complete_chunk and chunk_value.shape == chunk_shape:
            full = np.ascontiguousarray(chunk_value, dtype=dtype)
        else:
            raw = await be.read_chunk(store, path, metadata, tuple(chunk_coords))
            full = np.frombuffer(raw, dtype=dtype).reshape(chunk_shape).copy()
            full[chunk_selection] = chunk_value
        await be.write_chunk(
            store,
            path,
            metadata,
            tuple(chunk_coords),
            np.ascontiguousarray(full).tobytes(),
        )
