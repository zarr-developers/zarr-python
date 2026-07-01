from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr.core.array import AsyncArray, create_codec_pipeline
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer.core import NDBuffer, default_buffer_prototype
from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON
from zarr.core.group import GroupMetadata
from zarr.core.metadata.io import save_metadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata
from zarr.crud._backend import CrudBackend, NodeExistsError
from zarr.crud._common import parse_array_metadata
from zarr.errors import NodeNotFoundError
from zarr.storage._common import StorePath

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from zarr.abc.store import Store
    from zarr.core.array_spec import ArrayConfigParams
    from zarr.core.common import JSON


# Pin internal AsyncArrays to the native engine so a non-native global
# `array.engine` cannot route their reads back through crud (see read_subset).
_NATIVE_ENGINE_CONFIG: ArrayConfigParams = {"engine": "zarr"}


def _native_dtype(meta_obj: ArrayV3Metadata | ArrayV2Metadata) -> np.dtype[Any]:
    """Numpy dtype in native byte order (zarrs and the facade assume native)."""
    return meta_obj.dtype.to_native_dtype().newbyteorder("=")


def _chunk_shape(meta_obj: ArrayV3Metadata | ArrayV2Metadata) -> tuple[int, ...]:
    if isinstance(meta_obj, ArrayV3Metadata):
        grid = meta_obj.chunk_grid
        if not isinstance(grid, RegularChunkGridMetadata):
            raise NotImplementedError("only regular chunk grids are supported")
        return tuple(grid.chunk_shape)
    return tuple(meta_obj.chunks)


def _array_spec(meta_obj: ArrayV3Metadata | ArrayV2Metadata, shape: tuple[int, ...]) -> ArraySpec:
    order = meta_obj.order if isinstance(meta_obj, ArrayV2Metadata) else "C"
    return ArraySpec(
        shape=shape,
        dtype=meta_obj.dtype,
        fill_value=meta_obj.fill_value,
        config=ArrayConfig.from_dict({"order": order}),
        prototype=default_buffer_prototype(),
    )


def _is_all_fill_value(
    arr: np.ndarray[Any, np.dtype[Any]], fill_value: Any, dtype: np.dtype[Any]
) -> bool:
    """Whether every element of `arr` equals the fill value (NaN-aware for floats)."""
    if fill_value is None:
        return False
    fill = np.asarray(fill_value, dtype=dtype)
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        return bool(np.array_equal(arr, np.broadcast_to(fill, arr.shape), equal_nan=True))
    return bool(np.all(arr == fill))


class ReferenceBackend(CrudBackend):
    """Pure-Python CRUD backend wrapping zarr-python's own machinery.

    Constructs no high-level `Array` for chunk operations (it drives the codec
    pipeline directly); it does reuse `AsyncArray.getitem` for multi-chunk
    subset reads, which is exactly the `BasicIndexer` + codec-pipeline read path.
    """

    async def _node_exists(self, store: Store, path: str) -> bool:
        proto = default_buffer_prototype()
        sp = StorePath(store, path.strip("/"))
        for meta_key in (ZARR_JSON, ZARRAY_JSON, ZGROUP_JSON):
            if await (sp / meta_key).get(prototype=proto) is not None:
                return True
        return False

    async def create_array(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        meta_obj = parse_array_metadata(metadata)
        await self._create(store, path, meta_obj, overwrite=overwrite)

    async def create_group(
        self, store: Store, path: str, metadata: Mapping[str, JSON], *, overwrite: bool
    ) -> None:
        meta_obj = GroupMetadata.from_dict(dict(metadata))
        await self._create(store, path, meta_obj, overwrite=overwrite)

    async def _create(self, store: Store, path: str, meta_obj: Any, *, overwrite: bool) -> None:
        sp = StorePath(store, path.strip("/"))
        if overwrite:
            await store.delete_dir(path.strip("/"))
        elif await self._node_exists(store, path):
            raise NodeExistsError(f"a node already exists at path {path!r}")
        await save_metadata(sp, meta_obj, ensure_parents=True)

    async def read_metadata(self, store: Store, path: str) -> dict[str, JSON]:
        from zarr.core._json import buffer_to_json_object

        proto = default_buffer_prototype()
        sp = StorePath(store, path.strip("/"))
        buf = await (sp / ZARR_JSON).get(prototype=proto)
        if buf is not None:
            return buffer_to_json_object(buf)
        for meta_key in (ZARRAY_JSON, ZGROUP_JSON):
            b = await (sp / meta_key).get(prototype=proto)
            if b is not None:
                doc = buffer_to_json_object(b)
                zattrs = await (sp / ZATTRS_JSON).get(prototype=proto)
                if zattrs is not None:
                    doc["attributes"] = buffer_to_json_object(zattrs)
                return doc
        raise NodeNotFoundError(f"no node found at path {path!r}")

    async def read_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> bytes:
        meta_obj = parse_array_metadata(metadata)
        shape = _chunk_shape(meta_obj)
        np_dtype = _native_dtype(meta_obj)
        sp = StorePath(store, path.strip("/"))
        chunk_key = meta_obj.encode_chunk_key(coords)
        buf = await (sp / chunk_key).get(prototype=default_buffer_prototype())
        if buf is None:
            arr = np.full(shape, meta_obj.fill_value, dtype=np_dtype)
        else:
            pipeline = create_codec_pipeline(meta_obj)
            spec = _array_spec(meta_obj, shape)
            decoded = list(await pipeline.decode([(buf, spec)]))
            nd_buf = decoded[0]
            if nd_buf is None:
                arr = np.full(shape, meta_obj.fill_value, dtype=np_dtype)
            else:
                arr = np.asarray(nd_buf.as_numpy_array(), dtype=np_dtype)
        return np.ascontiguousarray(arr).tobytes()

    async def read_subset(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        start: Sequence[int],
        shape: Sequence[int],
    ) -> bytes:
        meta_obj = parse_array_metadata(metadata)
        np_dtype = _native_dtype(meta_obj)
        # Pin the native engine: this AsyncArray is an internal implementation
        # detail, and without the pin a non-native global `array.engine` would
        # route its getitem back through crud, looping into this method.
        async_arr = AsyncArray(
            metadata=meta_obj,
            store_path=StorePath(store, path.strip("/")),
            config=_NATIVE_ENGINE_CONFIG,
        )
        selection = tuple(slice(s, s + length) for s, length in zip(start, shape, strict=True))
        result = await async_arr.getitem(selection)
        return np.ascontiguousarray(np.asarray(result, dtype=np_dtype)).tobytes()

    async def write_chunk(
        self,
        store: Store,
        path: str,
        metadata: Mapping[str, JSON],
        coords: tuple[int, ...],
        data: bytes,
    ) -> None:
        meta_obj = parse_array_metadata(metadata)
        shape = _chunk_shape(meta_obj)
        np_dtype = _native_dtype(meta_obj)
        sp = StorePath(store, path.strip("/"))
        chunk_key = meta_obj.encode_chunk_key(coords)
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        if _is_all_fill_value(arr, meta_obj.fill_value, np_dtype):
            await (sp / chunk_key).delete()
            return
        pipeline = create_codec_pipeline(meta_obj)
        spec = _array_spec(meta_obj, shape)
        encoded = list(await pipeline.encode([(NDBuffer.from_ndarray_like(arr), spec)]))
        buf = encoded[0]
        if buf is None:
            await (sp / chunk_key).delete()
        else:
            await (sp / chunk_key).set(buf)

    async def delete_chunk(
        self, store: Store, path: str, metadata: Mapping[str, JSON], coords: tuple[int, ...]
    ) -> None:
        meta_obj = parse_array_metadata(metadata)
        sp = StorePath(store, path.strip("/"))
        await (sp / meta_obj.encode_chunk_key(coords)).delete()

    async def delete_node(self, store: Store, path: str) -> None:
        if not await self._node_exists(store, path):
            raise NodeNotFoundError(f"no node found at path {path!r}")
        await store.delete_dir(path.strip("/"))

    async def list_children(self, store: Store, path: str) -> list[tuple[str, dict[str, JSON]]]:
        p = path.strip("/")
        if not await self._node_exists(store, path):
            raise NodeNotFoundError(f"no node found at path {path!r}")
        prefix = f"{p}/" if p else ""
        children: list[tuple[str, dict[str, JSON]]] = []
        async for name in store.list_dir(prefix):
            child_path = f"{p}/{name}" if p else name
            if await self._node_exists(store, child_path):
                children.append((name, await self.read_metadata(store, child_path)))
        return children
