from __future__ import annotations

from typing import Any

import zarr.api.asynchronous as async_api
from zarr.array import Array, AsyncArray
from zarr.buffer import NDArrayLike
from zarr.common import JSON, ChunkCoords, OpenMode, ZarrFormat
from zarr.group import Group
from zarr.store import StoreLike
from zarr.sync import sync


def consolidate_metadata(*args: Any, **kwargs: Any) -> Group:
    return Group(sync(async_api.consolidate_metadata(*args, **kwargs)))


def copy(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    return sync(async_api.copy(*args, **kwargs))


def copy_all(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    return sync(async_api.copy_all(*args, **kwargs))


def copy_store(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    return sync(async_api.copy_store(*args, **kwargs))


def load(
    store: StoreLike, zarr_version: ZarrFormat | None = None, path: str | None = None
) -> NDArrayLike | dict[str, NDArrayLike]:
    return sync(async_api.load(store=store, zarr_version=zarr_version, path=path))


def open(
    *,
    store: StoreLike | None = None,
    mode: OpenMode | None = None,  # type and value changed
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.open
) -> Array | Group:
    obj = sync(
        async_api.open(
            store=store,
            mode=mode,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            **kwargs,
        )
    )
    if isinstance(obj, AsyncArray):
        return Array(obj)
    else:
        return Group(obj)


def open_consolidated(*args: Any, **kwargs: Any) -> Group:
    return Group(sync(async_api.open_consolidated(*args, **kwargs)))


def save(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.save
) -> None:
    return sync(
        async_api.save(
            store, *args, zarr_version=zarr_version, zarr_format=zarr_format, path=path, **kwargs
        )
    )


def save_array(
    store: StoreLike,
    arr: NDArrayLike,
    *,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.save_array
) -> None:
    return sync(
        async_api.save_array(
            store=store,
            arr=arr,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            **kwargs,
        )
    )


def save_group(
    store: StoreLike,
    *args: NDArrayLike,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: NDArrayLike,
) -> None:
    return sync(
        async_api.save_group(
            store,
            *args,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            **kwargs,
        )
    )


def tree(*args: Any, **kwargs: Any) -> None:
    return sync(async_api.tree(*args, **kwargs))


# TODO: add type annotations for kwargs
def array(data: NDArrayLike, **kwargs: Any) -> Array:
    return Array(sync(async_api.array(data=data, **kwargs)))


def group(
    *,  # Note: this is a change from v2
    store: StoreLike | None = None,
    overwrite: bool = False,
    chunk_store: StoreLike | None = None,  # not used in async_api
    cache_attrs: bool | None = None,  # default changed, not used in async_api
    synchronizer: Any | None = None,  # not used in async_api
    path: str | None = None,
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used in async_api
    attributes: dict[str, JSON] | None = None,
) -> Group:
    return Group(
        sync(
            async_api.group(
                store=store,
                overwrite=overwrite,
                chunk_store=chunk_store,
                cache_attrs=cache_attrs,
                synchronizer=synchronizer,
                path=path,
                zarr_version=zarr_version,
                zarr_format=zarr_format,
                meta_array=meta_array,
                attributes=attributes,
            )
        )
    )


def open_group(
    *,  # Note: this is a change from v2
    store: StoreLike | None = None,
    mode: OpenMode | None = None,  # not used in async api
    cache_attrs: bool | None = None,  # default changed, not used in async api
    synchronizer: Any = None,  # not used in async api
    path: str | None = None,
    chunk_store: StoreLike | None = None,  # not used in async api
    storage_options: dict[str, Any] | None = None,  # not used in async api
    zarr_version: ZarrFormat | None = None,  # deprecated
    zarr_format: ZarrFormat | None = None,
    meta_array: Any | None = None,  # not used in async api
) -> Group:
    return Group(
        sync(
            async_api.open_group(
                store=store,
                mode=mode,
                cache_attrs=cache_attrs,
                synchronizer=synchronizer,
                path=path,
                chunk_store=chunk_store,
                storage_options=storage_options,
                zarr_version=zarr_version,
                zarr_format=zarr_format,
                meta_array=meta_array,
            )
        )
    )


# TODO: add type annotations for kwargs
def create(*args: Any, **kwargs: Any) -> Array:
    return Array(sync(async_api.create(*args, **kwargs)))


# TODO: add type annotations for kwargs
def empty(shape: ChunkCoords, **kwargs: Any) -> Array:
    return Array(sync(async_api.empty(shape, **kwargs)))


# TODO: move ArrayLike to common module
# TODO: add type annotations for kwargs
def empty_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    return Array(sync(async_api.empty_like(a, **kwargs)))


# TODO: add type annotations for kwargs and fill_value
def full(shape: ChunkCoords, fill_value: Any, **kwargs: Any) -> Array:
    return Array(sync(async_api.full(shape=shape, fill_value=fill_value, **kwargs)))


# TODO: move ArrayLike to common module
# TODO: add type annotations for kwargs
def full_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    return Array(sync(async_api.full_like(a, **kwargs)))


# TODO: add type annotations for kwargs
def ones(shape: ChunkCoords, **kwargs: Any) -> Array:
    return Array(sync(async_api.ones(shape, **kwargs)))


# TODO: add type annotations for kwargs
def ones_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    return Array(sync(async_api.ones_like(a, **kwargs)))


# TODO: update this once async_api.open_array is fully implemented
def open_array(*args: Any, **kwargs: Any) -> Array:
    return Array(sync(async_api.open_array(*args, **kwargs)))


# TODO: add type annotations for kwargs
def open_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    return Array(sync(async_api.open_like(a, **kwargs)))


# TODO: add type annotations for kwargs
def zeros(*args: Any, **kwargs: Any) -> Array:
    return Array(sync(async_api.zeros(*args, **kwargs)))


# TODO: add type annotations for kwargs
def zeros_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    return Array(sync(async_api.zeros_like(a, **kwargs)))


consolidate_metadata.__doc__ = async_api.copy.__doc__
copy.__doc__ = async_api.copy.__doc__
copy_all.__doc__ = async_api.copy_all.__doc__
copy_store.__doc__ = async_api.copy_store.__doc__
load.__doc__ = async_api.load.__doc__
open.__doc__ = async_api.open.__doc__
open_consolidated.__doc__ = async_api.open_consolidated.__doc__
save.__doc__ = async_api.save.__doc__
save_array.__doc__ = async_api.save_array.__doc__
save_group.__doc__ = async_api.save_group.__doc__
tree.__doc__ = async_api.tree.__doc__
array.__doc__ = async_api.array.__doc__
group.__doc__ = async_api.group.__doc__
open_group.__doc__ = async_api.open_group.__doc__
create.__doc__ = async_api.create.__doc__
empty.__doc__ = async_api.empty.__doc__
empty_like.__doc__ = async_api.empty_like.__doc__
full.__doc__ = async_api.full.__doc__
full_like.__doc__ = async_api.full_like.__doc__
ones.__doc__ = async_api.ones.__doc__
ones_like.__doc__ = async_api.ones_like.__doc__
open_array.__doc__ = async_api.open_array.__doc__
open_like.__doc__ = async_api.open_like.__doc__
zeros.__doc__ = async_api.zeros.__doc__
zeros_like.__doc__ = async_api.zeros_like.__doc__
