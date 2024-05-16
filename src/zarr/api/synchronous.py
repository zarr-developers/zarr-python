from __future__ import annotations

from typing import Union, Any
import numpy.typing as npt

from zarr.store import StoreLike
from zarr.array import Array
from zarr.group import Group
import zarr.api.asynchronous as async_api
from zarr.sync import sync
from zarr.common import ZarrFormat


def consolidate_metadata(*args: Any, **kwargs: Any) -> Group:
    # TODO
    return Group(sync(async_api.consolidate_metadata(*args, **kwargs)))


def copy(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    # TODO
    return sync(async_api.copy(*args, **kwargs))


def copy_all(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    # TODO
    return sync(async_api.copy_all(*args, **kwargs))


def copy_store(*args: Any, **kwargs: Any) -> tuple[int, int, int]:
    # TODO
    return sync(async_api.copy_store(*args, **kwargs))


def load(
    store: StoreLike, zarr_version: ZarrFormat | None = None, path: str | None = None
) -> Union[npt.ArrayLike, dict[str, npt.ArrayLike]]:
    """
    Load data from an array or group into memory.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system or name of zip file.
    path : str or None, optional
        The path within the store from which to load.

    Returns
    -------
    out
        If the path contains an array, out will be a numpy array. If the path contains
        a group, out will be a dict-like object where keys are array names and values
        are numpy arrays.

    See Also
    --------
    save, savez

    Notes
    -----
    If loading data from a group of arrays, data will not be immediately loaded into
    memory. Rather, arrays will be loaded into memory as they are requested.
    """
    return sync(async_api.load(store=store, zarr_version=zarr_version, path=path))


def open(
    store: StoreLike | None = None,
    mode: str = "a",
    *,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.open
) -> Union[Array, Group]:
    """Convenience function to open a group or array using file-mode-like semantics.

    Parameters
    ----------
    store : Store or string, optional
        Store or path to directory in file system or name of zip file.
    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the store to open.
    **kwargs
        Additional parameters are passed through to :func:`zarr.creation.open_array` or
        :func:`zarr.hierarchy.open_group`.

    Returns
    -------
    z : AsyncArray or AsyncGroup
        Array or group, depending on what exists in the given store.
    """
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
    if isinstance(obj, async_api.AsyncArray):
        return Array(obj)
    else:
        return Group(obj)


def open_consolidated(*args: Any, **kwargs: Any) -> Group:
    return Group(sync(async_api.open_consolidated(*args, **kwargs)))


def save(
    store: StoreLike,
    *args: npt.ArrayLike,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.save
) -> None:
    """Convenience function to save an array or group of arrays to the local file system.

    Parameters
    ----------
    store : Store or string
        Store or path to directory in file system or name of zip file.
    args : ndarray
        NumPy arrays with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the group where the arrays will be saved.
    kwargs
        NumPy arrays with data to save.
    """
    return sync(
        async_api.save(
            store, *args, zarr_version=zarr_version, zarr_format=zarr_format, path=path, **kwargs
        )
    )


def save_array(
    store: StoreLike,
    arr: npt.ArrayLike,
    *,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: Any,  # TODO: type kwargs as valid args to async_api.save_array
) -> None:
    """Convenience function to save a NumPy array to the local file system, following a
    similar API to the NumPy save() function.

    Parameters
    ----------
    store : Store or string
        Store or path to directory in file system or name of zip file.
    arr : ndarray
        NumPy array with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        The path within the store where the array will be saved.
    kwargs
        Passed through to :func:`create`, e.g., compressor.
    """
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
    *args: npt.ArrayLike,
    zarr_version: ZarrFormat | None = None,
    zarr_format: ZarrFormat | None = None,
    path: str | None = None,
    **kwargs: npt.ArrayLike,
) -> None:
    """Convenience function to save several NumPy arrays to the local file system, following a
    similar API to the NumPy savez()/savez_compressed() functions.

    Parameters
    ----------
    store : Store or string
        Store or path to directory in file system or name of zip file.
    args : ndarray
        NumPy arrays with data to save.
    zarr_format : {2, 3, None}, optional
        The zarr format to use when saving.
    path : str or None, optional
        Path within the store where the group will be saved.
    kwargs
        NumPy arrays with data to save.
    """
    return sync(
        async_api.save_group(
            store=store,
            *args,
            zarr_version=zarr_version,
            zarr_format=zarr_format,
            path=path,
            **kwargs,
        )
    )


# TODO: implement or deprecate
# def tree(*args: Any, **kwargs: Any) -> "TreeViewer":
#     return sync(async_api.tree(*args, **kwargs))


# TODO: add type annotations for kwargs
def array(data: npt.ArrayLike, **kwargs: Any) -> Array:
    """Create an array filled with `data`.

    The `data` argument should be a array-like object. For
    other parameter definitions see :func:`zarr.api.synchronous.create`.
    """
    return Array(sync(async_api.array(data=data, **kwargs)))


# TODO: add type annotations for kwargs
def create(*args: Any, **kwargs: Any) -> Array:
    return Array(sync(async_api.create(*args, **kwargs)))


# TODO: move shapelike to common module
# TODO: add type annotations for kwargs
def empty(shape: async_api.ShapeLike, **kwargs: Any) -> Array:
    """Create an empty array.

    For parameter definitions see :func:`zarr.api.asynchronous.create`.

    Notes
    -----
    The contents of an empty Zarr array are not defined. On attempting to
    retrieve data from an empty Zarr array, any values may be returned,
    and these are not guaranteed to be stable from one access to the next.
    """
    return Array(sync(async_api.empty(shape, **kwargs)))


# TODO: move ArrayLike to common module
# TODO: add type annotations for kwargs
def empty_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    """Create an empty array like `a`."""
    return Array(sync(async_api.empty_like(a, **kwargs)))


# TODO: add type annotations for kwargs and fill_value
def full(shape: async_api.ShapeLike, fill_value: Any, **kwargs: Any) -> Array:
    """Create an array, with `fill_value` being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.api.asynchronous.create`.
    """
    return Array(sync(async_api.full(shape=shape, fill_value=fill_value, **kwargs)))


# TODO: move ArrayLike to common module
# TODO: add type annotations for kwargs
async def full_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    """Create a filled array like `a`."""
    return Array(sync(async_api.full_like(a, **kwargs)))


# TODO: add type annotations for kwargs
# TODO: move ShapeLike to common module
def ones(shape: async_api.ShapeLike, **kwargs: Any) -> Array:
    """Create an array, with one being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.api.asynchronous.create`.

    Returns
    -------
    Array
        The new array.
    """
    return Array(sync(async_api.ones(shape, **kwargs)))


# TODO: add type annotations for kwargs
def ones_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    """Create an array of ones like `a`."""
    return Array(sync(async_api.ones_like(a, **kwargs)))


# TODO: update this once async_api.open_array is fully implemented
def open_array(*args: Any, **kwargs: Any) -> Array:
    """Open an array using file-mode-like semantics.

    Parameters
    ----------
    TODO

    Returns
    -------
    AsyncArray
        The opened array.
    """
    return Array(sync(async_api.open_array(*args, **kwargs)))


# TODO: add type annotations for kwargs
def open_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    """Open a persistent array like `a`.

    Parameters
    ----------
    a : Array
        The shape and data-type of a define these same attributes of the returned array.
    path : str
        The path to the new array.
    **kwargs
        Any keyword arguments to pass to the array constructor.

    Returns
    -------
    Array
        The opened array.
    """
    return Array(sync(async_api.open_like(a, **kwargs)))


# TODO: add type annotations for kwargs
def zeros(*args: Any, **kwargs: Any) -> Array:
    """
    Create an array, with zero being used as the default value for
    uninitialized portions of the array.

    For parameter definitions see :func:`zarr.creation.create`.

    Returns:
    Array
        The new array.
    """
    return Array(sync(async_api.zeros(*args, **kwargs)))


# TODO: add type annotations for kwargs
def zeros_like(a: async_api.ArrayLike, **kwargs: Any) -> Array:
    """Create an array of zeros like `a`."""
    return Array(sync(async_api.zeros_like(a, **kwargs)))
