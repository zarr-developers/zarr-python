# -*- coding: utf-8 -*-
"""Convenience functions for storing and loading data."""
from __future__ import absolute_import, print_function, division
from collections import Mapping


from zarr.core import Array
from zarr.creation import open_array, normalize_store_arg, array as _create_array
from zarr.hierarchy import open_group, group as _create_group, Group
from zarr.storage import contains_array, contains_group
from zarr.errors import err_path_not_found
from zarr.util import normalize_storage_path


# noinspection PyShadowingBuiltins
def open(store, mode='a', **kwargs):
    """Convenience function to open a group or array using file-mode-like semantics.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system or name of zip file.
    mode : {'r', 'r+', 'a', 'w', 'w-'}, optional
        Persistence mode: 'r' means read only (must exist); 'r+' means
        read/write (must exist); 'a' means read/write (create if doesn't
        exist); 'w' means create (overwrite if exists); 'w-' means create
        (fail if exists).
    **kwargs
        Additional parameters are passed through to :func:`zarr.open_array` or
        :func:`zarr.open_group`.

    See Also
    --------
    zarr.open_array, zarr.open_group

    Examples
    --------

    Storing data in a directory 'data/example.zarr' on the local file system::

        >>> import zarr
        >>> store = 'data/example.zarr'
        >>> zw = zarr.open(store, mode='w', shape=100, dtype='i4')  # open new array
        >>> zw
        <zarr.core.Array (100,) int32>
        >>> za = zarr.open(store, mode='a')  # open existing array for reading and writing
        >>> za
        <zarr.core.Array (100,) int32>
        >>> zr = zarr.open(store, mode='r')  # open existing array read-only
        >>> zr
        <zarr.core.Array (100,) int32 read-only>
        >>> gw = zarr.open(store, mode='w')  # open new group, overwriting any previous data
        >>> gw
        <zarr.hierarchy.Group '/'>
        >>> ga = zarr.open(store, mode='a')  # open existing group for reading and writing
        >>> ga
        <zarr.hierarchy.Group '/'>
        >>> gr = zarr.open(store, mode='r')  # open existing group read-only
        >>> gr
        <zarr.hierarchy.Group '/' read-only>

    """

    path = kwargs.get('path', None)
    # handle polymorphic store arg
    store = normalize_store_arg(store, clobber=(mode == 'w'))
    path = normalize_storage_path(path)

    if mode in {'w', 'w-', 'x'}:
        if 'shape' in kwargs:
            return open_array(store, mode=mode, **kwargs)
        else:
            return open_group(store, mode=mode, **kwargs)

    elif mode == 'a':
        if contains_array(store, path):
            return open_array(store, mode=mode, **kwargs)
        elif contains_group(store, path):
            return open_group(store, mode=mode, **kwargs)
        elif 'shape' in kwargs:
            return open_array(store, mode=mode, **kwargs)
        else:
            return open_group(store, mode=mode, **kwargs)

    else:
        if contains_array(store, path):
            return open_array(store, mode=mode, **kwargs)
        elif contains_group(store, path):
            return open_group(store, mode=mode, **kwargs)
        else:
            err_path_not_found(path)


def save_array(store, arr, **kwargs):
    """Convenience function to save a NumPy array to the local file system, following a similar
    API to the NumPy save() function.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system or name of zip file.
    arr : ndarray
        NumPy array with data to save.
    kwargs
        Passed through to :func:`create`, e.g., compressor.

    Examples
    --------
    Save an array to a directory on the file system (uses a :class:`DirectoryStore`)::

        >>> import zarr
        >>> import numpy as np
        >>> arr = np.arange(10000)
        >>> zarr.save_array('data/example.zarr', arr)
        >>> zarr.load('data/example.zarr')
        array([   0,    1,    2, ..., 9997, 9998, 9999])

    Save an array to a single file (uses a :class:`ZipStore`)::

        >>> zarr.save_array('data/example.zip', arr)
        >>> zarr.load('data/example.zip')
        array([   0,    1,    2, ..., 9997, 9998, 9999])

    """
    may_need_closing = isinstance(store, str)
    store = normalize_store_arg(store, clobber=True)
    try:
        _create_array(arr, store=store, overwrite=True, **kwargs)
    finally:
        if may_need_closing and hasattr(store, 'close'):
            # needed to ensure zip file records are written
            store.close()


def save_group(store, *args, **kwargs):
    """Convenience function to save several NumPy arrays to the local file system, following a
    similar API to the NumPy savez()/savez_compressed() functions.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system or name of zip file.
    args : ndarray
        NumPy arrays with data to save.
    kwargs
        NumPy arrays with data to save.

    Examples
    --------
    Save several arrays to a directory on the file system (uses a :class:`DirectoryStore`)::

        >>> import zarr
        >>> import numpy as np
        >>> a1 = np.arange(10000)
        >>> a2 = np.arange(10000, 0, -1)
        >>> zarr.save_group('data/example.zarr', a1, a2)
        >>> loader = zarr.load('data/example.zarr')
        >>> loader
        <LazyLoader: arr_0, arr_1>
        >>> loader['arr_0']
        array([   0,    1,    2, ..., 9997, 9998, 9999])
        >>> loader['arr_1']
        array([10000,  9999,  9998, ...,     3,     2,     1])

    Save several arrays using named keyword arguments::

        >>> zarr.save_group('data/example.zarr', foo=a1, bar=a2)
        >>> loader = zarr.load('data/example.zarr')
        >>> loader
        <LazyLoader: bar, foo>
        >>> loader['foo']
        array([   0,    1,    2, ..., 9997, 9998, 9999])
        >>> loader['bar']
        array([10000,  9999,  9998, ...,     3,     2,     1])

    Store several arrays in a single zip file (uses a :class:`ZipStore`)::

        >>> zarr.save_group('data/example.zip', foo=a1, bar=a2)
        >>> loader = zarr.load('data/example.zip')
        >>> loader
        <LazyLoader: bar, foo>
        >>> loader['foo']
        array([   0,    1,    2, ..., 9997, 9998, 9999])
        >>> loader['bar']
        array([10000,  9999,  9998, ...,     3,     2,     1])

    Notes
    -----
    Default compression options will be used.

    """
    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError('at least one array must be provided')
    # handle polymorphic store arg
    may_need_closing = isinstance(store, str)
    store = normalize_store_arg(store, clobber=True)
    try:
        grp = _create_group(store, overwrite=True)
        for i, arr in enumerate(args):
            k = 'arr_{}'.format(i)
            grp.create_dataset(k, data=arr, overwrite=True)
        for k, arr in kwargs.items():
            grp.create_dataset(k, data=arr, overwrite=True)
    finally:
        if may_need_closing and hasattr(store, 'close'):
            # needed to ensure zip file records are written
            store.close()


def save(store, *args, **kwargs):
    """Convenience function to save an array or arrays to the local file system.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system or name of zip file.
    args : ndarray
        NumPy arrays with data to save.
    kwargs
        NumPy arrays with data to save.

    Examples
    --------
    Save an array to a directory on the file system (uses a :class:`DirectoryStore`)::

        >>> import zarr
        >>> import numpy as np
        >>> arr = np.arange(10000)
        >>> zarr.save('data/example.zarr', arr)
        >>> zarr.load('data/example.zarr')
        array([   0,    1,    2, ..., 9997, 9998, 9999])

    Save an array to a single file (uses a :class:`ZipStore`)::

        >>> zarr.save('data/example.zip', arr)
        >>> zarr.load('data/example.zip')
        array([   0,    1,    2, ..., 9997, 9998, 9999])

    Save several arrays to a directory on the file system (uses a :class:`DirectoryStore`)::

        >>> import zarr
        >>> import numpy as np
        >>> a1 = np.arange(10000)
        >>> a2 = np.arange(10000, 0, -1)
        >>> zarr.save('data/example.zarr', a1, a2)
        >>> loader = zarr.load('data/example.zarr')
        >>> loader
        <LazyLoader: arr_0, arr_1>
        >>> loader['arr_0']
        array([   0,    1,    2, ..., 9997, 9998, 9999])
        >>> loader['arr_1']
        array([10000,  9999,  9998, ...,     3,     2,     1])

    Save several arrays using named keyword arguments::

        >>> zarr.save('data/example.zarr', foo=a1, bar=a2)
        >>> loader = zarr.load('data/example.zarr')
        >>> loader
        <LazyLoader: bar, foo>
        >>> loader['foo']
        array([   0,    1,    2, ..., 9997, 9998, 9999])
        >>> loader['bar']
        array([10000,  9999,  9998, ...,     3,     2,     1])

    Store several arrays in a single zip file (uses a :class:`ZipStore`)::

        >>> zarr.save('data/example.zip', foo=a1, bar=a2)
        >>> loader = zarr.load('data/example.zip')
        >>> loader
        <LazyLoader: bar, foo>
        >>> loader['foo']
        array([   0,    1,    2, ..., 9997, 9998, 9999])
        >>> loader['bar']
        array([10000,  9999,  9998, ...,     3,     2,     1])

    See Also
    --------
    save_array, save_group

    """
    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError('at least one array must be provided')
    if len(args) == 1 and len(kwargs) == 0:
        save_array(store, args[0])
    else:
        save_group(store, *args, **kwargs)


class LazyLoader(Mapping):

    def __init__(self, grp):
        self.grp = grp
        self.cache = dict()

    def __getitem__(self, item):
        try:
            return self.cache[item]
        except KeyError:
            arr = self.grp[item][...]
            self.cache[item] = arr
            return arr

    def __len__(self):
        return len(self.grp)

    def __iter__(self):
        return iter(self.grp)

    def __contains__(self, item):
        return item in self.grp

    def __repr__(self):
        r = '<LazyLoader: '
        r += ', '.join(sorted(self.grp.array_keys()))
        r += '>'
        return r


def load(store):
    """Load data from an array or group into memory.

    Parameters
    ----------
    store : MutableMapping or string
        Store or path to directory in file system or name of zip file.

    Returns
    -------
    out
        If the store contains an array, out will be a numpy array. If the store contains a group,
        out will be a dict-like object where keys are array names and values are numpy arrays.

    See Also
    --------
    save, savez

    Notes
    -----
    If loading data from a group of arrays, data will not be immediately loaded into memory.
    Rather, arrays will be loaded into memory as they are requested.

    """
    # handle polymorphic store arg
    store = normalize_store_arg(store)
    if contains_array(store, path=None):
        return Array(store=store, path=None)[...]
    elif contains_group(store, path=None):
        grp = Group(store=store, path=None)
        return LazyLoader(grp)
