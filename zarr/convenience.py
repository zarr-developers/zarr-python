# -*- coding: utf-8 -*-
"""Convenience functions for storing and loading data."""
from __future__ import absolute_import, print_function, division
from collections import Mapping
import io
import re
import itertools


from zarr.core import Array
from zarr.creation import (open_array, normalize_store_arg,
                           array as _create_array)
from zarr.hierarchy import open_group, group as _create_group, Group
from zarr.storage import contains_array, contains_group
from zarr.errors import err_path_not_found, CopyError
from zarr.util import normalize_storage_path, TreeViewer, buffer_size
from zarr.compat import PY2, text_type


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
        >>> gw = zarr.open(store, mode='w')  # open new group, overwriting previous data
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
    """Convenience function to save a NumPy array to the local file system, following a
    similar API to the NumPy save() function.

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
    Save several arrays to a directory on the file system (uses a
    :class:`DirectoryStore`):

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
    """Convenience function to save an array or group of arrays to the local file system.

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

    Save an array to a Zip file (uses a :class:`ZipStore`)::

        >>> zarr.save('data/example.zip', arr)
        >>> zarr.load('data/example.zip')
        array([   0,    1,    2, ..., 9997, 9998, 9999])

    Save several arrays to a directory on the file system (uses a
    :class:`DirectoryStore` and stores arrays in a group)::

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
        If the store contains an array, out will be a numpy array. If the store contains
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
    # handle polymorphic store arg
    store = normalize_store_arg(store)
    if contains_array(store, path=None):
        return Array(store=store, path=None)[...]
    elif contains_group(store, path=None):
        grp = Group(store=store, path=None)
        return LazyLoader(grp)


def tree(grp, expand=False, level=None):
    """Provide a ``print``-able display of the hierarchy. This function is provided
    mainly as a convenience for obtaining a tree view of an h5py group - zarr groups
    have a ``.tree()`` method.

    Parameters
    ----------
    grp : Group
        Zarr or h5py group.
    expand : bool, optional
        Only relevant for HTML representation. If True, tree will be fully expanded.
    level : int, optional
        Maximum depth to descend into hierarchy.

    Examples
    --------
    >>> import zarr
    >>> g1 = zarr.group()
    >>> g2 = g1.create_group('foo')
    >>> g3 = g1.create_group('bar')
    >>> g4 = g3.create_group('baz')
    >>> g5 = g3.create_group('qux')
    >>> d1 = g5.create_dataset('baz', shape=100, chunks=10)
    >>> g1.tree()
    /
     ├── bar
     │   ├── baz
     │   └── qux
     │       └── baz (100,) float64
     └── foo
    >>> import h5py
    >>> h5f = h5py.File('data/example.h5', mode='w')
    >>> zarr.copy_all(g1, h5f)
    (5, 0, 800)
    >>> zarr.tree(h5f)
    /
     ├── bar
     │   ├── baz
     │   └── qux
     │       └── baz (100,) float64
     └── foo

    See Also
    --------
    zarr.hierarchy.Group.tree

    Notes
    -----
    Please note that this is an experimental feature. The behaviour of this
    function is still evolving and the default output and/or parameters may change
    in future versions.

    """

    return TreeViewer(grp, expand=expand, level=level)


class _LogWriter(object):

    def __init__(self, log):
        self.log_func = None
        self.log_file = None
        self.needs_closing = False
        if log is None:
            # don't do any logging
            pass
        elif callable(log):
            self.log_func = log
        elif isinstance(log, str):
            self.log_file = io.open(log, mode='w')
            self.needs_closing = True
        else:
            if not hasattr(log, 'write'):
                raise TypeError('log must be a callable function, file path or '
                                'file-like object, found %r' % log)
            self.log_file = log
            self.needs_closing = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.log_file is not None and self.needs_closing:
            self.log_file.close()

    def __call__(self, *args, **kwargs):
        if self.log_file is not None:
            kwargs['file'] = self.log_file
            if PY2:  # pragma: py3 no cover
                # expect file opened in text mode, need to adapt message
                args = [text_type(a) for a in args]
            print(*args, **kwargs)
            if hasattr(self.log_file, 'flush'):
                # get immediate feedback
                self.log_file.flush()
        elif self.log_func is not None:
            self.log_func(*args, **kwargs)


def _log_copy_summary(log, dry_run, n_copied, n_skipped, n_bytes_copied):
    # log a final message with a summary of what happened
    if dry_run:
        message = 'dry run: '
    else:
        message = 'all done: '
    message += '{:,} copied, {:,} skipped'.format(n_copied, n_skipped)
    if not dry_run:
        message += ', {:,} bytes copied'.format(n_bytes_copied)
    log(message)


def copy_store(source, dest, source_path='', dest_path='', excludes=None,
               includes=None, flags=0, if_exists='raise', dry_run=False,
               log=None):
    """Copy data directly from the `source` store to the `dest` store. Use this
    function when you want to copy a group or array in the most efficient way,
    preserving all configuration and attributes. This function is more efficient
    than the copy() or copy_all() functions because it avoids de-compressing and
    re-compressing data, rather the compressed chunk data for each array are
    copied directly between stores.

    Parameters
    ----------
    source : Mapping
        Store to copy data from.
    dest : MutableMapping
        Store to copy data into.
    source_path : str, optional
        Only copy data from under this path in the source store.
    dest_path : str, optional
        Copy data into this path in the destination store.
    excludes : sequence of str, optional
        One or more regular expressions which will be matched against keys in
        the source store. Any matching key will not be copied.
    includes : sequence of str, optional
        One or more regular expressions which will be matched against keys in
        the source store and will override any excludes also matching.
    flags : int, optional
        Regular expression flags used for matching excludes and includes.
    if_exists : {'raise', 'replace', 'skip'}, optional
        How to handle keys that already exist in the destination store. If
        'raise' then a CopyError is raised on the first key already present
        in the destination store. If 'replace' then any data will be replaced in
        the destination. If 'skip' then any existing keys will not be copied.
    dry_run : bool, optional
        If True, don't actually copy anything, just log what would have
        happened.
    log : callable, file path or file-like object, optional
        If provided, will be used to log progress information.

    Returns
    -------
    n_copied : int
        Number of items copied.
    n_skipped : int
        Number of items skipped.
    n_bytes_copied : int
        Number of bytes of data that were actually copied.

    Examples
    --------

    >>> import zarr
    >>> store1 = zarr.DirectoryStore('data/example.zarr')
    >>> root = zarr.group(store1, overwrite=True)
    >>> foo = root.create_group('foo')
    >>> bar = foo.create_group('bar')
    >>> baz = bar.create_dataset('baz', shape=100, chunks=50, dtype='i8')
    >>> import numpy as np
    >>> baz[:] = np.arange(100)
    >>> root.tree()
    /
     └── foo
         └── bar
             └── baz (100,) int64
    >>> from sys import stdout
    >>> store2 = zarr.ZipStore('data/example.zip', mode='w')
    >>> zarr.copy_store(store1, store2, log=stdout)
    copy .zgroup
    copy foo/.zgroup
    copy foo/bar/.zgroup
    copy foo/bar/baz/.zarray
    copy foo/bar/baz/0
    copy foo/bar/baz/1
    all done: 6 copied, 0 skipped, 566 bytes copied
    (6, 0, 566)
    >>> new_root = zarr.group(store2)
    >>> new_root.tree()
    /
     └── foo
         └── bar
             └── baz (100,) int64
    >>> new_root['foo/bar/baz'][:]
    array([ 0,  1,  2,  ..., 97, 98, 99])
    >>> store2.close()  # zip stores need to be closed

    Notes
    -----
    Please note that this is an experimental feature. The behaviour of this
    function is still evolving and the default behaviour and/or parameters may change
    in future versions.

    """

    # normalize paths
    source_path = normalize_storage_path(source_path)
    dest_path = normalize_storage_path(dest_path)
    if source_path:
        source_path = source_path + '/'
    if dest_path:
        dest_path = dest_path + '/'

    # normalize excludes and includes
    if excludes is None:
        excludes = []
    elif isinstance(excludes, str):
        excludes = [excludes]
    if includes is None:
        includes = []
    elif isinstance(includes, str):
        includes = [includes]
    excludes = [re.compile(e, flags) for e in excludes]
    includes = [re.compile(i, flags) for i in includes]

    # check if_exists parameter
    valid_if_exists = ['raise', 'replace', 'skip']
    if if_exists not in valid_if_exists:
        raise ValueError('if_exists must be one of {!r}; found {!r}'
                         .format(valid_if_exists, if_exists))

    # setup counting variables
    n_copied = n_skipped = n_bytes_copied = 0

    # setup logging
    with _LogWriter(log) as log:

        # iterate over source keys
        for source_key in sorted(source.keys()):

            # filter to keys under source path
            if source_key.startswith(source_path):

                # process excludes and includes
                exclude = False
                for prog in excludes:
                    if prog.search(source_key):
                        exclude = True
                        break
                if exclude:
                    for prog in includes:
                        if prog.search(source_key):
                            exclude = False
                            break
                if exclude:
                    continue

                # map key to destination path
                key_suffix = source_key[len(source_path):]
                dest_key = dest_path + key_suffix

                # create a descriptive label for this operation
                descr = source_key
                if dest_key != source_key:
                    descr = descr + ' -> ' + dest_key

                # decide what to do
                do_copy = True
                if if_exists != 'replace':
                    if dest_key in dest:
                        if if_exists == 'raise':
                            raise CopyError('key {!r} exists in destination'
                                            .format(dest_key))
                        elif if_exists == 'skip':
                            do_copy = False

                # take action
                if do_copy:
                    log('copy {}'.format(descr))
                    if not dry_run:
                        data = source[source_key]
                        n_bytes_copied += buffer_size(data)
                        dest[dest_key] = data
                    n_copied += 1
                else:
                    log('skip {}'.format(descr))
                    n_skipped += 1

        # log a final message with a summary of what happened
        _log_copy_summary(log, dry_run, n_copied, n_skipped, n_bytes_copied)

    return n_copied, n_skipped, n_bytes_copied


def _check_dest_is_group(dest):
    if not hasattr(dest, 'create_dataset'):
        raise ValueError('dest must be a group, got {!r}'.format(dest))


def copy(source, dest, name=None, shallow=False, without_attrs=False, log=None,
         if_exists='raise', dry_run=False, **create_kws):
    """Copy the `source` array or group into the `dest` group.

    Parameters
    ----------
    source : group or array/dataset
        A zarr group or array, or an h5py group or dataset.
    dest : group
        A zarr or h5py group.
    name : str, optional
        Name to copy the object to.
    shallow : bool, optional
        If True, only copy immediate children of `source`.
    without_attrs : bool, optional
        Do not copy user attributes.
    log : callable, file path or file-like object, optional
        If provided, will be used to log progress information.
    if_exists : {'raise', 'replace', 'skip', 'skip_initialized'}, optional
        How to handle arrays that already exist in the destination group. If
        'raise' then a CopyError is raised on the first array already present
        in the destination group. If 'replace' then any array will be
        replaced in the destination. If 'skip' then any existing arrays will
        not be copied. If 'skip_initialized' then any existing arrays with
        all chunks initialized will not be copied (not available when copying to
        h5py).
    dry_run : bool, optional
        If True, don't actually copy anything, just log what would have
        happened.
    **create_kws
        Passed through to the create_dataset method when copying an array/dataset.

    Returns
    -------
    n_copied : int
        Number of items copied.
    n_skipped : int
        Number of items skipped.
    n_bytes_copied : int
        Number of bytes of data that were actually copied.

    Examples
    --------
    Here's an example of copying a group named 'foo' from an HDF5 file to a
    Zarr group::

        >>> import h5py
        >>> import zarr
        >>> import numpy as np
        >>> source = h5py.File('data/example.h5', mode='w')
        >>> foo = source.create_group('foo')
        >>> baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
        >>> spam = source.create_dataset('spam', data=np.arange(100, 200), chunks=(30,))
        >>> zarr.tree(source)
        /
         ├── foo
         │   └── bar
         │       └── baz (100,) int64
         └── spam (100,) int64
        >>> dest = zarr.group()
        >>> from sys import stdout
        >>> zarr.copy(source['foo'], dest, log=stdout)
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        all done: 3 copied, 0 skipped, 800 bytes copied
        (3, 0, 800)
        >>> dest.tree()  # N.B., no spam
        /
         └── foo
             └── bar
                 └── baz (100,) int64
        >>> source.close()

    The ``if_exists`` parameter provides options for how to handle pre-existing data in
    the destination. Here are some examples of these options, also using
    ``dry_run=True`` to find out what would happen without actually copying anything::

        >>> source = zarr.group()
        >>> dest = zarr.group()
        >>> baz = source.create_dataset('foo/bar/baz', data=np.arange(100))
        >>> spam = source.create_dataset('foo/spam', data=np.arange(1000))
        >>> existing_spam = dest.create_dataset('foo/spam', data=np.arange(1000))
        >>> from sys import stdout
        >>> try:
        ...     zarr.copy(source['foo'], dest, log=stdout, dry_run=True)
        ... except zarr.CopyError as e:
        ...     print(e)
        ...
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        an object 'spam' already exists in destination '/foo'
        >>> zarr.copy(source['foo'], dest, log=stdout, if_exists='replace', dry_run=True)
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        copy /foo/spam (1000,) int64
        dry run: 4 copied, 0 skipped
        (4, 0, 0)
        >>> zarr.copy(source['foo'], dest, log=stdout, if_exists='skip', dry_run=True)
        copy /foo
        copy /foo/bar
        copy /foo/bar/baz (100,) int64
        skip /foo/spam (1000,) int64
        dry run: 3 copied, 1 skipped
        (3, 1, 0)

    Notes
    -----
    Please note that this is an experimental feature. The behaviour of this
    function is still evolving and the default behaviour and/or parameters may change
    in future versions.

    """

    # value checks
    _check_dest_is_group(dest)

    # setup logging
    with _LogWriter(log) as log:

        # do the copying
        n_copied, n_skipped, n_bytes_copied = _copy(
            log, source, dest, name=name, root=True, shallow=shallow,
            without_attrs=without_attrs, if_exists=if_exists, dry_run=dry_run,
            **create_kws
        )

        # log a final message with a summary of what happened
        _log_copy_summary(log, dry_run, n_copied, n_skipped, n_bytes_copied)

        return n_copied, n_skipped, n_bytes_copied


def _copy(log, source, dest, name, root, shallow, without_attrs, if_exists,
          dry_run, **create_kws):
    # N.B., if this is a dry run, dest may be None

    # setup counting variables
    n_copied = n_skipped = n_bytes_copied = 0

    # are we copying to/from h5py?
    source_h5py = source.__module__.startswith('h5py.')
    dest_h5py = dest is not None and dest.__module__.startswith('h5py.')

    # check if_exists parameter
    valid_if_exists = ['raise', 'replace', 'skip', 'skip_initialized']
    if if_exists not in valid_if_exists:
        raise ValueError('if_exists must be one of {!r}; found {!r}'
                         .format(valid_if_exists, if_exists))
    if dest_h5py and if_exists == 'skip_initialized':
        raise ValueError('{!r} can only be used when copying to zarr'
                         .format(if_exists))

    # determine name to copy to
    if name is None:
        name = source.name.split('/')[-1]
        if not name:
            # this can happen if source is the root group
            raise TypeError('source has no name, please provide the `name` '
                            'parameter to indicate a name to copy to')

    if hasattr(source, 'shape'):
        # copy a dataset/array

        # check if already exists, decide what to do
        do_copy = True
        exists = dest is not None and name in dest
        if exists:
            if if_exists == 'raise':
                raise CopyError('an object {!r} already exists in destination '
                                '{!r}'.format(name, dest.name))
            elif if_exists == 'skip':
                do_copy = False
            elif if_exists == 'skip_initialized':
                ds = dest[name]
                if ds.nchunks_initialized == ds.nchunks:
                    do_copy = False

        # take action
        if do_copy:

            # log a message about what we're going to do
            log('copy {} {} {}'.format(source.name, source.shape, source.dtype))

            if not dry_run:

                # clear the way
                if exists:
                    del dest[name]

                # setup creation keyword arguments
                kws = create_kws.copy()

                # setup chunks option, preserve by default
                kws.setdefault('chunks', source.chunks)

                # setup compression options
                if source_h5py:
                    if dest_h5py:
                        # h5py -> h5py; preserve compression options by default
                        kws.setdefault('compression', source.compression)
                        kws.setdefault('compression_opts', source.compression_opts)
                        kws.setdefault('shuffle', source.shuffle)
                        kws.setdefault('fletcher32', source.fletcher32)
                        kws.setdefault('fillvalue', source.fillvalue)
                    else:
                        # h5py -> zarr; use zarr default compression options
                        kws.setdefault('fill_value', source.fillvalue)
                else:
                    if dest_h5py:
                        # zarr -> h5py; use some vaguely sensible defaults
                        kws.setdefault('chunks', True)
                        kws.setdefault('compression', 'gzip')
                        kws.setdefault('compression_opts', 1)
                        kws.setdefault('shuffle', False)
                        kws.setdefault('fillvalue', source.fill_value)
                    else:
                        # zarr -> zarr; preserve compression options by default
                        kws.setdefault('compressor', source.compressor)
                        kws.setdefault('filters', source.filters)
                        kws.setdefault('order', source.order)
                        kws.setdefault('fill_value', source.fill_value)

                # create new dataset in destination
                ds = dest.create_dataset(name, shape=source.shape,
                                         dtype=source.dtype, **kws)

                # copy data - N.B., go chunk by chunk to avoid loading
                # everything into memory
                shape = ds.shape
                chunks = ds.chunks
                chunk_offsets = [range(0, s, c) for s, c in zip(shape, chunks)]
                for offset in itertools.product(*chunk_offsets):
                    sel = tuple(slice(o, min(s, o + c))
                                for o, s, c in zip(offset, shape, chunks))
                    ds[sel] = source[sel]
                n_bytes_copied += ds.size * ds.dtype.itemsize

                # copy attributes
                if not without_attrs:
                    ds.attrs.update(source.attrs)

            n_copied += 1

        else:
            log('skip {} {} {}'.format(source.name, source.shape, source.dtype))
            n_skipped += 1

    elif root or not shallow:
        # copy a group

        # check if an array is in the way
        do_copy = True
        exists_array = (dest is not None and
                        name in dest and
                        hasattr(dest[name], 'shape'))
        if exists_array:
            if if_exists == 'raise':
                raise CopyError('an array {!r} already exists in destination '
                                '{!r}'.format(name, dest.name))
            elif if_exists == 'skip':
                do_copy = False

        # take action
        if do_copy:

            # log action
            log('copy {}'.format(source.name))

            if not dry_run:

                # clear the way
                if exists_array:
                    del dest[name]

                # require group in destination
                grp = dest.require_group(name)

                # copy attributes
                if not without_attrs:
                    grp.attrs.update(source.attrs)

            else:

                # setup for dry run without creating any groups in the
                # destination
                if dest is not None:
                    grp = dest.get(name, None)
                else:
                    grp = None

            # recurse
            for k in source.keys():
                c, s, b = _copy(
                    log, source[k], grp, name=k, root=False, shallow=shallow,
                    without_attrs=without_attrs, if_exists=if_exists,
                    dry_run=dry_run, **create_kws)
                n_copied += c
                n_skipped += s
                n_bytes_copied += b

            n_copied += 1

        else:
            log('skip {}'.format(source.name))
            n_skipped += 1

    return n_copied, n_skipped, n_bytes_copied


def copy_all(source, dest, shallow=False, without_attrs=False, log=None,
             if_exists='raise', dry_run=False, **create_kws):
    """Copy all children of the `source` group into the `dest` group.

    Parameters
    ----------
    source : group or array/dataset
        A zarr group or array, or an h5py group or dataset.
    dest : group
        A zarr or h5py group.
    shallow : bool, optional
        If True, only copy immediate children of `source`.
    without_attrs : bool, optional
        Do not copy user attributes.
    log : callable, file path or file-like object, optional
        If provided, will be used to log progress information.
    if_exists : {'raise', 'replace', 'skip', 'skip_initialized'}, optional
        How to handle arrays that already exist in the destination group. If
        'raise' then a CopyError is raised on the first array already present
        in the destination group. If 'replace' then any array will be
        replaced in the destination. If 'skip' then any existing arrays will
        not be copied. If 'skip_initialized' then any existing arrays with
        all chunks initialized will not be copied (not available when copying to
        h5py).
    dry_run : bool, optional
        If True, don't actually copy anything, just log what would have
        happened.
    **create_kws
        Passed through to the create_dataset method when copying an
        array/dataset.

    Returns
    -------
    n_copied : int
        Number of items copied.
    n_skipped : int
        Number of items skipped.
    n_bytes_copied : int
        Number of bytes of data that were actually copied.

    Examples
    --------
    >>> import h5py
    >>> import zarr
    >>> import numpy as np
    >>> source = h5py.File('data/example.h5', mode='w')
    >>> foo = source.create_group('foo')
    >>> baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
    >>> spam = source.create_dataset('spam', data=np.arange(100, 200), chunks=(30,))
    >>> zarr.tree(source)
    /
     ├── foo
     │   └── bar
     │       └── baz (100,) int64
     └── spam (100,) int64
    >>> dest = zarr.group()
    >>> import sys
    >>> zarr.copy_all(source, dest, log=sys.stdout)
    copy /foo
    copy /foo/bar
    copy /foo/bar/baz (100,) int64
    copy /spam (100,) int64
    all done: 4 copied, 0 skipped, 1,600 bytes copied
    (4, 0, 1600)
    >>> dest.tree()
    /
     ├── foo
     │   └── bar
     │       └── baz (100,) int64
     └── spam (100,) int64
    >>> source.close()

    Notes
    -----
    Please note that this is an experimental feature. The behaviour of this
    function is still evolving and the default behaviour and/or parameters may change
    in future versions.

    """

    # value checks
    _check_dest_is_group(dest)

    # setup counting variables
    n_copied = n_skipped = n_bytes_copied = 0

    # setup logging
    with _LogWriter(log) as log:

        for k in source.keys():
            c, s, b = _copy(
                log, source[k], dest, name=k, root=False, shallow=shallow,
                without_attrs=without_attrs, if_exists=if_exists,
                dry_run=dry_run, **create_kws)
            n_copied += c
            n_skipped += s
            n_bytes_copied += b

        # log a final message with a summary of what happened
        _log_copy_summary(log, dry_run, n_copied, n_skipped, n_bytes_copied)

    return n_copied, n_skipped, n_bytes_copied
