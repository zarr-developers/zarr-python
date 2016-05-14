# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping
import os
import tempfile
import json


import numpy as np


from zarr.util import normalize_shape, normalize_chunks, normalize_order
from zarr.compressors import get_compressor_cls
from zarr.meta import encode_metadata


def init_store(store, shape, chunks, dtype=None, compression='default',
               compression_opts=None, fill_value=None,
               order='C', overwrite=False):
    """Initialise an array store with the given configuration.

    Parameters
    ----------
    store : MutableMapping
        A mapping that supports string keys and byte sequence values.
    shape : int or tuple of ints
        Array shape.
    chunks : int or tuple of ints
        Chunk shape.
    dtype : string or dtype, optional
        NumPy dtype.
    compression : string, optional
        Name of primary compression library, e.g., 'blosc', 'zlib', 'bz2',
        'lzma'.
    compression_opts : object, optional
        Options to primary compressor. E.g., for blosc, provide a dictionary
        with keys 'cname', 'clevel' and 'shuffle'.
    fill_value : object
        Default value to use for uninitialised portions of the array.
    order : {'C', 'F'}, optional
        Memory layout to be used within each chunk.
    overwrite : bool, optional
        If True, erase all data in `store` prior to initialisation.

    Examples
    --------
    >>> import zarr
    >>> store = dict()
    >>> zarr.init_store(store, shape=(10000, 10000), chunks=(1000, 1000))
    >>> sorted(store.keys())
    ['attrs', 'meta']
    >>> print(str(store['meta'], 'ascii'))
    {
        "chunks": [
            1000,
            1000
        ],
        "compression": "blosc",
        "compression_opts": {
            "clevel": 5,
            "cname": "blosclz",
            "shuffle": 1
        },
        "dtype": "<f8",
        "fill_value": null,
        "order": "C",
        "shape": [
            10000,
            10000
        ],
        "zarr_format": 1
    }
    >>> print(str(store['attrs'], 'ascii'))
    {}

    Notes
    -----
    The initialisation process involves normalising all array metadata,
    encoding as JSON and storing under the 'meta' key. User attributes are also
    initialised and stored as JSON under the 'attrs' key.

    """

    # guard conditions
    empty = len(store) == 0
    if not empty and not overwrite:
        raise ValueError('store is not empty')

    # normalise metadata
    shape = normalize_shape(shape)
    chunks = normalize_chunks(chunks, shape)
    dtype = np.dtype(dtype)
    compressor_cls = get_compressor_cls(compression)
    compression = compressor_cls.canonical_name
    compression_opts = compressor_cls.normalize_opts(
        compression_opts
    )
    order = normalize_order(order)

    # delete any pre-existing items in store
    store.clear()

    # initialise metadata
    meta = dict(shape=shape, chunks=chunks, dtype=dtype,
                compression=compression, compression_opts=compression_opts,
                fill_value=fill_value, order=order)
    store['meta'] = encode_metadata(meta)

    # initialise attributes
    store['attrs'] = json.dumps(dict()).encode('ascii')


class DirectoryStore(MutableMapping):
    """Mutable Mapping interface to a directory. Keys must be strings,
    values must be bytes-like objects.

    Parameters
    ----------
    path : string
        Location of directory.

    Examples
    --------
    >>> import zarr
    >>> store = zarr.DirectoryStore('example.zarr')
    >>> zarr.init_store(store, shape=(10000, 10000), chunks=(1000, 1000),
    ...                 fill_value=0, overwrite=True)
    >>> import os
    >>> sorted(os.listdir('example.zarr'))
    ['attrs', 'meta']
    >>> print(open('example.zarr/meta').read())
    {
        "chunks": [
            1000,
            1000
        ],
        "compression": "blosc",
        "compression_opts": {
            "clevel": 5,
            "cname": "blosclz",
            "shuffle": 1
        },
        "dtype": "<f8",
        "fill_value": 0,
        "order": "C",
        "shape": [
            10000,
            10000
        ],
        "zarr_format": 1
    }
    >>> print(open('example.zarr/attrs').read())
    {}
    >>> z = zarr.Array(store)
    >>> z
    zarr.core.Array((10000, 10000), float64, chunks=(1000, 1000), order=C)
      compression: blosc; compression_opts: {'clevel': 5, 'cname': 'blosclz', 'shuffle': 1}
      nbytes: 762.9M; nbytes_stored: 317; ratio: 2523659.3; initialized: 0/100
      store: zarr.storage.DirectoryStore
    >>> z[:] = 1
    >>> len(os.listdir('example.zarr'))
    102
    >>> sorted(os.listdir('example.zarr'))[0:5]
    ['0.0', '0.1', '0.2', '0.3', '0.4']
    >>> print(open('example.zarr/0.0', 'rb').read(10))
    b'\\x02\\x01\\x01\\x08\\x00\\x12z\\x00\\x00\\x80'

    See Also
    --------
    zarr.creation.open

    """  # flake8: noqa

    def __init__(self, path):

        # guard conditions
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise ValueError('path does not exist')
        elif not os.path.isdir(path):
            raise ValueError('path is not a directory')

        self.path = path

    def __getitem__(self, key):

        # guard conditions
        if key not in self:
            raise KeyError(key)

        with open(os.path.join(self.path, key), 'rb') as f:
            return f.read()

    def __setitem__(self, key, value):
        # accept any value that can be written to a file

        # destination path for key
        dest_path = os.path.join(self.path, key)

        # write to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir=self.path,
                                         prefix=key + '.',
                                         suffix='.partial') as f:
            f.write(value)
            temp_path = f.name

        # move temporary file into place
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)

    def __delitem__(self, key):

        # guard conditions
        if key not in self:
            raise KeyError(key)

        os.remove(os.path.join(self.path, key))

    def __contains__(self, key):
        return os.path.isfile(os.path.join(self.path, key))

    def keys(self):
        for key in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, key)):
                yield key

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return sum(1 for _ in self.keys())

    @property
    def size(self):
        """Total size of all values in number of bytes."""
        return sum(os.path.getsize(os.path.join(self.path, key))
                   for key in self.keys())
