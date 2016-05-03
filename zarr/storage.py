# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from collections import MutableMapping
import os
import tempfile


class DirectoryStore(MutableMapping):
    """Mutable Mapping interface to a directory. Keys must be strings,
    values must be bytes.

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

        # guard conditions
        if not isinstance(value, bytes):
            raise ValueError('value must be of type bytes')

        # write to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False,
                                         dir=self.path,
                                         prefix=key + '.',
                                         suffix='.partial') as f:
            f.write(value)
            temp_path = f.name

        # move temporary file into place
        os.rename(temp_path, os.path.join(self.path, key))

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
