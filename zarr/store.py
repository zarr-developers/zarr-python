# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import json
import os
import shutil

import numpy as np


from zarr.util import normalize_shape, normalize_chunks, normalize_cparams, \
    normalize_resize_args
from zarr.mappings import JSONFileMap, DirectoryMap
from zarr.compat import PY2, itervalues


class ArrayStore(object):
    """Storage for a single array.

    Parameters
    ----------
    meta : MutableMapping
        Array configuration metadata. Must contain at least 'shape' and
        'chunks'.
    data : MutableMapping
        Holds a mapping from chunk indices to compressed chunk data.
    attrs : MutableMapping
        Holds user-defined attributes.

    Examples
    --------
    >>> import zarr
    >>> meta = dict(shape=(100,), chunks=(10,))
    >>> data = dict()
    >>> attrs = dict()
    >>> store = zarr.ArrayStore(meta, data, attrs)
    >>> z = zarr.Array(store)
    >>> meta['dtype']
    dtype('float64')
    >>> z[:] = 42
    >>> sorted(data.keys())
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    """

    def __init__(self, meta=None, data=None, attrs=None, read_only=False,
                 **extra_meta):
        if data is None:
            data = dict()
        if attrs is None:
            attrs = dict()
        if 'attrs' in data:
            for k, v in json.loads(data['attrs']).items():
                if k not in attrs:
                    attrs[k] = v
        if meta is None:
            meta = dict()
        if 'meta' in data:
            from .meta import loads
            for k, v in loads(data['meta']).items():
                if k not in meta:
                    meta[k] = v
        meta.update(extra_meta)

        # normalize configuration metadata
        shape = normalize_shape(meta['shape'])
        chunks = normalize_chunks(meta['chunks'], shape)
        dtype = np.dtype(meta.get('dtype', None))  # float64 default
        cname, clevel, shuffle = \
            normalize_cparams(meta.get('cname', None),
                              meta.get('clevel', None),
                              meta.get('shuffle', None))
        fill_value = meta.get('fill_value', None)
        meta.update(shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                    clevel=clevel, shuffle=shuffle, fill_value=fill_value)

        # setup members
        self.meta = meta
        self.data = data
        self.attrs = attrs
        self.read_only = read_only

    @property
    def cbytes(self):
        """The total number of bytes of data held for the array."""

        if hasattr(self.data, 'size'):
            # pass through
            return self.data.size
        elif isinstance(self.data, dict):
            # cheap to compute by summing length of values
            return sum(len(v) for v in itervalues(self.data))
        else:
            return -1

    @property
    def initialized(self):
        """The number of chunks that have been initialized."""
        return len(self.data)

    def resize(self, *args):
        """Resize the array."""

        # normalize new shape argument
        old_shape = self.meta['shape']
        new_shape = normalize_resize_args(old_shape, *args)

        # determine the new number and arrangement of chunks
        chunks = self.meta['chunks']
        new_cdata_shape = tuple(int(np.ceil(s / c))
                                for s, c in zip(new_shape, chunks))

        # remove any chunks not within range
        for ckey in list(self.data):
            cidx = map(int, ckey.split('.'))
            if all(i < c for i, c in zip(cidx, new_cdata_shape)):
                pass  # keep the chunk
            else:
                del self.data[ckey]

        # update metadata
        self.meta['shape'] = new_shape
        self.flush()

    def flush(self):
        """ Encode and flush metadata and attrs to data """
        from . import meta
        self.data['meta'] = meta.dumps(self.meta)
        self.data['attrs'] = json.dumps(self.attrs, indent=4, sort_keys=True)


class MemoryStore(ArrayStore):
    """Array store using dictionaries."""

    def __init__(self, shape, chunks, dtype=None, cname=None, clevel=None,
                 shuffle=None, fill_value=None):
        meta = dict(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            fill_value=fill_value
        )
        data = dict()
        attrs = dict()
        super(MemoryStore, self).__init__(meta, data, attrs)

    def flush(self):
        pass


METAPATH = '__zmeta__'
DATAPATH = '__zdata__'
ATTRPATH = '__zattr__'


class DirectoryStore(ArrayStore):
    """Array store using a file system."""

    def __init__(self, path, mode='a', shape=None, chunks=None, dtype=None,
                 cname=None, clevel=None, shuffle=None, fill_value=None):

        # use metadata file as indicator of array existence
        meta_path = os.path.join(path, METAPATH)

        # use same mode semantics as h5py, although N.B., here `path` is a
        # directory:
        # r : readonly, must exist
        # r+ : read/write, must exist
        # w : create, delete if exists
        # w- or x : create, fail if exists
        # a : read/write if exists, create otherwise (default)

        # handle mode
        create = False
        if mode in ['r', 'r+']:
            pass
        elif mode == 'w':
            if os.path.exists(path):
                shutil.rmtree(path)
            create = True
        elif mode in ['w-', 'x']:
            if os.path.exists(meta_path):
                raise ValueError('array exists: %s' % path)
            create = True
        elif mode == 'a':
            if not os.path.exists(meta_path):
                create = True
        else:
            raise ValueError('bad mode: %r' % mode)

        # instantiate
        if create:
            meta, data, attrs = _create_directory_store(
                path, shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                clevel=clevel, shuffle=shuffle, fill_value=fill_value
            )
        else:
            meta, data, attrs = _open_directory_store(path, mode)
        read_only = mode == 'r'
        super(DirectoryStore, self).__init__(meta, data, attrs,
                                             read_only=read_only)


def _create_directory_store(path, shape, chunks, dtype=None, cname=None,
                            clevel=None, shuffle=None, fill_value=None):

        # create directories
        data_path = os.path.join(path, DATAPATH)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # setup metadata
        meta = MetadataJSONFile(os.path.join(path, METAPATH))
        meta.update(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            fill_value=fill_value
        )

        # setup data
        data = DirectoryMap(data_path)

        # setup attrs
        attrs = JSONFileMap(os.path.join(path, ATTRPATH))

        return meta, data, attrs


def _open_directory_store(path, mode):

    read_only = (mode == 'r')

    # setup metadata
    meta = MetadataJSONFile(os.path.join(path, METAPATH))

    # setup data
    data = DirectoryMap(os.path.join(path, DATAPATH), read_only=read_only)

    # setup attrs
    attrs = JSONFileMap(os.path.join(path, ATTRPATH), read_only=read_only)

    return meta, data, attrs


class MetadataJSONFile(JSONFileMap):

    def __init__(self, path):
        super(MetadataJSONFile, self).__init__(path)

    def __getitem__(self, key):
        value = super(MetadataJSONFile, self).__getitem__(key)
        return json_decode_metadata(key, value)

    def __setitem__(self, key, value):
        value = json_encode_metadata(key, value)
        super(MetadataJSONFile, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        d = dict()
        d.update(*args, **kwargs)
        for key, value in d.items():
            d[key] = json_encode_metadata(key, value)
        super(MetadataJSONFile, self).update(d)


def json_encode_metadata(key, value):
    if value is None:
        pass
    elif not PY2 and key == 'cname' and isinstance(value, bytes):
        value = str(value, 'ascii')
    elif key == 'dtype':
        value = json_encode_dtype(value)
    return value


def json_decode_metadata(key, value):
    if value is None:
        pass
    elif key in ('shape', 'chunks'):
        if isinstance(value, list):
            value = tuple(value)
    elif key == 'cname':
        value = value.encode('ascii')
    elif key == 'dtype':
        return json_decode_dtype(value)
    return value


def json_encode_dtype(d):
    if not isinstance(d, np.dtype):
        return d
    elif d.fields is None:
        return d.str
    else:
        return d.descr


def _json_decode_dtype_descr(d):
    # need to convert list of lists to list of tuples
    if isinstance(d, list):
        # recurse to handle nested structures
        if PY2:
            # under PY2 numpy rejects unicode field names
            d = [(f.encode('ascii'), _json_decode_dtype_descr(v)) for f, v in d]
        else:
            d = [(f, _json_decode_dtype_descr(v)) for f, v in d]
    return d


def json_decode_dtype(d):
    d = _json_decode_dtype_descr(d)
    return np.dtype(d)
