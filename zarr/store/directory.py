# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os
import shutil
import json


import numpy as np


from zarr.store.base import ArrayStore
from zarr.util import normalize_shape, normalize_chunks, normalize_cparams
from zarr.compat import PY2
from zarr.mappings import frozendict, Directory


METAPATH = '__zmeta__'
DATAPATH = '__zdata__'
ATTRPATH = '__zattr__'


class DirectoryStore(ArrayStore):

    def __init__(self, path, mode='a', shape=None, chunks=None, dtype=None,
                 cname=None, clevel=None, shuffle=None, fill_value=None):

        # use same mode semantics as h5py, although N.B., here `path` is a
        # directory:
        # r : readonly, must exist
        # r+ : read/write, must exist
        # w : create, delete if exists
        # w- or x : create, fail if exists
        # a : read/write if exists, create otherwise (default)

        # use metadata file as indicator of array existence
        meta_path = os.path.join(path, METAPATH)

        if mode in ['r', 'r+']:
            self._open(path)

        elif mode == 'w':
            if os.path.exists(path):
                shutil.rmtree(path)
            self._create(path, shape=shape, chunks=chunks, dtype=dtype,
                         cname=cname, clevel=clevel, shuffle=shuffle,
                         fill_value=fill_value)

        elif mode in ['w-', 'x']:
            if os.path.exists(meta_path):
                raise ValueError('array exists: %s' % path)
            self._create(path, shape=shape, chunks=chunks, dtype=dtype,
                         cname=cname, clevel=clevel, shuffle=shuffle,
                         fill_value=fill_value)

        elif mode == 'a':
            if os.path.exists(meta_path):
                self._open(path)
            else:
                self._create(path, shape=shape, chunks=chunks, dtype=dtype,
                             cname=cname, clevel=clevel, shuffle=shuffle,
                             fill_value=fill_value)

        else:
            raise ValueError('bad mode: %r' % mode)

        self._path = path
        self._mode = mode

    def _create(self, path, shape=None, chunks=None, dtype=None,
                cname=None, clevel=None, shuffle=None, fill_value=None):

        # create directories
        data_path = os.path.join(path, DATAPATH)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # normalize arguments
        shape = normalize_shape(shape)
        chunks = normalize_chunks(chunks, shape)
        dtype = np.dtype(dtype)
        cname, clevel, shuffle = normalize_cparams(cname, clevel, shuffle)

        # setup meta
        self._meta = frozendict(
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            fill_value=fill_value
        )

        # write metadata
        write_array_metadata(path, self._meta)

        # setup data
        self._data = Directory(data_path)

        # TODO setup attrs

    def _open(self, path):

        # read metadata
        self._meta = read_array_metadata(path)

        # setup data
        self._data = Directory(os.path.join(path, DATAPATH))

        # TODO setup attrs

    @property
    def meta(self):
        return self._meta

    @property
    def data(self):
        return self._data

    @property
    def attrs(self):
        # TODO
        pass

    @property
    def cbytes(self):
        return self._data.size()

    @property
    def initialized(self):
        return len(self._data)


def read_array_metadata(path):

    # check path exists
    if not os.path.exists(path):
        raise ValueError('path not found: %s' % path)

    # check metadata file
    meta_path = os.path.join(path, METAPATH)
    if not os.path.exists(meta_path):
        raise ValueError('array metadata not found: %s' % path)

    # read from file
    with open(meta_path) as f:
        meta = json.load(f)

    # decode some values
    meta['shape'] = tuple(meta['shape'])
    meta['chunks'] = tuple(meta['chunks'])
    meta['cname'] = meta['cname'].encode('ascii')
    meta['dtype'] = decode_dtype(meta['dtype'])

    return frozendict(meta)


def write_array_metadata(path, meta):

    # setup dictionary with values that can be serialized to json
    json_meta = dict(meta)
    if not PY2:
        json_meta['cname'] = str(meta['cname'], 'ascii')
    json_meta['dtype'] = encode_dtype(meta['dtype'])

    # write to file
    meta_path = os.path.join(path, METAPATH)
    with open(meta_path, 'w') as f:
        json.dump(json_meta, f, indent=4, sort_keys=True)


def encode_dtype(d):
    if d.fields is None:
        return d.str
    else:
        return d.descr


def _decode_dtype_descr(d):
    # need to convert list of lists to list of tuples
    if isinstance(d, list):
        # recurse to handle nested structures
        if PY2:
            # under PY2 numpy rejects unicode field names
            d = [(f.encode('ascii'), _decode_dtype_descr(v)) for f, v in d]
        else:
            d = [(f, _decode_dtype_descr(v)) for f, v in d]
    return d


def decode_dtype(d):
    d = _decode_dtype_descr(d)
    return np.dtype(d)
