# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os
import shutil


import numpy as np


from zarr.store.base import ArrayStore
from zarr.util import normalize_shape, normalize_chunks, normalize_cparams
from zarr.compat import PY2
from zarr.mappings import Directory, JSONFile


METAPATH = '__zmeta__'
DATAPATH = '__zdata__'
ATTRPATH = '__zattr__'


class DirectoryStore(ArrayStore):

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
        self._mode = mode

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
            meta, data, attrs = self._create(
                path, shape=shape, chunks=chunks, dtype=dtype, cname=cname,
                clevel=clevel, shuffle=shuffle, fill_value=fill_value
            )
        else:
            meta, data, attrs = self._open(path)
        super(DirectoryStore, self).__init__(meta, data, attrs)

        # setup additional members
        self._path = path
        self._mode = mode

    def _create(self, path, shape, chunks, dtype=None, cname=None,
                clevel=None, shuffle=None, fill_value=None):

        # create directories
        data_path = os.path.join(path, DATAPATH)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # normalize arguments now to avoid any encoding issues
        shape = normalize_shape(shape)
        chunks = normalize_chunks(chunks, shape)
        dtype = np.dtype(dtype)
        cname, clevel, shuffle = normalize_cparams(cname, clevel, shuffle)

        # setup meta
        meta_path = os.path.join(path, METAPATH)
        meta = MetadataJSONFile(
            meta_path,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            cname=cname,
            clevel=clevel,
            shuffle=shuffle,
            fill_value=fill_value
        )

        # setup data
        data = Directory(data_path)

        # setup attrs
        attrs = JSONFile(os.path.join(path, ATTRPATH))

        return meta, data, attrs

    def _open(self, path):

        # read metadata
        meta = MetadataJSONFile(os.path.join(path, METAPATH))

        # setup data
        data = Directory(os.path.join(path, DATAPATH))

        # setup attrs
        attrs = JSONFile(os.path.join(path, ATTRPATH),
                         read_only=(self._mode == 'r'))

        return meta, data, attrs


class MetadataJSONFile(JSONFile):
    
    def __init__(self, path, **kwargs):
        kwargs = {key: encode_metadata(key, value)
                  for key, value in kwargs.items()}
        super(MetadataJSONFile, self).__init__(path, **kwargs)

    def __getitem__(self, key):
        value = super(MetadataJSONFile, self).__getitem__(key)
        return decode_metadata(key, value)

    def __setitem__(self, key, value):
        value = encode_metadata(key, value)
        super(MetadataJSONFile, self).__setitem__(key, value)
    
    
def encode_metadata(key, value):
    if not PY2 and key == 'cname':
        value = str(value, 'ascii')
    if key == 'dtype':
        value = encode_dtype(value)
    return value


def decode_metadata(key, value):
    if key in ('shape', 'chunks'):
        value = tuple(value)
    elif key == 'cname':
        value = value.encode('ascii')
    elif key == 'dtype':
        return decode_dtype(value)
    return value


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
