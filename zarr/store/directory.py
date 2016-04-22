# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import os
import shutil


import numpy as np


from zarr.store.base import ArrayStore
from zarr.compat import PY2
from zarr.mappings import DirectoryMap, JSONFileMap


METAPATH = '__zmeta__'
DATAPATH = '__zdata__'
ATTRPATH = '__zattr__'


def _create_directory_store(path, shape, chunks, dtype=None, cname=None,
                            clevel=None, shuffle=None, fill_value=None):

        # create directories
        data_path = os.path.join(path, DATAPATH)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # setup meta
        meta_path = os.path.join(path, METAPATH)
        meta = JSONMetadata(
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
        data = DirectoryMap(data_path)

        # setup attrs
        attrs = JSONFileMap(os.path.join(path, ATTRPATH))

        return meta, data, attrs


def _open_directory_store(path, mode):

    # setup metadata
    meta = JSONMetadata(os.path.join(path, METAPATH))

    # setup data
    data = DirectoryMap(os.path.join(path, DATAPATH))

    # setup attrs
    attrs = JSONFileMap(os.path.join(path, ATTRPATH),
                        read_only=(mode == 'r'))

    return meta, data, attrs


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
        super(DirectoryStore, self).__init__(meta, data, attrs)


class JSONMetadata(JSONFileMap):
    
    def __init__(self, path, **kwargs):
        kwargs = {key: json_encode_metadata(key, value)
                  for key, value in kwargs.items()}
        super(JSONMetadata, self).__init__(path, **kwargs)

    def __getitem__(self, key):
        value = super(JSONMetadata, self).__getitem__(key)
        return json_decode_metadata(key, value)

    def __setitem__(self, key, value):
        value = json_encode_metadata(key, value)
        super(JSONMetadata, self).__setitem__(key, value)
    
    
def json_encode_metadata(key, value):
    if not PY2 and key == 'cname':
        value = str(value, 'ascii')
    if key == 'dtype':
        value = json_encode_dtype(value)
    return value


def json_decode_metadata(key, value):
    if key in ('shape', 'chunks'):
        value = tuple(value)
    elif key == 'cname':
        value = value.encode('ascii')
    elif key == 'dtype':
        return json_decode_dtype(value)
    return value


def json_encode_dtype(d):
    if d.fields is None:
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
