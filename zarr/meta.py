# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import os
import json
import ast
import numpy as np


from zarr import defaults as _defaults


def read_array_metadata(path):

    # check path exists
    if not os.path.exists(path):
        raise ValueError('path not found: %s' % path)

    # check metadata file
    meta_path = os.path.join(path, _defaults.metapath)
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

    return meta


def write_array_metadata(path, shape, chunks, dtype, cname, clevel, shuffle,
                         fill_value):

    # construct metadata dictionary
    meta = dict(
        shape=shape,
        chunks=chunks,
        dtype=encode_dtype(dtype),
        cname=str(cname, 'ascii'),
        clevel=clevel,
        shuffle=shuffle,
        fill_value=fill_value,
    )

    # write to file
    meta_path = os.path.join(path, _defaults.metapath)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4, sort_keys=True)


def encode_dtype(d):
    if d.fields is None:
        return d.str
    else:
        return str(d)


def decode_dtype(s):
    try:
        return np.dtype(s)
    except ValueError:
        return np.dtype(ast.literal_eval(s))
