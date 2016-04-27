# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
import sys


import numpy as np


PY2 = sys.version_info[0] == 2


def decode_metadata(b):
    meta = json.loads(b)
    meta = dict(
        shape=tuple(meta['shape']),
        chunks=tuple(meta['chunks']),
        dtype=decode_dtype(meta['dtype']),
        compression=meta['compression'],
        compression_opts=meta['compression_opts'],
        fill_value=meta['fill_value'],
    )
    return meta


def encode_metadata(meta):
    meta = dict(
        shape=meta['shape'],
        chunks=meta['chunks'],
        dtype=encode_dtype(meta['dtype']),
        compression=meta['compression'],
        compression_opts=meta['compression_opts'],
        fill_value=meta['fill_value'],
    )
    return json.dumps(meta, indent=4, sort_keys=True)


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
            d = [(f.encode('ascii'), _decode_dtype_descr(v)) 
                 for f, v in d]
        else:
            d = [(f, _decode_dtype_descr(v)) for f, v in d]
    return d


def decode_dtype(d):
    d = _decode_dtype_descr(d)
    return np.dtype(d)
