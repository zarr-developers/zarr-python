# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import json
import sys
import numpy as np

from zarr import defaults as _defaults


PY2 = sys.version_info[0] == 2


def loads(b):
    meta = json.loads(b)
    if meta['compression'] not in ('blosc', None):
        raise NotImplementedError("Only Blosc and no-compression are supported")
    meta = dict(
        shape=tuple(meta['shape']),
        chunks=tuple(meta['chunks']),
        dtype=decode_dtype(meta['dtype']),
        fill_value=meta['fill_value'],
        cname=meta['compression_opts']['cname'].encode('ascii'),
        clevel=meta['compression_opts']['clevel'],
        shuffle=meta['compression_opts']['shuffle']
    )

    return meta


def dumps(meta):
    meta = dict(
        shape=meta['shape'],
        chunks=meta['chunks'],
        dtype=encode_dtype(meta['dtype']),
        fill_value=meta['fill_value'],
        compression='blosc',
        compression_opts=dict(
            cname=meta['cname'] if PY2 else str(meta['cname'], 'ascii'),
            clevel=meta['clevel'],
            shuffle=meta['shuffle'])
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
            d = [(f.encode('ascii'), _decode_dtype_descr(v)) for f, v in d]
        else:
            d = [(f, _decode_dtype_descr(v)) for f, v in d]
    return d


def decode_dtype(d):
    d = _decode_dtype_descr(d)
    return np.dtype(d)
