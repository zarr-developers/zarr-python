# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import json
import sys
import numpy as np

from zarr import defaults as _defaults


PY2 = sys.version_info[0] == 2


def loads(b):
    meta = json.loads(b)

    # decode some values
    meta['shape'] = tuple(meta['shape'])
    meta['chunks'] = tuple(meta['chunks'])
    meta['cname'] = meta['cname'].encode('ascii')
    meta['dtype'] = decode_dtype(meta['dtype'])

    return meta


def dumps(meta):
    # construct metadata dictionary
    meta = dict(
        shape=meta['shape'],
        chunks=meta['chunks'],
        dtype=encode_dtype(meta['dtype']),
        cname=meta['cname'] if PY2 else str(meta['cname'], 'ascii'),
        clevel=meta['clevel'],
        shuffle=meta['shuffle'],
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
            d = [(f.encode('ascii'), _decode_dtype_descr(v)) for f, v in d]
        else:
            d = [(f, _decode_dtype_descr(v)) for f, v in d]
    return d


def decode_dtype(d):
    d = _decode_dtype_descr(d)
    return np.dtype(d)
