# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json


from nose.tools import eq_ as eq, assert_is_none, assert_raises
import numpy as np


from zarr.compat import PY2
from zarr.meta import decode_array_metadata, encode_dtype, decode_dtype, \
    ZARR_FORMAT
from zarr.errors import MetadataError


def test_decode_array():

    # typical
    b = '''{
        "zarr_format": %s,
        "shape": [100],
        "chunks": [10],
        "dtype": "<f8",
        "compression": "zlib",
        "compression_opts": 1,
        "fill_value": null,
        "order": "C"
    }''' % ZARR_FORMAT
    if not PY2:
        b = b.encode('ascii')
    meta = decode_array_metadata(b)
    eq(ZARR_FORMAT, meta['zarr_format'])
    eq((100,), meta['shape'])
    eq((10,), meta['chunks'])
    eq(np.dtype('<f8'), meta['dtype'])
    eq('zlib', meta['compression'])
    eq(1, meta['compression_opts'])
    assert_is_none(meta['fill_value'])
    eq('C', meta['order'])

    # variations
    b = '''{
        "zarr_format": %s,
        "shape": [100, 100],
        "chunks": [10, 10],
        "dtype": [["a", "i4"], ["b", "S10"]],
        "compression": "blosc",
        "compression_opts": {
            "cname": "lz4",
            "clevel": 3,
            "shuffle": 2
        },
        "fill_value": 42,
        "order": "F"
    }''' % ZARR_FORMAT
    if not PY2:
        b = b.encode('ascii')
    meta = decode_array_metadata(b)
    eq(ZARR_FORMAT, meta['zarr_format'])
    eq((100, 100), meta['shape'])
    eq((10, 10), meta['chunks'])
    # check structured dtype
    eq(np.dtype([('a', 'i4'), ('b', 'S10')]), meta['dtype'])
    # check structured compression_opts
    eq(dict(cname='lz4', clevel=3, shuffle=2), meta['compression_opts'])
    # check fill value
    eq(42, meta['fill_value'])
    eq('F', meta['order'])

    # unsupported format
    b = '''{
        "zarr_format": %s,
        "shape": [100],
        "chunks": [10],
        "dtype": "<f8",
        "compression": "zlib",
        "compression_opts": 1,
        "fill_value": null,
        "order": "C"
    }''' % (ZARR_FORMAT - 1)
    if not PY2:
        b = b.encode('ascii')
    with assert_raises(MetadataError):
        decode_array_metadata(b)

    # missing fields
    b = '''{
        "zarr_format": %s
    }''' % ZARR_FORMAT
    if not PY2:
        b = b.encode('ascii')
    with assert_raises(MetadataError):
        decode_array_metadata(b)


def test_encode_decode_dtype():

    for dt in ['f8', [('a', 'f8')], [('a', 'f8'), ('b', 'i1')]]:
        e = encode_dtype(np.dtype(dt))
        s = json.dumps(e)  # check JSON serializable
        o = json.loads(s)
        d = decode_dtype(o)
        eq(np.dtype(dt), d)
