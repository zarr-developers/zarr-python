# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
import math


from nose.tools import eq_ as eq, assert_is_none, assert_raises, assert_is
import numpy as np


from zarr.compat import PY2
from zarr.meta import decode_array_metadata, encode_dtype, decode_dtype, \
    ZARR_FORMAT, decode_group_metadata, encode_array_metadata
from zarr.errors import MetadataError


def test_encode_decode_array_1():

    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('f8'),
        compression='zlib',
        compression_opts=1,
        fill_value=None,
        order='C'
    )

    meta_bytes = '''{
    "chunks": [
        10
    ],
    "compression": "zlib",
    "compression_opts": 1,
    "dtype": "<f8",
    "fill_value": null,
    "order": "C",
    "shape": [
        100
    ],
    "zarr_format": %s
}''' % ZARR_FORMAT
    if not PY2:
        meta_bytes = meta_bytes.encode('ascii')

    # test encoding
    meta_enc = encode_array_metadata(meta)
    eq(meta_bytes, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_bytes)
    eq(ZARR_FORMAT, meta_dec['zarr_format'])
    eq(meta['shape'], meta_dec['shape'])
    eq(meta['chunks'], meta_dec['chunks'])
    eq(meta['dtype'], meta_dec['dtype'])
    eq(meta['compression'], meta_dec['compression'])
    eq(meta['compression_opts'], meta_dec['compression_opts'])
    eq(meta['order'], meta_dec['order'])
    assert_is_none(meta_dec['fill_value'])


def test_encode_decode_array_2():

    # some variations
    meta = dict(
        shape=(100, 100),
        chunks=(10, 10),
        dtype=np.dtype([('a', 'i4'), ('b', 'S10')]),
        compression='blosc',
        compression_opts=dict(cname='lz4', clevel=3, shuffle=2),
        fill_value=42,
        order='F'
    )

    meta_bytes = '''{
    "chunks": [
        10,
        10
    ],
    "compression": "blosc",
    "compression_opts": {
        "clevel": 3,
        "cname": "lz4",
        "shuffle": 2
    },
    "dtype": [
        [
            "a",
            "<i4"
        ],
        [
            "b",
            "|S10"
        ]
    ],
    "fill_value": 42,
    "order": "F",
    "shape": [
        100,
        100
    ],
    "zarr_format": %s
}''' % ZARR_FORMAT
    if not PY2:
        meta_bytes = meta_bytes.encode('ascii')

    # test encoding
    meta_enc = encode_array_metadata(meta)
    eq(meta_bytes, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_bytes)
    eq(ZARR_FORMAT, meta_dec['zarr_format'])
    eq(meta['shape'], meta_dec['shape'])
    eq(meta['chunks'], meta_dec['chunks'])
    eq(meta['dtype'], meta_dec['dtype'])
    eq(meta['compression'], meta_dec['compression'])
    eq(meta['compression_opts'], meta_dec['compression_opts'])
    eq(meta['order'], meta_dec['order'])
    eq(meta['fill_value'], meta_dec['fill_value'])


def test_encode_decode_array_nan_fill_value():

    for fill in math.nan, np.nan:

        meta = dict(
            shape=(100,),
            chunks=(10,),
            dtype=np.dtype('f8'),
            compression='zlib',
            compression_opts=1,
            fill_value=fill,
            order='C'
        )

        # test fill value round trip
        meta_enc = encode_array_metadata(meta)
        meta_dec = decode_array_metadata(meta_enc)
        actual = meta_dec['fill_value']
        print(repr(actual))
        print(type(actual))
        assert np.isnan(actual)


def test_decode_array_unsupported_format():

    # unsupported format
    meta_bytes = '''{
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
        meta_bytes = meta_bytes.encode('ascii')
    with assert_raises(MetadataError):
        decode_array_metadata(meta_bytes)


def test_decode_array_missing_fields():

    # missing fields
    meta_bytes = '''{
        "zarr_format": %s
    }''' % ZARR_FORMAT
    if not PY2:
        meta_bytes = meta_bytes.encode('ascii')
    with assert_raises(MetadataError):
        decode_array_metadata(meta_bytes)


def test_encode_decode_dtype():

    for dt in ['f8', [('a', 'f8')], [('a', 'f8'), ('b', 'i1')]]:
        e = encode_dtype(np.dtype(dt))
        s = json.dumps(e)  # check JSON serializable
        o = json.loads(s)
        d = decode_dtype(o)
        eq(np.dtype(dt), d)


def test_decode_group():

    # typical
    b = '''{
        "zarr_format": %s
    }''' % ZARR_FORMAT
    if not PY2:
        b = b.encode('ascii')
    meta = decode_group_metadata(b)
    eq(ZARR_FORMAT, meta['zarr_format'])

    # unsupported format
    b = '''{
        "zarr_format": %s
    }''' % (ZARR_FORMAT - 1)
    if not PY2:
        b = b.encode('ascii')
    with assert_raises(MetadataError):
        decode_group_metadata(b)
