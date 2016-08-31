# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json


from nose.tools import eq_ as eq, assert_is_none, assert_raises
import numpy as np


from zarr.compat import binary_type, text_type
from zarr.meta import decode_array_metadata, encode_dtype, decode_dtype, \
    ZARR_FORMAT, decode_group_metadata, encode_array_metadata
from zarr.errors import MetadataError
from zarr.filters import DeltaFilter


def assert_json_eq(expect, actual):  # pragma: no cover
    if isinstance(expect, binary_type):
        expect = text_type(expect, 'ascii')
    if isinstance(actual, binary_type):
        actual = text_type(actual, 'ascii')
    ej = json.loads(expect)
    aj = json.loads(actual)
    eq(ej, aj)


def test_encode_decode_array_1():

    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('f8'),
        compression='zlib',
        compression_opts=1,
        fill_value=None,
        filters=None,
        order='C'
    )

    meta_json = '''{
        "chunks": [10],
        "compression": "zlib",
        "compression_opts": 1,
        "dtype": "<f8",
        "fill_value": null,
        "filters": null,
        "order": "C",
        "shape": [100],
        "zarr_format": %s
    }''' % ZARR_FORMAT

    # test encoding
    meta_enc = encode_array_metadata(meta)
    assert_json_eq(meta_json, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_enc)
    eq(ZARR_FORMAT, meta_dec['zarr_format'])
    eq(meta['shape'], meta_dec['shape'])
    eq(meta['chunks'], meta_dec['chunks'])
    eq(meta['dtype'], meta_dec['dtype'])
    eq(meta['compression'], meta_dec['compression'])
    eq(meta['compression_opts'], meta_dec['compression_opts'])
    eq(meta['order'], meta_dec['order'])
    assert_is_none(meta_dec['fill_value'])
    assert_is_none(meta_dec['filters'])


def test_encode_decode_array_2():

    # some variations
    df = DeltaFilter(astype='u2', dtype='V14')
    meta = dict(
        shape=(100, 100),
        chunks=(10, 10),
        dtype=np.dtype([('a', 'i4'), ('b', 'S10')]),
        compression='blosc',
        compression_opts=dict(cname='lz4', clevel=3, shuffle=2),
        fill_value=42,
        order='F',
        filters=[df]
    )

    meta_json = '''{
        "chunks": [10, 10],
        "compression": "blosc",
        "compression_opts": {
            "clevel": 3,
            "cname": "lz4",
            "shuffle": 2
        },
        "dtype": [["a", "<i4"], ["b", "|S10"]],
        "fill_value": 42,
        "filters": [
            {"name": "delta", "astype": "<u2", "dtype": "|V14"}
        ],
        "order": "F",
        "shape": [100, 100],
        "zarr_format": %s
    }''' % ZARR_FORMAT

    # test encoding
    meta_enc = encode_array_metadata(meta)
    assert_json_eq(meta_json, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_enc)
    eq(ZARR_FORMAT, meta_dec['zarr_format'])
    eq(meta['shape'], meta_dec['shape'])
    eq(meta['chunks'], meta_dec['chunks'])
    eq(meta['dtype'], meta_dec['dtype'])
    eq(meta['compression'], meta_dec['compression'])
    eq(meta['compression_opts'], meta_dec['compression_opts'])
    eq(meta['order'], meta_dec['order'])
    eq(meta['fill_value'], meta_dec['fill_value'])
    eq([df.get_filter_config()], meta_dec['filters'])


def test_encode_decode_array_fill_values():

    fills = (
        (np.nan, "NaN", np.isnan),
        (np.NINF, "-Infinity", np.isneginf),
        (np.PINF, "Infinity", np.isposinf),
    )

    for v, s, f in fills:

        meta = dict(
            shape=(100,),
            chunks=(10,),
            dtype=np.dtype('f8'),
            compression='zlib',
            compression_opts=1,
            fill_value=v,
            filters=None,
            order='C'
        )

        meta_json = '''{
            "chunks": [10],
            "compression": "zlib",
            "compression_opts": 1,
            "dtype": "<f8",
            "fill_value": "%s",
            "filters": null,
            "order": "C",
            "shape": [100],
            "zarr_format": %s
        }''' % (s, ZARR_FORMAT)

        # test encoding
        meta_enc = encode_array_metadata(meta)
        assert_json_eq(meta_json, meta_enc)

        # test decoding
        meta_dec = decode_array_metadata(meta_enc)
        actual = meta_dec['fill_value']
        assert f(actual)


def test_decode_array_unsupported_format():

    # unsupported format
    meta_json = '''{
        "zarr_format": %s,
        "shape": [100],
        "chunks": [10],
        "dtype": "<f8",
        "compression": "zlib",
        "compression_opts": 1,
        "fill_value": null,
        "order": "C"
    }''' % (ZARR_FORMAT - 1)
    with assert_raises(MetadataError):
        decode_array_metadata(meta_json)


def test_decode_array_missing_fields():

    # missing fields
    meta_json = '''{
        "zarr_format": %s
    }''' % ZARR_FORMAT
    with assert_raises(MetadataError):
        decode_array_metadata(meta_json)


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
    meta = decode_group_metadata(b)
    eq(ZARR_FORMAT, meta['zarr_format'])

    # unsupported format
    b = '''{
        "zarr_format": %s
    }''' % (ZARR_FORMAT - 1)
    with assert_raises(MetadataError):
        decode_group_metadata(b)
