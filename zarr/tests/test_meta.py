# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import json
import base64


import numpy as np
import pytest


from zarr.compat import binary_type, text_type, PY2
from zarr.meta import (decode_array_metadata, encode_dtype, decode_dtype, ZARR_FORMAT,
                       decode_group_metadata, encode_array_metadata)
from zarr.errors import MetadataError
from zarr.codecs import Delta, Zlib, Blosc


def assert_json_equal(expect, actual):
    if isinstance(expect, binary_type):  # pragma: py3 no cover
        expect = text_type(expect, 'ascii')
    if isinstance(actual, binary_type):
        actual = text_type(actual, 'ascii')
    ej = json.loads(expect)
    aj = json.loads(actual)
    assert ej == aj


def test_encode_decode_array_1():

    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('f8'),
        compressor=Zlib(1).get_config(),
        fill_value=None,
        filters=None,
        order='C'
    )

    meta_json = '''{
        "chunks": [10],
        "compressor": {"id": "zlib", "level": 1},
        "dtype": "<f8",
        "fill_value": null,
        "filters": null,
        "order": "C",
        "shape": [100],
        "zarr_format": %s
    }''' % ZARR_FORMAT

    # test encoding
    meta_enc = encode_array_metadata(meta)
    assert_json_equal(meta_json, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_enc)
    assert ZARR_FORMAT == meta_dec['zarr_format']
    assert meta['shape'] == meta_dec['shape']
    assert meta['chunks'] == meta_dec['chunks']
    assert meta['dtype'] == meta_dec['dtype']
    assert meta['compressor'] == meta_dec['compressor']
    assert meta['order'] == meta_dec['order']
    assert meta_dec['fill_value'] is None
    assert meta_dec['filters'] is None


def test_encode_decode_array_2():

    # some variations
    df = Delta(astype='u2', dtype='V14')
    compressor = Blosc(cname='lz4', clevel=3, shuffle=2)
    dtype = np.dtype([('a', 'i4'), ('b', 'S10')])
    fill_value = np.zeros((), dtype=dtype)[()]
    meta = dict(
        shape=(100, 100),
        chunks=(10, 10),
        dtype=dtype,
        compressor=compressor.get_config(),
        fill_value=fill_value,
        order='F',
        filters=[df.get_config()]
    )

    meta_json = '''{
        "chunks": [10, 10],
        "compressor": {
            "id": "blosc",
            "clevel": 3,
            "cname": "lz4",
            "shuffle": 2,
            "blocksize": 0
        },
        "dtype": [["a", "<i4"], ["b", "|S10"]],
        "fill_value": "AAAAAAAAAAAAAAAAAAA=",
        "filters": [
            {"id": "delta", "astype": "<u2", "dtype": "|V14"}
        ],
        "order": "F",
        "shape": [100, 100],
        "zarr_format": %s
    }''' % ZARR_FORMAT

    # test encoding
    meta_enc = encode_array_metadata(meta)
    assert_json_equal(meta_json, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_enc)
    assert ZARR_FORMAT == meta_dec['zarr_format']
    assert meta['shape'] == meta_dec['shape']
    assert meta['chunks'] == meta_dec['chunks']
    assert meta['dtype'] == meta_dec['dtype']
    assert meta['compressor'] == meta_dec['compressor']
    assert meta['order'] == meta_dec['order']
    assert fill_value == meta_dec['fill_value']
    assert [df.get_config()] == meta_dec['filters']


def test_encode_decode_array_complex():

    # some variations
    for k in ['c8', 'c16']:
        compressor = Blosc(cname='lz4', clevel=3, shuffle=2)
        dtype = np.dtype(k)
        fill_value = dtype.type(np.nan-1j)
        meta = dict(
            shape=(100, 100),
            chunks=(10, 10),
            dtype=dtype,
            compressor=compressor.get_config(),
            fill_value=fill_value,
            order=dtype.char,
            filters=[]
        )

        meta_json = '''{
            "chunks": [10, 10],
            "compressor": {
                "id": "blosc",
                "clevel": 3,
                "cname": "lz4",
                "shuffle": 2,
                "blocksize": 0
            },
            "dtype": "%s",
            "fill_value": ["NaN", -1.0],
            "filters": [],
            "order": "%s",
            "shape": [100, 100],
            "zarr_format": %s
        }''' % (dtype.str, dtype.char, ZARR_FORMAT)

        # test encoding
        meta_enc = encode_array_metadata(meta)
        assert_json_equal(meta_json, meta_enc)

        # test decoding
        meta_dec = decode_array_metadata(meta_enc)
        assert ZARR_FORMAT == meta_dec['zarr_format']
        assert meta['shape'] == meta_dec['shape']
        assert meta['chunks'] == meta_dec['chunks']
        assert meta['dtype'] == meta_dec['dtype']
        assert meta['compressor'] == meta_dec['compressor']
        assert meta['order'] == meta_dec['order']
        # Based off of this SO answer: https://stackoverflow.com/a/49972198
        assert np.all(
            fill_value.view((np.uint8, fill_value.itemsize)) ==
            meta_dec['fill_value'].view((np.uint8, meta_dec['fill_value'].itemsize))
        )
        assert [] == meta_dec['filters']


def test_encode_decode_array_datetime_timedelta():

    # some variations
    for k in ['m8[s]', 'M8[s]']:
        compressor = Blosc(cname='lz4', clevel=3, shuffle=2)
        dtype = np.dtype(k)
        fill_value = dtype.type("NaT")
        meta = dict(
            shape=(100, 100),
            chunks=(10, 10),
            dtype=dtype,
            compressor=compressor.get_config(),
            fill_value=fill_value,
            order=dtype.char,
            filters=[]
        )

        meta_json = '''{
            "chunks": [10, 10],
            "compressor": {
                "id": "blosc",
                "clevel": 3,
                "cname": "lz4",
                "shuffle": 2,
                "blocksize": 0
            },
            "dtype": "%s",
            "fill_value": -9223372036854775808,
            "filters": [],
            "order": "%s",
            "shape": [100, 100],
            "zarr_format": %s
        }''' % (dtype.str, dtype.char, ZARR_FORMAT)

        # test encoding
        meta_enc = encode_array_metadata(meta)
        assert_json_equal(meta_json, meta_enc)

        # test decoding
        meta_dec = decode_array_metadata(meta_enc)
        assert ZARR_FORMAT == meta_dec['zarr_format']
        assert meta['shape'] == meta_dec['shape']
        assert meta['chunks'] == meta_dec['chunks']
        assert meta['dtype'] == meta_dec['dtype']
        assert meta['compressor'] == meta_dec['compressor']
        assert meta['order'] == meta_dec['order']
        # Based off of this SO answer: https://stackoverflow.com/a/49972198
        assert np.all(
            fill_value.view((np.uint8, fill_value.itemsize)) ==
            meta_dec['fill_value'].view((np.uint8, meta_dec['fill_value'].itemsize))
        )
        assert [] == meta_dec['filters']


def test_encode_decode_array_dtype_shape():

    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('(10, 10)f8'),
        compressor=Zlib(1).get_config(),
        fill_value=None,
        filters=None,
        order='C'
    )

    meta_json = '''{
        "chunks": [10],
        "compressor": {"id": "zlib", "level": 1},
        "dtype": "<f8",
        "fill_value": null,
        "filters": null,
        "order": "C",
        "shape": [100, 10, 10],
        "zarr_format": %s
    }''' % ZARR_FORMAT

    # test encoding
    meta_enc = encode_array_metadata(meta)
    assert_json_equal(meta_json, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_enc)
    assert ZARR_FORMAT == meta_dec['zarr_format']
    # to maintain consistency with numpy unstructured arrays, unpack dimensions into shape
    assert meta['shape'] + meta['dtype'].shape == meta_dec['shape']
    assert meta['chunks'] == meta_dec['chunks']
    # to maintain consistency with numpy unstructured arrays, unpack dtypes
    assert meta['dtype'].base == meta_dec['dtype']
    assert meta['compressor'] == meta_dec['compressor']
    assert meta['order'] == meta_dec['order']
    assert meta_dec['fill_value'] is None
    assert meta_dec['filters'] is None


def test_encode_decode_array_structured():

    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('i8, (10, 10)f8, (5, 10, 15)u1'),
        compressor=Zlib(1).get_config(),
        fill_value=None,
        filters=None,
        order='C'
    )

    meta_json = '''{
        "chunks": [10],
        "compressor": {"id": "zlib", "level": 1},
        "dtype": [["f0", "<i8"], ["f1", "<f8", [10, 10]], ["f2", "|u1", [5, 10, 15]]],
        "fill_value": null,
        "filters": null,
        "order": "C",
        "shape": [100],
        "zarr_format": %s
    }''' % ZARR_FORMAT

    # test encoding
    meta_enc = encode_array_metadata(meta)
    assert_json_equal(meta_json, meta_enc)

    # test decoding
    meta_dec = decode_array_metadata(meta_enc)
    assert ZARR_FORMAT == meta_dec['zarr_format']
    # to maintain consistency with numpy unstructured arrays, unpack dimensions into shape
    assert meta['shape'] + meta['dtype'].shape == meta_dec['shape']
    assert meta['chunks'] == meta_dec['chunks']
    # to maintain consistency with numpy unstructured arrays, unpack dimensions into shape
    assert meta['dtype'].base == meta_dec['dtype']
    assert meta['compressor'] == meta_dec['compressor']
    assert meta['order'] == meta_dec['order']
    assert meta_dec['fill_value'] is None
    assert meta_dec['filters'] is None


def test_encode_decode_fill_values_nan():

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
            compressor=Zlib(1).get_config(),
            fill_value=v,
            filters=None,
            order='C'
        )

        meta_json = '''{
            "chunks": [10],
            "compressor": {"id": "zlib", "level": 1},
            "dtype": "<f8",
            "fill_value": "%s",
            "filters": null,
            "order": "C",
            "shape": [100],
            "zarr_format": %s
        }''' % (s, ZARR_FORMAT)

        # test encoding
        meta_enc = encode_array_metadata(meta)
        assert_json_equal(meta_json, meta_enc)

        # test decoding
        meta_dec = decode_array_metadata(meta_enc)
        actual = meta_dec['fill_value']
        assert f(actual)


def test_encode_decode_fill_values_bytes():

    dtype = np.dtype('S10')
    fills = b'foo', bytes(10)

    for v in fills:

        # setup and encode metadata
        meta = dict(
            shape=(100,),
            chunks=(10,),
            dtype=dtype,
            compressor=Zlib(1).get_config(),
            fill_value=v,
            filters=None,
            order='C'
        )
        meta_enc = encode_array_metadata(meta)

        # define expected metadata encoded as JSON
        s = base64.standard_b64encode(v)
        if not PY2:
            s = s.decode()
        meta_json = '''{
            "chunks": [10],
            "compressor": {"id": "zlib", "level": 1},
            "dtype": "|S10",
            "fill_value": "%s",
            "filters": null,
            "order": "C",
            "shape": [100],
            "zarr_format": %s
        }''' % (s, ZARR_FORMAT)

        # test encoding
        assert_json_equal(meta_json, meta_enc)

        # test decoding
        meta_dec = decode_array_metadata(meta_enc)
        actual = meta_dec['fill_value']
        expect = np.array(v, dtype=dtype)[()]
        assert expect == actual


def test_decode_array_unsupported_format():

    # unsupported format
    meta_json = '''{
        "zarr_format": %s,
        "shape": [100],
        "chunks": [10],
        "dtype": "<f8",
        "compressor": {"id": "zlib", "level": 1},
        "fill_value": null,
        "order": "C"
    }''' % (ZARR_FORMAT - 1)
    with pytest.raises(MetadataError):
        decode_array_metadata(meta_json)


def test_decode_array_missing_fields():

    # missing fields
    meta_json = '''{
        "zarr_format": %s
    }''' % ZARR_FORMAT
    with pytest.raises(MetadataError):
        decode_array_metadata(meta_json)


def test_encode_decode_dtype():

    for dt in ['f8', [('a', 'f8')], [('a', 'f8'), ('b', 'i1')]]:
        e = encode_dtype(np.dtype(dt))
        s = json.dumps(e)  # check JSON serializable
        o = json.loads(s)
        d = decode_dtype(o)
        assert np.dtype(dt) == d


def test_decode_group():

    # typical
    b = '''{
        "zarr_format": %s
    }''' % ZARR_FORMAT
    meta = decode_group_metadata(b)
    assert ZARR_FORMAT == meta['zarr_format']

    # unsupported format
    b = '''{
        "zarr_format": %s
    }''' % (ZARR_FORMAT - 1)
    with pytest.raises(MetadataError):
        decode_group_metadata(b)
