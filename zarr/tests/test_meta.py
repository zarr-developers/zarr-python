import base64
import copy
import json

import numpy as np
import pytest

from zarr.codecs import Blosc, Delta, Pickle, Zlib
from zarr.errors import MetadataError
from zarr.meta import (ZARR_FORMAT, decode_array_metadata, decode_dtype,
                       decode_group_metadata, encode_array_metadata,
                       encode_dtype, encode_fill_value, decode_fill_value,
                       get_extended_dtype_info, _v3_complex_types,
                       _v3_datetime_types, _default_entry_point_metadata_v3,
                       Metadata3)
from zarr.util import normalize_dtype, normalize_fill_value


def assert_json_equal(expect, actual):
    if isinstance(actual, bytes):
        actual = str(actual, 'ascii')
    ej = json.loads(expect)
    aj = json.loads(actual)
    assert ej == aj


def test_encode_decode_array_1():

    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('<f8'),
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
    df = Delta(astype='<u2', dtype='V14')
    compressor = Blosc(cname='lz4', clevel=3, shuffle=2)
    dtype = np.dtype([('a', '<i4'), ('b', 'S10')])
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
        dtype=np.dtype('(10, 10)<f8'),
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


def test_encode_decode_array_dtype_shape_v3():

    meta = dict(
        shape=(100,),
        chunk_grid=dict(type='regular',
                        chunk_shape=(10,),
                        separator=('/')),
        data_type=np.dtype('(10, 10)<f8'),
        compressor=Zlib(1),
        fill_value=None,
        chunk_memory_layout='C'
    )

    meta_json = '''{
        "attributes": {},
        "chunk_grid": {
            "chunk_shape": [10],
            "separator": "/",
            "type": "regular"
        },
        "chunk_memory_layout": "C",
        "compressor": {
            "codec": "https://purl.org/zarr/spec/codec/zlib/1.0",
            "configuration": {
                "level": 1
            }
        },
        "data_type": "<f8",
        "extensions": [],
        "fill_value": null,
        "shape": [100, 10, 10 ]
    }'''

    # test encoding
    meta_enc = Metadata3.encode_array_metadata(meta)
    assert_json_equal(meta_json, meta_enc)

    # test decoding
    meta_dec = Metadata3.decode_array_metadata(meta_enc)
    # to maintain consistency with numpy unstructured arrays, unpack dimensions into shape
    assert meta['shape'] + meta['data_type'].shape == meta_dec['shape']
    assert meta['chunk_grid'] == meta_dec['chunk_grid']
    # to maintain consistency with numpy unstructured arrays, unpack dtypes
    assert meta['data_type'].base == meta_dec['data_type']
    assert meta['compressor'] == meta_dec['compressor']
    assert meta['chunk_memory_layout'] == meta_dec['chunk_memory_layout']
    assert meta_dec['fill_value'] is None
    assert 'filters' not in meta_dec


def test_encode_decode_array_structured():

    meta = dict(
        shape=(100,),
        chunks=(10,),
        dtype=np.dtype('<i8, (10, 10)<f8, (5, 10, 15)u1'),
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
            dtype=np.dtype('<f8'),
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


@pytest.mark.parametrize(
    "fill_value,dtype,object_codec,result",
    [
        (
            (0.0, None),
            [('x', float), ('y', object)],
            Pickle(),
            True,  # Pass
        ),
        (
            (0.0, None),
            [('x', float), ('y', object)],
            None,
            False,  # Fail
        ),
    ],
)
def test_encode_fill_value(fill_value, dtype, object_codec, result):

    # normalize metadata (copied from _init_array_metadata)
    dtype, object_codec = normalize_dtype(dtype, object_codec)
    dtype = dtype.base
    fill_value = normalize_fill_value(fill_value, dtype)

    # test
    if result:
        encode_fill_value(fill_value, dtype, object_codec)
    else:
        with pytest.raises(ValueError):
            encode_fill_value(fill_value, dtype, object_codec)


@pytest.mark.parametrize(
    "fill_value,dtype,object_codec,result",
    [
        (
            (0.0, None),
            [('x', float), ('y', object)],
            Pickle(),
            True,  # Pass
        ),
        (
            (0.0, None),
            [('x', float), ('y', object)],
            None,
            False,  # Fail
        ),
    ],
)
def test_decode_fill_value(fill_value, dtype, object_codec, result):

    # normalize metadata (copied from _init_array_metadata)
    dtype, object_codec = normalize_dtype(dtype, object_codec)
    dtype = dtype.base
    fill_value = normalize_fill_value(fill_value, dtype)

    # test
    if result:
        v = encode_fill_value(fill_value, dtype, object_codec)
        decode_fill_value(v, dtype, object_codec)
    else:
        with pytest.raises(ValueError):
            # No encoding is possible
            decode_fill_value(fill_value, dtype, object_codec)


def test_get_extended_dtype_info():
    extended_types = list(_v3_complex_types) + list(_v3_datetime_types)
    extended_types += ['|S4', '|S8', '>U4', '<U4', '|O', '|V16']

    for dtype in extended_types:
        try:
            info = get_extended_dtype_info(np.asarray([], dtype=dtype).dtype)
        except TypeError:  # pragma: no cover
            # skip any numpy dtypes not supported by a particular architecture
            pass  # pragma: no cover
        assert 'extension' in info
        assert 'type' in info
        assert 'fallback' in info

    class invalid_dtype():

        str = 'unknown_type'

    with pytest.raises(ValueError):
        get_extended_dtype_info(invalid_dtype)


def test_metadata3_exceptions():

    with pytest.raises(KeyError):
        # dict must have a key named 'type'
        Metadata3.decode_dtype({})

    required = ["zarr_format", "metadata_encoding",  "metadata_key_suffix", "extensions"]
    for key in required:
        meta = copy.copy(_default_entry_point_metadata_v3)
        meta.pop('zarr_format')
        with pytest.raises(ValueError):
            # cannot encode metadata that is missing a required key
            Metadata3.encode_hierarchy_metadata(meta)

    meta = copy.copy(_default_entry_point_metadata_v3)
    meta["extra_key"] = []
    with pytest.raises(ValueError):
        # cannot encode metadata that has unexpected keys
        Metadata3.encode_hierarchy_metadata(meta)

    json = Metadata3.encode_hierarchy_metadata(_default_entry_point_metadata_v3)
    # form a new json bytes object with one of the keys missing
    temp = json.split(b'\n')
    temp = temp[:2] + temp[3:]
    bad_json = b'\n'.join(temp)
    with pytest.raises(ValueError):
        # cannot encode metadata that is missing a required key
        Metadata3.decode_hierarchy_metadata(bad_json)

    json = Metadata3.encode_hierarchy_metadata(_default_entry_point_metadata_v3)
    temp = json.split(b'\n')
    temp = temp[:2] + [b'    "unexpected": [],'] + temp[2:]
    bad_json = b'\n'.join(temp)
    with pytest.raises(ValueError):
        # cannot encode metadata that has extra, unexpected keys
        Metadata3.decode_hierarchy_metadata(bad_json)

    codec_meta = dict(configuration=None, codec='unknown')
    with pytest.raises(NotImplementedError):
        Metadata3._decode_codec_metadata(codec_meta)

    with pytest.raises(MetadataError):
        Metadata3.decode_array_metadata(dict())
