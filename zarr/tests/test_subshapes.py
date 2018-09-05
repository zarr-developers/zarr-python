# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
from tempfile import mkdtemp, mktemp
import json
import atexit
import shutil
import pickle
import os
import warnings


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest


from zarr.codecs import Zlib
from zarr.compat import binary_type, text_type, PY2
from zarr.creation import array, open_array
from zarr.meta import (decode_array_metadata, encode_dtype, decode_dtype, ZARR_FORMAT,
                       encode_array_metadata)

from zarr.storage import DirectoryStore


if PY2: # pragma: py3 no cover
    warnings.resetwarnings()
    warnings.simplefilter('always')


def assert_json_equal(expect, actual):
    if isinstance(expect, binary_type):  # pragma: py3 no cover
        expect = text_type(expect, 'ascii')
    if isinstance(actual, binary_type):
        actual = text_type(actual, 'ascii')
    ej = json.loads(expect)
    aj = json.loads(actual)
    assert ej == aj


class TestUnstructured(unittest.TestCase):

    dtype = np.dtype('(10, 10)f8')

    def test_encode_decode(self):

        meta = dict(
            shape=(100,),
            chunks=(10,),
            dtype=TestUnstructured.dtype,
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
        # NOTE(onalant): https://github.com/zarr-developers/zarr/pull/296#issuecomment-417608487
        # To maintain consistency with numpy unstructured arrays, unpack dimensions into shape.
        # assert meta['shape'] == meta_dec['shape']
        assert meta['chunks'] == meta_dec['chunks']
        # NOTE(onalant): https://github.com/zarr-developers/zarr/pull/296#issuecomment-417608487
        # To maintain consistency with numpy unstructured arrays, unpack dimensions into shape.
        # assert meta['dtype'] == meta_dec['dtype']
        assert meta['compressor'] == meta_dec['compressor']
        assert meta['order'] == meta_dec['order']
        assert meta_dec['fill_value'] is None
        assert meta_dec['filters'] is None


    def test_write_read(self):

        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        shape = (100, 100)
        sbytes = np.random.bytes(np.product(shape) * TestUnstructured.dtype.itemsize)
        shape += TestUnstructured.dtype.shape
        s = np.frombuffer(sbytes, dtype=TestUnstructured.dtype).reshape(shape)

        store = DirectoryStore(path)
        z = array(s, store=store)

        assert(s.shape == z.shape)
        assert(s.tobytes() == z[:].tobytes())

        del store
        del z

        store = DirectoryStore(path)
        z = open_array(store)

        assert(s.shape == z.shape)
        assert(s.tobytes() == z[:].tobytes())


class TestStructured(unittest.TestCase):

    dtype = np.dtype('i8, (10, 10)f8, (5, 10, 15)u1')

    def test_encode_decode(self):

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
        assert meta['shape'] == meta_dec['shape']
        assert meta['chunks'] == meta_dec['chunks']
        assert meta['dtype'] == meta_dec['dtype']
        assert meta['compressor'] == meta_dec['compressor']
        assert meta['order'] == meta_dec['order']
        assert meta_dec['fill_value'] is None
        assert meta_dec['filters'] is None


    def test_write_read(self):

        path = mkdtemp()
        atexit.register(shutil.rmtree, path)
        shape = (100, 100)
        sbytes = np.random.bytes(np.product(shape) * TestUnstructured.dtype.itemsize)
        shape += TestUnstructured.dtype.shape
        s = np.frombuffer(sbytes, dtype=TestUnstructured.dtype).reshape(shape)

        store = DirectoryStore(path)
        z = array(s, store=store)

        assert(s.shape == z.shape)
        assert(s.tobytes() == z[:].tobytes())

        del store
        del z

        store = DirectoryStore(path)
        z = open_array(store)

        assert(s.shape == z.shape)
        assert(s.tobytes() == z[:].tobytes())


