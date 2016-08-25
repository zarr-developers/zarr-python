# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq


from zarr.filters import DeltaFilter, ScaleOffsetFilter
from zarr.creation import array
from zarr.compat import PY2
from zarr.compressors import get_compressor_cls
from zarr.util import buffer_tobytes


class TestDeltaFilter(unittest.TestCase):

    def test_encode_decode(self):
        enc_dtype = 'u1'
        dec_dtype = 'f8'
        f = DeltaFilter(enc_dtype=enc_dtype, dec_dtype=dec_dtype)
        data = np.arange(10, dtype=dec_dtype)

        # test encoding
        expect_enc = np.array([0] + ([1] * 9), dtype=enc_dtype)
        enc = f.encode(data)
        assert_array_equal(expect_enc, enc)
        eq(np.dtype(enc_dtype), enc.dtype)

        # test decoding
        dec = f.decode(enc)
        assert_array_equal(data, dec)
        eq(np.dtype(dec_dtype), dec.dtype)


class TestScaleOffsetFilter(unittest.TestCase):

    def test_encode_decode(self):
        enc_dtype = 'u1'
        dec_dtype = 'f8'
        f = ScaleOffsetFilter(scale=10, offset=1000, enc_dtype=enc_dtype,
                              dec_dtype=dec_dtype)
        data = 1000 + np.arange(0, 100, 10, dtype=dec_dtype)

        # test encoding
        expected_enc = np.arange(10, dtype=enc_dtype)
        enc = f.encode(data)
        assert_array_equal(expected_enc, enc)
        eq(np.dtype(enc_dtype), enc.dtype)

        # test decoding
        dec = f.decode(enc)
        assert_array_equal(data, dec)
        eq(np.dtype(dec_dtype), dec.dtype)


compression_configs = [
    ('none', None),
    ('zlib', None),
    ('bz2', None),
    ('blosc', None)
]
if not PY2:
    compression_configs.append(('lzma', None))


def test_array_with_filters_1():

    # setup
    enc_dtype = 'u1'
    dec_dtype = 'f8'
    filters = [DeltaFilter(enc_dtype=enc_dtype, dec_dtype=dec_dtype)]
    data = np.arange(100, dtype=dec_dtype)

    for compression, compression_opts in compression_configs:
        print(compression, compression_opts)

        a = array(data, chunks=10, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_equal(data, a[:])

        # check chunks
        for i in range(10):
            cdata = a.store[str(i)]
            actual = np.frombuffer(a.compressor.decompress(cdata),
                                   dtype=enc_dtype)
            expect = np.array([i * 10] + ([1] * 9), dtype=enc_dtype)
            assert_array_equal(expect, actual)


def test_array_with_filters_2():

    # setup
    enc_dtype = 'u1'
    dec_dtype = 'f8'
    filters = [ScaleOffsetFilter(scale=10, offset=1000, enc_dtype=enc_dtype,
                                 dec_dtype=dec_dtype)]
    data = 1000 + np.arange(0, 100, 10, dtype=dec_dtype)

    for compression, compression_opts in compression_configs:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_equal(data, a[:])

        # check chunks
        for i in range(2):
            cdata = a.store[str(i)]
            actual = np.frombuffer(a.compressor.decompress(cdata),
                                   dtype=enc_dtype)
            expect = np.arange(i*5, (i*5)+5, 1, dtype=enc_dtype)
            assert_array_equal(expect, actual)


def test_compressor_as_filter_1():
    for compression, compression_opts in compression_configs:

        # setup compressor
        compressor_cls = get_compressor_cls(compression)
        compression_opts = compressor_cls.normalize_opts(compression_opts)
        compressor = compressor_cls(compression_opts)

        # setup filters
        enc_dtype = 'u1'
        dec_dtype = 'f8'
        filters = [
            DeltaFilter(enc_dtype=enc_dtype, dec_dtype=dec_dtype),
            compressor
        ]

        # setup data and arrays
        data = np.arange(100, dtype=dec_dtype)
        a1 = array(data, chunks=10, compression=None, filters=filters)
        a2 = array(data, chunks=10, compression=compression,
                   compression_opts=compression_opts, filters=filters[:1])

        # check storage
        for i in range(10):
            x = buffer_tobytes(a1.store[str(i)])
            y = buffer_tobytes(a2.store[str(i)])
            eq(x, y)

        # check data
        assert_array_equal(data, a1[:])
        assert_array_equal(a1[:], a2[:])
