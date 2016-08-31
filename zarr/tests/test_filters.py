# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import eq_ as eq, assert_raises


from zarr.filters import DeltaFilter, FixedScaleOffsetFilter, \
    QuantizeFilter, PackBitsFilter, CategoryFilter
from zarr.creation import array
from zarr.compat import PY2
from zarr.compressors import get_compressor_cls
from zarr.util import buffer_tobytes


class TestDeltaFilter(unittest.TestCase):

    def test_encode_decode(self):
        dtype = 'i8'
        astype = 'u1'
        f = DeltaFilter(dtype=dtype, astype=astype)
        data = np.arange(10, dtype=dtype)

        # test encoding
        expect_enc = np.array([0] + ([1] * 9), dtype=astype)
        enc = f.encode(data)
        assert_array_equal(expect_enc, enc)
        eq(np.dtype(astype), enc.dtype)

        # test decoding
        dec = f.decode(enc)
        assert_array_equal(data, dec)
        eq(np.dtype(dtype), dec.dtype)


class TestFixedScaleOffsetFilter(unittest.TestCase):

    def test_encode_decode(self):
        dtype = 'f8'
        astype = 'u1'
        f = FixedScaleOffsetFilter(scale=10, offset=1000, astype=astype,
                                   dtype=dtype)
        data = np.linspace(1000, 1001, 10, dtype=dtype)

        # test encoding
        expected_enc = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=astype)
        enc = f.encode(data)
        assert_array_equal(expected_enc, enc)
        eq(np.dtype(astype), enc.dtype)

        # test decoding
        dec = f.decode(enc)
        assert_array_almost_equal(data, dec, decimal=1)
        eq(np.dtype(dtype), dec.dtype)


class TestQuantizeFilter(unittest.TestCase):

    def test_encode_decode(self):
        dtype = 'f8'
        astype = 'f4'
        data = np.linspace(0, 1, 34, dtype=dtype)

        for digits in range(5):
            f = QuantizeFilter(digits=digits, dtype=dtype,
                               astype=astype)

            # test encoding
            enc = f.encode(data)
            eq(np.dtype(astype), enc.dtype)
            assert_array_almost_equal(data, enc, decimal=digits)

            # test decoding
            dec = f.decode(enc)
            # should be no-op
            assert_array_equal(enc, dec)
            eq(np.dtype(dtype), dec.dtype)

    def test_errors(self):
        with assert_raises(ValueError):
            # only float dtypes supported
            QuantizeFilter(digits=1, dtype='i4')
        with assert_raises(ValueError):
            # only float dtypes supported
            QuantizeFilter(digits=1, dtype='f4', astype='i4')


class TestPackBitsFilter(unittest.TestCase):

    def test_encode_decode(self):
        data = np.array([0, 1] * 8, dtype=bool)
        f = PackBitsFilter()

        for size in range(1, 17):
            print('size', size)

            # test encoding
            enc = f.encode(data[:size])
            eq(np.dtype('u1'), enc.dtype)
            expected_nbytes_packed = size // 8
            if (size % 8) > 0:
                expected_nbytes_packed += 1
            # need one more byte to store number of bits padded
            eq(expected_nbytes_packed + 1, enc.size)

            # test decoding
            dec = f.decode(enc)
            assert_array_equal(data[:size], dec)
            eq(np.dtype(bool), dec.dtype)


class TestCategoryFilter(unittest.TestCase):

    def test_encode_decode(self):
        data = np.array([b'foo', b'bar', b'foo', b'baz',
                         b'quux'])
        f = CategoryFilter(dtype=data.dtype, labels=['foo', 'bar', 'baz'])

        # test encoding
        enc = f.encode(data)
        expect = np.array([1, 2, 1, 3, 0], dtype='u1')
        assert_array_equal(expect, enc)
        eq(expect.dtype, enc.dtype)

        # test decoding
        dec = f.decode(enc)
        expect = data.copy()
        expect[expect == b'quux'] = b''
        assert_array_equal(expect, dec)
        eq(data.dtype, dec.dtype)

    def test_errors(self):
        with assert_raises(ValueError):
            # only support string dtype
            CategoryFilter(dtype='U2', labels=['foo', 'bar'])
        with assert_raises(ValueError):
            # bad labels
            CategoryFilter(dtype='S2', labels=[1, 2])

compression_configs = [
    ('none', None),
    ('zlib', None),
    ('bz2', None),
    ('blosc', None)
]
if not PY2:
    compression_configs.append(('lzma', None))


def test_array_with_delta_filter():

    # setup
    astype = 'u1'
    dtype = 'i8'
    filters = [DeltaFilter(astype=astype, dtype=dtype)]
    data = np.arange(100, dtype=dtype)

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
                                   dtype=astype)
            expect = np.array([i * 10] + ([1] * 9), dtype=astype)
            assert_array_equal(expect, actual)


def test_array_with_scaleoffset_filter():

    # setup
    astype = 'u1'
    dtype = 'f8'
    flt = FixedScaleOffsetFilter(scale=10, offset=1000, astype=astype,
                                 dtype=dtype)
    filters = [flt]
    data = np.linspace(1000, 1001, 34, dtype='f8')

    for compression, compression_opts in compression_configs:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_almost_equal(data, a[:], decimal=1)

        # check chunks
        for i in range(6):
            cdata = a.store[str(i)]
            actual = np.frombuffer(a.compressor.decompress(cdata),
                                   dtype=astype)
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_array_with_quantize_filter():

    # setup
    dtype = 'f8'
    digits = 3
    flt = QuantizeFilter(digits=digits, dtype=dtype)
    filters = [flt]
    data = np.linspace(0, 1, 34, dtype=dtype)

    for compression, compression_opts in compression_configs:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_almost_equal(data, a[:], decimal=digits)

        # check chunks
        for i in range(6):
            cdata = a.store[str(i)]
            actual = np.frombuffer(a.compressor.decompress(cdata),
                                   dtype=dtype)
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_array_with_packbits_filter():

    # setup
    flt = PackBitsFilter()
    filters = [flt]
    data = np.random.randint(0, 2, size=100, dtype=bool)

    for compression, compression_opts in compression_configs:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_equal(data, a[:])

        # check chunks
        for i in range(20):
            cdata = a.store[str(i)]
            actual = np.frombuffer(a.compressor.decompress(cdata),
                                   dtype='u1')
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_array_with_category_filter():

    # setup
    data = np.random.choice([b'foo', b'bar', b'baz'], size=100)
    flt = CategoryFilter(dtype=data.dtype, labels=['foo', 'bar', 'baz'])
    filters = [flt]

    for compression, compression_opts in compression_configs:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_equal(data, a[:])

        # check chunks
        for i in range(20):
            cdata = a.store[str(i)]
            actual = np.frombuffer(a.compressor.decompress(cdata),
                                   dtype='u1')
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_compressor_as_filter():
    for compression, compression_opts in compression_configs:

        # setup compressor
        compressor_cls = get_compressor_cls(compression)
        compression_opts = compressor_cls.normalize_opts(compression_opts)
        compressor = compressor_cls(compression_opts)

        # setup filters
        dtype = 'i8'
        filters = [
            DeltaFilter(dtype=dtype),
            compressor
        ]

        # setup data and arrays
        data = np.arange(10000, dtype=dtype)
        a1 = array(data, chunks=1000, compression=None, filters=filters)
        a2 = array(data, chunks=1000, compression=compression,
                   compression_opts=compression_opts, filters=filters[:1])

        # check storage
        for i in range(10):
            x = buffer_tobytes(a1.store[str(i)])
            y = buffer_tobytes(a2.store[str(i)])
            eq(x, y)

        # check data
        assert_array_equal(data, a1[:])
        assert_array_equal(a1[:], a2[:])
