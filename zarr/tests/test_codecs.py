# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import itertools
import array


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import eq_ as eq


from zarr.codecs import registry
from zarr.util import buffer_tobytes


class CodecTests(object):

    # override in sub-class
    name = None

    def init_codec(self, **kwargs):
        codec_cls = registry[self.name]
        codec = codec_cls(**kwargs)
        return codec

    def _test_encode(self, arr, **kwargs):

        # setup
        codec = self.init_codec(**kwargs)

        # encoding should support any object exporting the buffer protocol,
        # as well as array.array in PY2

        # test encoding of numpy array
        buf = arr
        enc = codec.encode(buf)
        enc_bytes = buffer_tobytes(enc)

        # test encoding of raw bytes
        buf = arr.tobytes()
        enc = codec.encode(buf)
        actual = buffer_tobytes(enc)
        eq(enc_bytes, actual)

        # test encoding of array.array
        buf = array.array('b', arr.tobytes())
        enc = codec.encode(buf)
        actual = buffer_tobytes(enc)
        eq(enc_bytes, actual)

    def _test_decode_lossless(self, arr, **kwargs):

        # setup
        codec = self.init_codec(**kwargs)

        # encode
        enc = codec.encode(arr)
        enc_bytes = buffer_tobytes(enc)

        # decoding should support any object exporting the buffer protocol,
        # as well as array.array in PY2

        # test decoding of raw bytes
        buf = enc_bytes
        dec = codec.decode(buf)
        assert_array_equal(arr, np.frombuffer(dec, dtype=arr.dtype))

        # test decoding of array.array
        buf = array.array('b', enc_bytes)
        dec = codec.decode(buf)
        assert_array_equal(arr, np.frombuffer(dec, dtype=arr.dtype))

        # test decoding of numpy array
        buf = np.frombuffer(enc_bytes, dtype='u1')
        dec = codec.decode(buf)
        assert_array_equal(arr, np.frombuffer(dec, dtype=arr.dtype))

        # test decoding into output
        out = np.empty_like(arr)
        codec.decode(enc_bytes, out=out)
        assert_array_equal(arr, out)

    def _test_decode_lossy(self, arr, decimal, **kwargs):

        # setup
        codec = self.init_codec(**kwargs)

        # encode
        enc = codec.encode(arr)
        enc_bytes = buffer_tobytes(enc)

        # decoding should support any object exporting the buffer protocol,
        # as well as array.array in PY2

        # test decoding of raw bytes
        buf = enc_bytes
        dec = codec.decode(buf)
        assert_array_almost_equal(arr, np.frombuffer(dec, dtype=arr.dtype),
                                  decimal=decimal)

        # test decoding of array.array
        buf = array.array('b', enc_bytes)
        dec = codec.decode(buf)
        assert_array_almost_equal(arr, np.frombuffer(dec, dtype=arr.dtype),
                                  decimal=decimal)

        # test decoding of numpy array
        buf = np.frombuffer(enc_bytes, dtype='u1')
        dec = codec.decode(buf)
        assert_array_almost_equal(arr, np.frombuffer(dec, dtype=arr.dtype),
                                  decimal=decimal)

        # test decoding into output
        out = np.empty_like(arr)
        codec.decode(enc_bytes, out=out)
        assert_array_almost_equal(arr, out, decimal=decimal)


class TestZlibCompressor(CodecTests, unittest.TestCase):

    name = 'zlib'
    arrs = [
        np.arange(1000, dtype='i4'),
        np.linspace(0, 1, 1000, dtype='f8'),
        np.random.randint(0, 2, size=1000, dtype=bool),
        np.random.normal(size=1000),
        np.random.choice([b'foo', b'bar', b'baz'], size=1000)
    ]
    configs = [
        dict(),
        dict(level=-1),
        dict(level=0),
        dict(level=1),
        dict(level=5),
        dict(level=9),
    ]

    def test_encode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            self._test_decode_lossless(arr, **config)


class TestBZ2Compressor(CodecTests, unittest.TestCase):

    name = 'bz2'
    arrs = [
        np.arange(1000, dtype='i4'),
        np.linspace(0, 1, 1000, dtype='f8'),
        np.random.randint(0, 2, size=1000, dtype=bool),
        np.random.normal(size=1000),
        np.random.choice([b'foo', b'bar', b'baz'], size=1000)
    ]
    configs = [
        dict(),
        dict(level=1),
        dict(level=5),
        dict(level=9),
    ]

    def test_encode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            self._test_decode_lossless(arr, **config)


try:
    import lzma
except ImportError:  # pragma: no cover
    pass
else:

    class TestLZMACompressor(CodecTests, unittest.TestCase):

        name = 'lzma'
        arrs = [
            np.arange(1000, dtype='i4'),
            np.linspace(0, 1, 1000, dtype='f8'),
            np.random.randint(0, 2, size=1000, dtype=bool),
            np.random.normal(size=1000),
            np.random.choice([b'foo', b'bar', b'baz'], size=1000)
        ]
        configs = [
            dict(),
            dict(preset=1),
            dict(preset=5),
            dict(preset=9),
            dict(format=lzma.FORMAT_RAW,
                 filters=[dict(id=lzma.FILTER_LZMA2, preset=1)])
        ]

        def test_encode(self):
            for arr, config in itertools.product(self.arrs, self.configs):
                self._test_encode(arr, **config)

        def test_decode(self):
            for arr, config in itertools.product(self.arrs, self.configs):
                self._test_decode_lossless(arr, **config)


try:
    from zarr import blosc
except ImportError:  # pragma: no cover
    pass
else:

    class TestBloscCompressor(CodecTests, unittest.TestCase):

        name = 'blosc'

        arrs = [
            np.arange(1000, dtype='i4'),
            np.linspace(0, 1, 1000, dtype='f8'),
            np.random.randint(0, 2, size=1000, dtype=bool),
            np.random.normal(size=1000),
            np.random.choice([b'a', b'b', b'c'], size=1000)
        ]
        configs = [
            dict(),
            dict(clevel=0),
            dict(cname='lz4'),
            dict(cname='lz4', clevel=1, shuffle=0),
            dict(cname='lz4', clevel=1, shuffle=1),
            dict(cname='lz4', clevel=1, shuffle=2),
            dict(cname='zlib', clevel=1, shuffle=2),
            dict(cname='zstd', clevel=1, shuffle=2),
            dict(cname='blosclz', clevel=1, shuffle=2),
            dict(cname='snappy', clevel=1, shuffle=2),
        ]

        def test_encode(self):

            # N.B., watch out here with blosc compressor, if the itemsize of
            # the source buffer is different then the results of compression
            # may be different.

            for arr, config in itertools.product(self.arrs, self.configs):
                if arr.dtype.itemsize == 1:
                    self._test_encode(arr, **config)

        def test_decode(self):
            for arr, config in itertools.product(self.arrs, self.configs):
                self._test_decode_lossless(arr, **config)


class TestDeltaFilter(CodecTests, unittest.TestCase):

    name = 'delta'

    arrs = [
        np.arange(1000, dtype='i4'),
        np.linspace(0, 1, 1000, dtype='f8'),
        np.random.randint(-10, 10, size=1000, dtype='i1'),
        np.random.normal(size=1000),
    ]

    def test_encode(self):
        for arr in self.arrs:
            self._test_encode(arr, dtype=arr.dtype)

    def test_decode(self):
        for arr in self.arrs:
            if arr.dtype.kind == 'f':
                self._test_decode_lossy(arr, decimal=10, dtype=arr.dtype)
            else:
                self._test_decode_lossless(arr, dtype=arr.dtype)

    def test_encode_output(self):
        dtype = 'i8'
        astype = 'i4'
        codec = self.init_codec(dtype=dtype, astype=astype)
        arr = np.arange(10, 20, 1, dtype=dtype)
        expect = np.array([10] + ([1] * 9), dtype=astype)
        actual = codec.encode(arr)
        assert_array_equal(expect, actual)
        eq(np.dtype(astype), actual.dtype)


class TestFixedScaleOffsetFilter(CodecTests, unittest.TestCase):

    name = 'fixedscaleoffset'

    arrs = [
        np.linspace(1000, 1001, 1000, dtype='f8'),
        np.random.normal(loc=1000, scale=1, size=1000).astype('f8'),
    ]
    configs = [
        dict(offset=1000, scale=10, dtype='f8', astype='i1'),
        dict(offset=1000, scale=10**2, dtype='f8', astype='i2'),
        dict(offset=1000, scale=10**6, dtype='f8', astype='i4'),
        dict(offset=1000, scale=10**12, dtype='f8', astype='i8'),
    ]

    def test_encode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            decimal = int(np.log10(config['scale']))
            print(config, decimal, arr[:5])
            self._test_decode_lossy(arr, decimal=decimal, **config)

    def test_encode_output(self):
        dtype = 'f8'
        astype = 'u1'
        codec = self.init_codec(scale=10, offset=1000, dtype=dtype,
                                astype=astype)
        arr = np.linspace(1000, 1001, 10, dtype=dtype)
        expect = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10], dtype=astype)
        actual = codec.encode(arr)
        assert_array_equal(expect, actual)
        eq(np.dtype(astype), actual.dtype)


class TestQuantizeFilter(CodecTests, unittest.TestCase):

    name = 'quantize'

    arrs = [
        np.linspace(0, 1, 1000, dtype='f8'),
        np.random.normal(loc=0, scale=1, size=1000).astype('f8'),
    ]
    configs = [
        dict(digits=1, dtype='f8', astype='f2'),
        dict(digits=6, dtype='f8', astype='f4'),
        dict(digits=12, dtype='f8', astype='f8'),
    ]

    def test_encode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            decimal = config['digits']
            print(config, decimal, arr[:5])
            self._test_decode_lossy(arr, decimal=decimal, **config)


class TestPackBitsFilter(CodecTests, unittest.TestCase):

    name = 'packbits'

    arr = np.random.randint(0, 2, size=1000, dtype=bool)

    def test_encode(self):
        for size in list(range(1, 17)) + [1000]:
            arr = self.arr[:size]
            self._test_encode(arr)

    def test_decode(self):
        for size in list(range(1, 17)) + [1000]:
            arr = self.arr[:size]
            self._test_decode_lossless(arr)


class TestCategorizeFilter(CodecTests, unittest.TestCase):

    name = 'categorize'
    labels = [b'foo', b'bar', b'baz', b'quux']
    arr = np.random.choice(labels, size=1000)

    def test_encode(self):
        self._test_encode(self.arr, labels=self.labels,
                          dtype=self.arr.dtype, astype='u1')

    def test_decode(self):
        self._test_decode_lossless(self.arr, labels=self.labels,
                                   dtype=self.arr.dtype, astype='u1')

    def test_encode_output(self):
        labels = ['foo', 'bar', 'baz']
        arr = np.array([b'foo', b'bar', b'foo', b'baz', b'quux'])
        codec = self.init_codec(labels=labels, dtype=arr.dtype, astype='u1')

        # test encoding
        expect = np.array([1, 2, 1, 3, 0], dtype='u1')
        enc = codec.encode(arr)
        assert_array_equal(expect, enc)
        eq(expect.dtype, enc.dtype)

        # test decoding
        dec = codec.decode(enc)
        expect = arr.copy()
        expect[expect == b'quux'] = b''
        assert_array_equal(expect, dec)
        eq(arr.dtype, dec.dtype)
