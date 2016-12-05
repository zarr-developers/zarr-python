# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import itertools
import array


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, \
    assert_raises
from nose.tools import eq_ as eq


from zarr.codecs import codec_registry, get_codec
from zarr.util import buffer_tobytes
from zarr.compat import PY2
from zarr import blosc


class CodecTests(object):

    # override in sub-class
    codec_id = None

    def init_codec(self, **kwargs):
        codec_cls = codec_registry[self.codec_id]
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
        buf = arr.tobytes(order='A')
        enc = codec.encode(buf)
        actual = buffer_tobytes(enc)
        eq(enc_bytes, actual)

        # test encoding of array.array
        buf = array.array('b', arr.tobytes(order='A'))
        enc = codec.encode(buf)
        actual = buffer_tobytes(enc)
        eq(enc_bytes, actual)

    def _test_decode_lossless(self, arr, **kwargs):
        if arr.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'

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
        dec = np.frombuffer(dec, dtype=arr.dtype).reshape(arr.shape,
                                                          order=order)
        assert_array_equal(arr, dec)

        # test decoding of array.array
        buf = array.array('b', enc_bytes)
        dec = codec.decode(buf)
        dec = np.frombuffer(dec, dtype=arr.dtype).reshape(arr.shape,
                                                          order=order)
        assert_array_equal(arr, dec)

        # test decoding of numpy array
        buf = np.frombuffer(enc_bytes, dtype='u1')
        dec = codec.decode(buf)
        dec = np.frombuffer(dec, dtype=arr.dtype).reshape(arr.shape,
                                                          order=order)
        assert_array_equal(arr, dec)

        # test decoding into numpy array
        out = np.empty_like(arr)
        codec.decode(enc_bytes, out=out)
        assert_array_equal(arr, out)

        # test decoding into bytearray
        out = bytearray(arr.nbytes)
        codec.decode(enc_bytes, out=out)
        expect = arr.tobytes(order='A')
        eq(expect, out)

    def _test_decode_lossy(self, arr, decimal, **kwargs):
        if arr.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'

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
        dec = np.frombuffer(dec, dtype=arr.dtype).reshape(arr.shape,
                                                          order=order)
        assert_array_almost_equal(arr, dec, decimal=decimal)

        # test decoding of array.array
        buf = array.array('b', enc_bytes)
        dec = codec.decode(buf)
        dec = np.frombuffer(dec, dtype=arr.dtype).reshape(arr.shape,
                                                          order=order)
        assert_array_almost_equal(arr, dec, decimal=decimal)

        # test decoding of numpy array
        buf = np.frombuffer(enc_bytes, dtype='u1')
        dec = codec.decode(buf)
        dec = np.frombuffer(dec, dtype=arr.dtype).reshape(arr.shape,
                                                          order=order)
        assert_array_almost_equal(arr, dec, decimal=decimal)

        # test decoding into numpy array
        out = np.empty_like(arr)
        codec.decode(enc_bytes, out=out)
        assert_array_almost_equal(arr, out, decimal=decimal)

        # test decoding into bytearray
        out = bytearray(arr.nbytes)
        codec.decode(enc_bytes, out=out)
        out = np.frombuffer(out, dtype=arr.dtype).reshape(arr.shape,
                                                          order=order)
        assert_array_almost_equal(arr, out, decimal=decimal)


test_arrays = [
    np.arange(1000, dtype='i4'),
    np.linspace(1000, 1001, 1000, dtype='f8'),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order='F'),
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10)
]


class TestZlib(CodecTests, unittest.TestCase):

    codec_id = 'zlib'

    configs = [
        dict(),
        dict(level=-1),
        dict(level=0),
        dict(level=1),
        dict(level=5),
        dict(level=9),
    ]

    def test_encode(self):
        for arr, config in itertools.product(test_arrays, self.configs):
            self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(test_arrays, self.configs):
            self._test_decode_lossless(arr, **config)


class TestBZ2(CodecTests, unittest.TestCase):

    codec_id = 'bz2'

    configs = [
        dict(),
        dict(level=1),
        dict(level=5),
        dict(level=9),
    ]

    def test_encode(self):
        for arr, config in itertools.product(test_arrays, self.configs):
            self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(test_arrays, self.configs):
            self._test_decode_lossless(arr, **config)


try:
    import lzma
except ImportError:  # pragma: no cover
    pass
else:

    class TestLZMA(CodecTests, unittest.TestCase):

        codec_id = 'lzma'

        configs = [
            dict(),
            dict(preset=1),
            dict(preset=5),
            dict(preset=9),
            dict(format=lzma.FORMAT_RAW,
                 filters=[dict(id=lzma.FILTER_LZMA2, preset=1)])
        ]

        def test_encode(self):
            for arr, config in itertools.product(test_arrays, self.configs):
                self._test_encode(arr, **config)

        def test_decode(self):
            for arr, config in itertools.product(test_arrays, self.configs):
                self._test_decode_lossless(arr, **config)


class TestBlosc(CodecTests, unittest.TestCase):

    codec_id = 'blosc'

    configs = [
        dict(),
        dict(clevel=0),
        dict(cname='lz4'),
        dict(cname='lz4', clevel=1, shuffle=blosc.NOSHUFFLE),
        dict(cname='lz4', clevel=1, shuffle=blosc.SHUFFLE),
        dict(cname='lz4', clevel=1, shuffle=blosc.BITSHUFFLE),
        dict(cname='zlib', clevel=1, shuffle=0),
        dict(cname='zstd', clevel=1, shuffle=1),
        dict(cname='blosclz', clevel=1, shuffle=2),
        dict(cname='snappy', clevel=1, shuffle=2),
    ]

    def test_encode(self):

        # N.B., watch out here with blosc compressor, if the itemsize of
        # the source buffer is different then the results of compression
        # may be different.

        for arr, config in itertools.product(test_arrays, self.configs):
            if arr.dtype.itemsize == 1:
                self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(test_arrays, self.configs):
            self._test_decode_lossless(arr, **config)


class TestDelta(CodecTests, unittest.TestCase):

    codec_id = 'delta'

    def test_encode(self):
        for arr in test_arrays:
            if arr.dtype.kind in {'f', 'i', 'u'}:
                self._test_encode(arr, dtype=arr.dtype)

    def test_decode(self):
        for arr in test_arrays:
            if arr.dtype.kind == 'f':
                self._test_decode_lossy(arr, decimal=10, dtype=arr.dtype)
            elif arr.dtype.kind in {'i', 'u'}:
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

    def test_repr(self):
        codec = self.init_codec(dtype='i8', astype='i4')
        expect = 'Delta(dtype=int64, astype=int32)'
        actual = repr(codec)
        eq(expect, actual)


class TestFixedScaleOffset(CodecTests, unittest.TestCase):

    codec_id = 'fixedscaleoffset'

    arrs = [
        np.linspace(1000, 1001, 1000, dtype='f8'),
        np.random.normal(loc=1000, scale=1, size=1000).astype('f8'),
        np.linspace(1000, 1001, 1000, dtype='f8').reshape(100, 10),
        np.linspace(1000, 1001, 1000, dtype='f8').reshape(100, 10, order='F'),
        np.linspace(1000, 1001, 1000, dtype='f8').reshape(10, 10, 10),
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

    def test_repr(self):
        dtype = 'f8'
        astype = 'u1'
        codec = self.init_codec(scale=10, offset=1000, dtype=dtype,
                                astype=astype)
        expect = 'FixedScaleOffset(scale=10, offset=1000, dtype=float64, ' \
                 'astype=uint8)'
        actual = repr(codec)
        eq(expect, actual)


class TestQuantize(CodecTests, unittest.TestCase):

    codec_id = 'quantize'

    arrs = [
        np.linspace(100, 200, 1000, dtype='f8'),
        np.random.normal(loc=0, scale=1, size=1000).astype('f8'),
        np.linspace(100, 200, 1000, dtype='f8').reshape(100, 10),
        np.linspace(100, 200, 1000, dtype='f8').reshape(100, 10, order='F'),
        np.linspace(100, 200, 1000, dtype='f8').reshape(10, 10, 10),
    ]

    configs = [
        dict(digits=-1, dtype='f8', astype='f2'),
        dict(digits=1, dtype='f8', astype='f2'),
        dict(digits=5, dtype='f8', astype='f4'),
        dict(digits=12, dtype='f8', astype='f8'),
    ]

    def test_encode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            self._test_encode(arr, **config)

    def test_decode(self):
        for arr, config in itertools.product(self.arrs, self.configs):
            decimal = config['digits']
            self._test_decode_lossy(arr, decimal=decimal, **config)

    def test_errors(self):
        with assert_raises(ValueError):
            self.init_codec(digits=-1, dtype='i4')

    def test_repr(self):
        dtype = 'f8'
        astype = 'f4'
        codec = self.init_codec(digits=2, dtype=dtype, astype=astype)
        expect = 'Quantize(digits=2, dtype=float64, astype=float32)'
        actual = repr(codec)
        eq(expect, actual)


class TestPackBits(CodecTests, unittest.TestCase):

    codec_id = 'packbits'

    arr = np.random.randint(0, 2, size=1000, dtype=bool)

    def test_encode(self):
        for size in list(range(1, 17)) + [1000]:
            arr = self.arr[:size]
            self._test_encode(arr)

    def test_decode(self):
        for size in list(range(1, 17)) + [1000]:
            arr = self.arr[:size]
            self._test_decode_lossless(arr)

    def test_repr(self):
        codec = self.init_codec()
        expect = 'PackBits()'
        actual = repr(codec)
        eq(expect, actual)


class TestCategorize(CodecTests, unittest.TestCase):

    codec_id = 'categorize'
    labels = [b'foo', b'bar', b'baz', b'quux']
    labels_u = ['foo', 'bar', 'baz', 'quux']
    arrs = [
        np.random.choice(labels, size=1000),
        np.random.choice(labels, size=(100, 10)),
        np.random.choice(labels, size=(10, 10, 10)),
        np.random.choice(labels, size=1000).reshape(100, 10, order='F'),
        np.random.choice(labels_u, size=1000),
    ]

    def test_encode(self):
        for arr in self.arrs:
            self._test_encode(arr, labels=self.labels, dtype=arr.dtype,
                              astype='u1')

    def test_decode(self):
        for arr in self.arrs:
            self._test_decode_lossless(arr, labels=self.labels,
                                       dtype=arr.dtype, astype='u1')

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

    def test_repr(self):
        if not PY2:
            labels = ['foo', 'bar', 'baz', 'quux', 'spong']
            dtype = '|S3'
            astype = 'u1'
            codec = self.init_codec(labels=labels, dtype=dtype, astype=astype)
            expect = "Categorize(dtype=|S3, astype=uint8, " \
                     "labels=[b'foo', b'bar', b'baz', ...])"
            actual = repr(codec)
            eq(expect, actual)

    def test_errors(self):
        with assert_raises(ValueError):
            self.init_codec(labels=[0, 1, 2], dtype='i8')
        with assert_raises(ValueError):
            self.init_codec(labels=['foo', 'bar', 0], dtype='S3')
        with assert_raises(ValueError):
            self.init_codec(labels=['foo', 'bar', 0], dtype='U3')


def test_get_codec():

    with assert_raises(ValueError):
        get_codec({'id': 'foo'})
