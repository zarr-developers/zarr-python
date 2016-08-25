# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import array


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq, assert_raises


from zarr.compressors import get_compressor_cls
from zarr.util import buffer_size, buffer_tobytes


class CompressorTests(object):

    compression = None

    @classmethod
    def init_compressor(cls, compression_opts=None):
        compression_cls = get_compressor_cls(cls.compression)
        compression_opts = compression_cls.normalize_opts(compression_opts)
        compressor = compression_cls(compression_opts)
        return compressor

    def _test_compress_decompress(self, compression_opts=None):

        comp = self.init_compressor(compression_opts)
        arr = np.arange(1000, dtype='i1')

        # compression should be able to compress data from any object
        # exporting the buffer protocol, as well as array.array in PY2

        # test compression of numpy array
        cdata_first = comp.compress(arr)
        cbytes = buffer_size(cdata_first)
        assert cbytes <= arr.nbytes
        cdata_bytes = buffer_tobytes(cdata_first)

        # N.B., watch out below with blosc compressor, if the itemsize of
        # the source buffer is different then the results of compression
        # will be different. Hence testing on all data with itemsize 1.

        # test compression of raw bytes
        data = arr.tobytes()
        cdata = comp.compress(data)
        actual = buffer_tobytes(cdata)
        eq(cdata_bytes, actual)

        # test compression of array.array
        data = array.array('b', arr.tobytes())
        cdata = comp.compress(data)
        actual = buffer_tobytes(cdata)
        eq(cdata_bytes, actual)

        # decompression should be able to compress data from any object
        # exporting the buffer protocol, as well as array.array in PY2

        # test decompression of raw bytes
        data = comp.decompress(cdata_bytes)
        assert_array_equal(arr, np.frombuffer(data, dtype=arr.dtype))

        # test decompression of array.array
        cdata = array.array('b', cdata_bytes)
        data = comp.decompress(cdata)
        assert_array_equal(arr, np.frombuffer(data, dtype=arr.dtype))

        # test decompression of numpy array
        cdata = np.frombuffer(cdata_bytes, dtype='u1')
        data = comp.decompress(cdata)
        assert_array_equal(arr, np.frombuffer(data, dtype=arr.dtype))

        # test decompression into destination
        dest = np.empty_like(arr)
        comp.decompress(cdata_bytes, dest)
        assert_array_equal(arr, dest)

    def test_compress_decompress_default(self):
        self._test_compress_decompress()


try:
    from zarr import blosc  # flake8: noqa
except ImportError:  # pragma: no cover
    print('Blosc not available, skipping Blosc compressor tests')
else:

    class TestBloscCompressor(unittest.TestCase, CompressorTests):

        compression = 'blosc'

        def test_normalize_opts(self):
            cls = get_compressor_cls(self.compression)

            # test defaults
            opts = cls.normalize_opts(None)
            default_opts = dict(
                cname=cls.default_cname,
                clevel=cls.default_clevel,
                shuffle=cls.default_shuffle
            )
            eq(default_opts, opts)

            # test invalid args
            with assert_raises(ValueError):
                cls.normalize_opts(dict(cname='foo'))
            with assert_raises(ValueError):
                cls.normalize_opts(dict(clevel=10))
            with assert_raises(ValueError):
                cls.normalize_opts(dict(shuffle=3))


class TestZlibCompressor(unittest.TestCase, CompressorTests):

    compression = 'zlib'

    def test_normalize_opts(self):
        cls = get_compressor_cls(self.compression)

        # test defaults
        opts = cls.normalize_opts(None)
        eq(cls.default_level, opts)

        # test invalid args
        with assert_raises(ValueError):
            cls.normalize_opts(10)


class TestBZ2Compressor(unittest.TestCase, CompressorTests):

    compression = 'bz2'

    def test_normalize_opts(self):
        cls = get_compressor_cls(self.compression)

        # test defaults
        opts = cls.normalize_opts(None)
        eq(cls.default_level, opts)

        # test invalid args
        with assert_raises(ValueError):
            cls.normalize_opts(10)


try:
    import lzma
except ImportError:  # pragma: no cover
    print('LZMA not available, skipping LZMA compressor tests')
else:

    class TestLZMACompressor(unittest.TestCase, CompressorTests):

        compression = 'lzma'

        def test_normalize_opts(self):
            cls = get_compressor_cls(self.compression)

            # test defaults
            opts = cls.normalize_opts(None)
            default_opts = dict(
                format=cls.default_format,
                check=cls.default_check,
                preset=cls.default_preset,
                filters=cls.default_filters,
            )
            eq(default_opts, opts)

            # test invalid args
            with assert_raises(ValueError):
                cls.normalize_opts(dict(format=100))
            with assert_raises(ValueError):
                cls.normalize_opts(dict(check=100))
            with assert_raises(ValueError):
                cls.normalize_opts(dict(preset=100))

            # test preset overridden if filters
            opts_with_filter = cls.normalize_opts(
                dict(preset=1, filters=[dict(id=lzma.FILTER_LZMA2, preset=9)])
            )
            opts = cls.normalize_opts(opts_with_filter)
            eq(None, opts['preset'])

        def test_compress_decompress_raw(self):
            opts = dict(format=lzma.FORMAT_RAW,
                        filters=[dict(id=lzma.FILTER_LZMA2, preset=1)])
            self._test_compress_decompress(opts)


class TestNoCompressor(unittest.TestCase, CompressorTests):

    compression = 'none'

    def test_normalize_opts(self):
        cls = get_compressor_cls(self.compression)

        # test defaults
        opts = cls.normalize_opts(None)
        eq(None, opts)

        # test invalid args
        with assert_raises(ValueError):
            cls.normalize_opts(9)


def test_get_compressor_cls():
    # expect ValueError, more friendly
    with assert_raises(ValueError):
        get_compressor_cls('foo')
