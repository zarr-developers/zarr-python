# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import eq_ as eq


from zarr.codecs import DeltaFilter, FixedScaleOffsetFilter, \
    QuantizeFilter, PackBitsFilter, CategorizeFilter, codec_registry
from zarr.creation import array
from zarr.compat import PY2
from zarr.util import buffer_tobytes


compression_setups = [
    (None, None),
    ('zlib', {}),
    ('bz2', {}),
    ('blosc', {})
]
if not PY2:
    compression_setups.append(('lzma', {}))


def test_array_with_delta_filter():

    # setup
    astype = 'u1'
    dtype = 'i8'
    filters = [DeltaFilter(astype=astype, dtype=dtype)]
    data = np.arange(100, dtype=dtype)

    for compression, compression_opts in compression_setups:
        print(compression, compression_opts)

        a = array(data, chunks=10, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_equal(data, a[:])

        # check chunks
        for i in range(10):
            cdata = a.store[str(i)]
            if a.compressor:
                chunk = a.compressor.decode(cdata)
            else:
                chunk = cdata
            actual = np.frombuffer(chunk, dtype=astype)
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

    for compression, compression_opts in compression_setups:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_almost_equal(data, a[:], decimal=1)

        # check chunks
        for i in range(6):
            cdata = a.store[str(i)]
            if a.compressor:
                chunk = a.compressor.decode(cdata)
            else:
                chunk = cdata
            actual = np.frombuffer(chunk, dtype=astype)
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_array_with_quantize_filter():

    # setup
    dtype = 'f8'
    digits = 3
    flt = QuantizeFilter(digits=digits, dtype=dtype)
    filters = [flt]
    data = np.linspace(0, 1, 34, dtype=dtype)

    for compression, compression_opts in compression_setups:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_almost_equal(data, a[:], decimal=digits)

        # check chunks
        for i in range(6):
            cdata = a.store[str(i)]
            if a.compressor:
                chunk = a.compressor.decode(cdata)
            else:
                chunk = cdata
            actual = np.frombuffer(chunk, dtype=dtype)
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_array_with_packbits_filter():

    # setup
    flt = PackBitsFilter()
    filters = [flt]
    data = np.random.randint(0, 2, size=100, dtype=bool)

    for compression, compression_opts in compression_setups:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_equal(data, a[:])

        # check chunks
        for i in range(20):
            cdata = a.store[str(i)]
            if a.compressor:
                chunk = a.compressor.decode(cdata)
            else:
                chunk = cdata
            actual = np.frombuffer(chunk, dtype='u1')
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_array_with_categorize_filter():

    # setup
    data = np.random.choice([b'foo', b'bar', b'baz'], size=100)
    flt = CategorizeFilter(dtype=data.dtype, labels=['foo', 'bar', 'baz'])
    filters = [flt]

    for compression, compression_opts in compression_setups:
        print(compression, compression_opts)

        a = array(data, chunks=5, compression=compression,
                  compression_opts=compression_opts, filters=filters)

        # check round-trip
        assert_array_equal(data, a[:])

        # check chunks
        for i in range(20):
            cdata = a.store[str(i)]
            if a.compressor:
                chunk = a.compressor.decode(cdata)
            else:
                chunk = cdata
            actual = np.frombuffer(chunk, dtype='u1')
            expect = flt.encode(data[i*5:(i*5)+5])
            assert_array_equal(expect, actual)


def test_compressor_as_filter():
    for compression, compression_opts in compression_setups:
        if compression is None:
            # skip
            continue

        # setup compressor
        compressor_cls = codec_registry[compression]
        compressor = compressor_cls(**compression_opts)

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
