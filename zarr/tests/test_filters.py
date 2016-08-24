# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest


import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_ as eq


from zarr.filters import DeltaFilter, ScaleOffsetFilter


class TestDeltaFilter(unittest.TestCase):

    def test_encode_decode(self):
        enc_dtype = 'u1'
        dec_dtype = 'f8'
        f = DeltaFilter(enc_dtype=enc_dtype, dec_dtype=dec_dtype)
        arr = np.arange(10, dtype=dec_dtype)

        # test encoding
        expect_enc = np.array([0] + ([1] * 9), dtype=enc_dtype)
        enc = f.encode(arr)
        assert_array_equal(expect_enc, enc)
        eq(np.dtype(enc_dtype), enc.dtype)

        # test decoding
        dec = f.decode(enc)
        assert_array_equal(arr, dec)
        eq(np.dtype(dec_dtype), dec.dtype)


class TestScaleOffsetFilter(unittest.TestCase):

    def test_encode_decode(self):
        enc_dtype = 'u1'
        dec_dtype = 'f8'
        f = ScaleOffsetFilter(scale=10, offset=1000, enc_dtype=enc_dtype,
                              dec_dtype=dec_dtype)
        arr = 1000 + np.arange(0, 100, 10, dtype=dec_dtype)

        # test encoding
        expected_enc = np.arange(10, dtype=enc_dtype)
        enc = f.encode(arr)
        assert_array_equal(expected_enc, enc)
        eq(np.dtype(enc_dtype), enc.dtype)

        # test decoding
        dec = f.decode(enc)
        assert_array_equal(arr, dec)
        eq(np.dtype(dec_dtype), dec.dtype)
