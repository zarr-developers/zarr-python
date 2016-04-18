# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from numpy.testing import assert_array_equal


from zarr.blosc import compress, decompress


def test_round_trip():

    for use_context in True, False:

        a = np.arange(1000, dtype='i4')
        cdata = compress(a, b'blosclz', 5, 1, use_context)
        assert isinstance(cdata, bytes)
        assert len(cdata) < a.nbytes

        b = np.empty_like(a)
        decompress(cdata, b, use_context)
        assert_array_equal(a, b)
