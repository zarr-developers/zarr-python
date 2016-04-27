# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import numpy as np
from numpy.testing import assert_array_equal


from zarr import blosc


def test_round_trip():

    for use_context in True, False:

        blosc.use_context(use_context)

        a = np.arange(1000, dtype='i4')
        cdata = blosc.compress(a, b'blosclz', 5, 1)
        assert isinstance(cdata, bytes)
        assert len(cdata) < a.nbytes

        b = np.empty_like(a)
        blosc.decompress(cdata, b)
        assert_array_equal(a, b)
