# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from nose.tools import eq_ as eq, assert_is_none
import numpy as np


from zarr.meta import decode_metadata, encode_metadata


def test_decode():

    # typical
    b = b'''{
        "shape": [100],
        "chunks": [10],
        "dtype": "<f8",
        "compression": "zlib",
        "compression_opts": 1,
        "fill_value": null
        }'''
    meta = decode_metadata(b)
    eq((100,), meta['shape'])
    eq((10,), meta['chunks'])
    eq(np.dtype('<f8'), meta['dtype'])
    eq('zlib', meta['compression'])
    eq(1, meta['compression_opts'])
    assert_is_none(meta['fill_value'])



# from zarr.meta import dumps, loads
# import zarr
#
# from nose.tools import eq_ as eq


# def test_simple():
#     for dt in ['f8', [('a', 'f8')], [('a', 'f8'), ('b', 'i1')]]:
#         for compression in [{'cname': 'blosclz', 'shuffle': True},
#                             {'cname': 'zlib', 'clevel': 5},
#                             {'cname': None}]:
#             x = zarr.empty(shape=(1000, 1000), chunks=(100, 100), dtype=dt,
#                             **compression)
#             meta = x.store.meta
#             eq(loads(dumps(meta)), meta)
