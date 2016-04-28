# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

# import json
# import pickle
#
# from nose.tools import eq_ as eq, assert_is_instance
# import numpy as np
# from numpy.testing import assert_array_equal
#
# import zarr
# from zarr import meta
# from zarr.core import Array


# def test_arraystore():
#     data = dict()
#     attrs = dict()
#     meta = dict(shape=(100, 100), chunks=(10, 10), dtype='f4')
#     store = ArrayStore(meta, data, attrs)
#     z = Array(store)
#     eq(z.shape, (100, 100))
#     eq(z.chunks, (10, 10))
#     eq(z.dtype, np.float32)
#     assert_is_instance(z.cbytes, int)
#
#     assert z.store is store
#     assert store.meta is meta
#     assert store.data is data
#     assert store.attrs is attrs
#     eq(set(data), set())
#     z[:] = 1
#     eq(set(data), {'%d.%d' % (i, j) for i in range(10) for j in range(10)})
#
#     assert_array_equal(z[:5, :5], np.ones((5, 5), dtype='f4'))
#
#
# def test_pickle():
#     store = ArrayStore({}, {}, {},
#                       shape=(100, 100), chunks=(10, 10), dtype='f4')
#     x = Array(store)
#     x[:] = 1
#
#     x.attrs['hello'] = 'world'
#     y = pickle.loads(pickle.dumps(x))
#
#     assert x.store.meta == y.store.meta
#     assert x.attrs == y.attrs
#     assert_array_equal(x[:], y[:])
#
#
# def test_resize():
#     data = dict()
#     store = ArrayStore({}, data, {},
#                       shape=(100, 100), chunks=(10, 10), dtype='f4')
#     x = Array(store)
#     x[:] = 1
#     x.resize((50, 200))
#
#     assert x.store.data is data
#     eq(set(data) - {'meta', 'attrs'},
#        {'%d.%d' % (i, j) for i in range(5) for j in range(10)})
#
#
# def test_flush_metadata_on_resize():
#     x = Array(ArrayStore(shape=(100, 100), chunks=(10, 10), dtype='f4'))
#     x.resize((50, 200))
#     eq(meta.loads(x.store.data['meta']), x.store.meta)
#
#
# def test_flush_meta_to_data():
#     store = ArrayStore({}, {}, {},
#                       shape=(100, 100), chunks=(10, 10), dtype='f4')
#     x = Array(store)
#     x.attrs['hello'] = 'world'
#
#     store.flush()
#     eq(zarr.meta.loads(store.data['meta']), store.meta)
#     eq(json.loads(store.data['attrs']), store.attrs)
#
#
# def test_defaults():
#     template = zarr.empty(shape=(1000, 1000), chunks=(100, 100), dtype='i2')
#     template.attrs['one'] = '1'
#     template.attrs['two'] = '2'
#
#     data = {'meta': zarr.meta.dumps(template.store.meta),
#             'attrs': json.dumps(template.store.attrs)}
#     store = ArrayStore(data=data)
#
#     eq(store.meta, template.store.meta)
#     eq(store.attrs, template.store.attrs)
#
#     x = Array(store)
#     for attr in ['shape', 'chunks', 'dtype', 'cname', 'shuffle']:
#         eq(getattr(x, attr), getattr(template, attr))
#
#     # Test prefer explicit meta over stored metadata
#     store = ArrayStore({'chunks': (200, 200)},
#                        data,
#                        {'one': 100, 'three': 3},
#                        dtype='i4')
#
#     eq(store.meta['shape'], (1000, 1000))  # taken from data
#     eq(store.meta['chunks'], (200, 200))   # taken from given metadata
#     eq(store.meta['dtype'], 'i4')          # taken from kwargs
