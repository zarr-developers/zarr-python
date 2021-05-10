import atexit
import os.path
import shutil
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from zarr.codecs import Zlib
from zarr.core import Array
from zarr.creation import (array, create, empty, empty_like, full, full_like,
                           ones, ones_like, open_array, open_like, zeros,
                           zeros_like)
from zarr.hierarchy import open_group
from zarr.n5 import N5Store
from zarr.storage import DirectoryStore
from zarr.sync import ThreadSynchronizer


# something bcolz-like
class MockBcolzArray(object):

    def __init__(self, data, chunklen):
        self.data = data
        self.chunklen = chunklen

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __getitem__(self, item):
        return self.data[item]


# something h5py-like
class MockH5pyDataset(object):

    def __init__(self, data, chunks):
        self.data = data
        self.chunks = chunks

    def __getattr__(self, item):
        return getattr(self.data, item)

    def __getitem__(self, item):
        return self.data[item]


def test_array():

    # with numpy array
    a = np.arange(100)
    z = array(a, chunks=10)
    assert a.shape == z.shape
    assert a.dtype == z.dtype
    assert_array_equal(a, z[:])

    # with array-like
    a = list(range(100))
    z = array(a, chunks=10)
    assert (100,) == z.shape
    assert np.asarray(a).dtype == z.dtype
    assert_array_equal(np.asarray(a), z[:])

    # with another zarr array
    z2 = array(z)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert_array_equal(z[:], z2[:])

    # with chunky array-likes

    b = np.arange(1000).reshape(100, 10)
    c = MockBcolzArray(b, 10)
    z3 = array(c)
    assert c.shape == z3.shape
    assert (10, 10) == z3.chunks

    b = np.arange(1000).reshape(100, 10)
    c = MockH5pyDataset(b, chunks=(10, 2))
    z4 = array(c)
    assert c.shape == z4.shape
    assert (10, 2) == z4.chunks

    c = MockH5pyDataset(b, chunks=None)
    z5 = array(c)
    assert c.shape == z5.shape
    assert isinstance(z5.chunks, tuple)

    # with dtype=None
    a = np.arange(100, dtype='i4')
    z = array(a, dtype=None)
    assert_array_equal(a[:], z[:])
    assert a.dtype == z.dtype

    # with dtype=something else
    a = np.arange(100, dtype='i4')
    z = array(a, dtype='i8')
    assert_array_equal(a[:], z[:])
    assert np.dtype('i8') == z.dtype


def test_empty():
    z = empty(100, chunks=10)
    assert (100,) == z.shape
    assert (10,) == z.chunks


def test_zeros():
    z = zeros(100, chunks=10)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.zeros(100), z[:])


def test_ones():
    z = ones(100, chunks=10)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.ones(100), z[:])


def test_full():
    z = full(100, chunks=10, fill_value=42, dtype='i4')
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42, dtype='i4'), z[:])

    # nan
    z = full(100, chunks=10, fill_value=np.nan, dtype='f8')
    assert np.all(np.isnan(z[:]))

    # NaT
    z = full(100, chunks=10, fill_value='NaT', dtype='M8[s]')
    assert np.all(np.isnat(z[:]))
    z = full(100, chunks=10, fill_value='NaT', dtype='m8[s]')
    assert np.all(np.isnat(z[:]))

    # byte string dtype
    v = b'xxx'
    z = full(100, chunks=10, fill_value=v, dtype='S3')
    assert v == z[0]
    a = z[...]
    assert z.dtype == a.dtype
    assert v == a[0]
    assert np.all(a == v)

    # unicode string dtype
    v = 'xxx'
    z = full(100, chunks=10, fill_value=v, dtype='U3')
    assert v == z[0]
    a = z[...]
    assert z.dtype == a.dtype
    assert v == a[0]
    assert np.all(a == v)

    # bytes fill value / unicode dtype
    v = b'xxx'
    with pytest.raises(ValueError):
        full(100, chunks=10, fill_value=v, dtype='U3')


def test_open_array():

    store = 'data/array.zarr'

    # mode == 'w'
    z = open_array(store, mode='w', shape=100, chunks=10)
    z[:] = 42
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])

    # mode in 'r', 'r+'
    open_group('data/group.zarr', mode='w')
    for mode in 'r', 'r+':
        with pytest.raises(ValueError):
            open_array('doesnotexist', mode=mode)
        with pytest.raises(ValueError):
            open_array('data/group.zarr', mode=mode)
    z = open_array(store, mode='r')
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])
    with pytest.raises(PermissionError):
        z[:] = 43
    z = open_array(store, mode='r+')
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])
    z[:] = 43
    assert_array_equal(np.full(100, fill_value=43), z[:])

    # mode == 'a'
    shutil.rmtree(store)
    z = open_array(store, mode='a', shape=100, chunks=10)
    z[:] = 42
    assert isinstance(z, Array)
    assert isinstance(z.store, DirectoryStore)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])
    with pytest.raises(ValueError):
        open_array('data/group.zarr', mode='a')

    # mode in 'w-', 'x'
    for mode in 'w-', 'x':
        shutil.rmtree(store)
        z = open_array(store, mode=mode, shape=100, chunks=10)
        z[:] = 42
        assert isinstance(z, Array)
        assert isinstance(z.store, DirectoryStore)
        assert (100,) == z.shape
        assert (10,) == z.chunks
        assert_array_equal(np.full(100, fill_value=42), z[:])
        with pytest.raises(ValueError):
            open_array(store, mode=mode)
        with pytest.raises(ValueError):
            open_array('data/group.zarr', mode=mode)

    # with synchronizer
    z = open_array(store, synchronizer=ThreadSynchronizer())
    assert isinstance(z, Array)

    # with path
    z = open_array(store, shape=100, path='foo/bar', mode='w')
    assert isinstance(z, Array)
    assert 'foo/bar' == z.path

    # with chunk store
    meta_store = 'data/meta.zarr'
    chunk_store = 'data/chunks.zarr'
    z = open_array(store=meta_store, chunk_store=chunk_store, shape=11, mode='w')
    z[:] = 42
    assert os.path.abspath(meta_store) == z.store.path
    assert os.path.abspath(chunk_store) == z.chunk_store.path

    # for N5 store
    store = 'data/array.n5'
    z = open_array(store, mode='w', shape=100, chunks=10)
    z[:] = 42
    assert isinstance(z, Array)
    assert isinstance(z.store, N5Store)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert_array_equal(np.full(100, fill_value=42), z[:])

    store = 'data/group.n5'
    z = open_group(store, mode='w')
    i = z.create_group('inner')
    a = i.zeros("array", shape=100, chunks=10)
    a[:] = 42

    # Edit inner/attributes.json to not include "n5"
    with open('data/group.n5/inner/attributes.json', 'w') as o:
        o.write("{}")

    # Re-open
    a = open_group(store)["inner"]["array"]
    assert isinstance(a, Array)
    assert isinstance(z.store, N5Store)
    assert (100,) == a.shape
    assert (10,) == a.chunks
    assert_array_equal(np.full(100, fill_value=42), a[:])


def test_empty_like():

    # zarr array
    z = empty(100, chunks=10, dtype='f4', compressor=Zlib(5),
              order='F')
    z2 = empty_like(z)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order

    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = empty_like(a)
    assert a.shape == z3.shape
    assert (100,) == z3.chunks
    assert a.dtype == z3.dtype
    assert z3.fill_value is None

    # something slightly silly
    a = [0] * 100
    z3 = empty_like(a, shape=200)
    assert (200,) == z3.shape

    # other array-likes
    b = np.arange(1000).reshape(100, 10)
    c = MockBcolzArray(b, 10)
    z = empty_like(c)
    assert b.shape == z.shape
    assert (10, 10) == z.chunks
    c = MockH5pyDataset(b, chunks=(10, 2))
    z = empty_like(c)
    assert b.shape == z.shape
    assert (10, 2) == z.chunks
    c = MockH5pyDataset(b, chunks=None)
    z = empty_like(c)
    assert b.shape == z.shape
    assert isinstance(z.chunks, tuple)


def test_zeros_like():
    # zarr array
    z = zeros(100, chunks=10, dtype='f4', compressor=Zlib(5),
              order='F')
    z2 = zeros_like(z)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order
    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = zeros_like(a, chunks=10)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 0 == z3.fill_value


def test_ones_like():
    # zarr array
    z = ones(100, chunks=10, dtype='f4', compressor=Zlib(5),
             order='F')
    z2 = ones_like(z)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order
    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = ones_like(a, chunks=10)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 1 == z3.fill_value


def test_full_like():
    z = full(100, chunks=10, dtype='f4', compressor=Zlib(5),
             fill_value=42, order='F')
    z2 = full_like(z)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order
    # numpy array
    a = np.empty(100, dtype='f4')
    z3 = full_like(a, chunks=10, fill_value=42)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 42 == z3.fill_value
    with pytest.raises(TypeError):
        # fill_value missing
        full_like(a, chunks=10)


def test_open_like():
    # zarr array
    path = tempfile.mktemp()
    atexit.register(shutil.rmtree, path)
    z = full(100, chunks=10, dtype='f4', compressor=Zlib(5),
             fill_value=42, order='F')
    z2 = open_like(z, path)
    assert z.shape == z2.shape
    assert z.chunks == z2.chunks
    assert z.dtype == z2.dtype
    assert z.compressor.get_config() == z2.compressor.get_config()
    assert z.fill_value == z2.fill_value
    assert z.order == z2.order
    # numpy array
    path = tempfile.mktemp()
    atexit.register(shutil.rmtree, path)
    a = np.empty(100, dtype='f4')
    z3 = open_like(a, path, chunks=10)
    assert a.shape == z3.shape
    assert (10,) == z3.chunks
    assert a.dtype == z3.dtype
    assert 0 == z3.fill_value


def test_create():

    # defaults
    z = create(100)
    assert isinstance(z, Array)
    assert (100,) == z.shape
    assert (100,) == z.chunks  # auto-chunks
    assert np.dtype(None) == z.dtype
    assert 'blosc' == z.compressor.codec_id
    assert 0 == z.fill_value

    # all specified
    z = create(100, chunks=10, dtype='i4', compressor=Zlib(1),
               fill_value=42, order='F')
    assert isinstance(z, Array)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert np.dtype('i4') == z.dtype
    assert 'zlib' == z.compressor.codec_id
    assert 1 == z.compressor.level
    assert 42 == z.fill_value
    assert 'F' == z.order

    # with synchronizer
    synchronizer = ThreadSynchronizer()
    z = create(100, chunks=10, synchronizer=synchronizer)
    assert isinstance(z, Array)
    assert (100,) == z.shape
    assert (10,) == z.chunks
    assert synchronizer is z.synchronizer

    # don't allow string as compressor arg
    with pytest.raises(ValueError):
        create(100, chunks=10, compressor='zlib')

    # h5py compatibility

    z = create(100, compression='zlib', compression_opts=9)
    assert 'zlib' == z.compressor.codec_id
    assert 9 == z.compressor.level

    z = create(100, compression='default')
    assert 'blosc' == z.compressor.codec_id

    # errors
    with pytest.raises(ValueError):
        # bad compression argument
        create(100, compression=1)
    with pytest.raises(ValueError):
        # bad fill value
        create(100, dtype='i4', fill_value='foo')

    # auto chunks
    z = create(1000000000, chunks=True)
    assert z.chunks[0] < z.shape[0]
    z = create(1000000000, chunks=None)  # backwards-compatibility
    assert z.chunks[0] < z.shape[0]
    # no chunks
    z = create(1000000000, chunks=False)
    assert z.chunks == z.shape


def test_compression_args():

    z = create(100, compression='zlib', compression_opts=9)
    assert isinstance(z, Array)
    assert 'zlib' == z.compressor.codec_id
    assert 9 == z.compressor.level

    # 'compressor' overrides 'compression'
    z = create(100, compressor=Zlib(9), compression='bz2', compression_opts=1)
    assert isinstance(z, Array)
    assert 'zlib' == z.compressor.codec_id
    assert 9 == z.compressor.level

    # 'compressor' ignores 'compression_opts'
    z = create(100, compressor=Zlib(9), compression_opts=1)
    assert isinstance(z, Array)
    assert 'zlib' == z.compressor.codec_id
    assert 9 == z.compressor.level

    with pytest.warns(UserWarning):
        # 'compressor' overrides 'compression'
        create(100, compressor=Zlib(9), compression='bz2', compression_opts=1)
    with pytest.warns(UserWarning):
        # 'compressor' ignores 'compression_opts'
        create(100, compressor=Zlib(9), compression_opts=1)


def test_create_read_only():
    # https://github.com/alimanfoo/zarr/issues/151

    # create an array initially read-only, then enable writing
    z = create(100, read_only=True)
    assert z.read_only
    with pytest.raises(PermissionError):
        z[:] = 42
    z.read_only = False
    z[:] = 42
    assert np.all(z[...] == 42)
    z.read_only = True
    with pytest.raises(PermissionError):
        z[:] = 0

    # this is subtly different, but here we want to create an array with data, and then
    # have it be read-only
    a = np.arange(100)
    z = array(a, read_only=True)
    assert_array_equal(a, z[...])
    assert z.read_only
    with pytest.raises(PermissionError):
        z[:] = 42
