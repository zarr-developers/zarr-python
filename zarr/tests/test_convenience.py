# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit
import os


from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_equal
from numcodecs import Zlib
import pytest


from zarr.convenience import open, save, save_group, load, copy_store, copy
from zarr.storage import atexit_rmtree
from zarr.core import Array
from zarr.hierarchy import Group, group


def test_open_array():

    store = tempfile.mkdtemp()
    atexit.register(atexit_rmtree, store)

    # open array, create if doesn't exist
    z = open(store, mode='a', shape=100)
    assert isinstance(z, Array)
    assert z.shape == (100,)

    # open array, overwrite
    z = open(store, mode='w', shape=200)
    assert isinstance(z, Array)
    assert z.shape == (200,)

    # open array, read-only
    z = open(store, mode='r')
    assert isinstance(z, Array)
    assert z.shape == (200,)
    assert z.read_only

    # path not found
    with assert_raises(ValueError):
        open('doesnotexist', mode='r')


def test_open_group():

    store = tempfile.mkdtemp()
    atexit.register(atexit_rmtree, store)

    # open group, create if doesn't exist
    g = open(store, mode='a')
    g.create_group('foo')
    assert isinstance(g, Group)
    assert 'foo' in g

    # open group, overwrite
    g = open(store, mode='w')
    assert isinstance(g, Group)
    assert 'foo' not in g

    # open group, read-only
    g = open(store, mode='r')
    assert isinstance(g, Group)
    assert g.read_only


def test_save_errors():
    with assert_raises(ValueError):
        # no arrays provided
        save_group('data/group.zarr')
    with assert_raises(ValueError):
        # no arrays provided
        save('data/group.zarr')


def test_lazy_loader():
    foo = np.arange(100)
    bar = np.arange(100, 0, -1)
    store = 'data/group.zarr'
    save(store, foo=foo, bar=bar)
    loader = load(store)
    assert 'foo' in loader
    assert 'bar' in loader
    assert 'baz' not in loader
    assert len(loader) == 2
    assert sorted(loader) == ['bar', 'foo']
    assert_array_equal(foo, loader['foo'])
    assert_array_equal(bar, loader['bar'])


def test_copy_store():

    # no paths
    source = dict()
    source['foo'] = b'xxx'
    source['bar'] = b'yyy'
    dest = dict()
    copy_store(source, dest)
    assert len(dest) == 2
    for key in source:
        assert source[key] == dest[key]

    # with source path
    source = dict()
    source['foo'] = b'xxx'
    source['bar/baz'] = b'yyy'
    source['bar/qux'] = b'zzz'
    # paths should be normalized
    for source_path in 'bar', 'bar/', '/bar', '/bar/':
        dest = dict()
        copy_store(source, dest, source_path=source_path)
        assert len(dest) == 2
        for key in source:
            if key.startswith('bar/'):
                dest_key = key.split('bar/')[1]
                assert source[key] == dest[dest_key]
            else:
                assert key not in dest

    # with dest path
    source = dict()
    source['foo'] = b'xxx'
    source['bar/baz'] = b'yyy'
    source['bar/qux'] = b'zzz'
    # paths should be normalized
    for dest_path in 'new', 'new/', '/new', '/new/':
        dest = dict()
        copy_store(source, dest, dest_path=dest_path)
        assert len(dest) == 3
        for key in source:
            dest_key = 'new/' + key
            assert source[key] == dest[dest_key]

    # with source and dest path
    source = dict()
    source['foo'] = b'xxx'
    source['bar/baz'] = b'yyy'
    source['bar/qux'] = b'zzz'
    # paths should be normalized
    for source_path in 'bar', 'bar/', '/bar', '/bar/':
        for dest_path in 'new', 'new/', '/new', '/new/':
            dest = dict()
            copy_store(source, dest, source_path=source_path, dest_path=dest_path)
            assert len(dest) == 2
            for key in source:
                if key.startswith('bar/'):
                    dest_key = 'new/' + key.split('bar/')[1]
                    assert source[key] == dest[dest_key]
                else:
                    assert key not in dest
                    assert ('new/' + key) not in dest

    # with excludes/includes
    source = dict()
    source['foo'] = b'xxx'
    source['bar/baz'] = b'yyy'
    source['bar/qux'] = b'zzz'
    # single excludes
    dest = dict()
    excludes = 'f.*'
    copy_store(source, dest, excludes=excludes)
    assert len(dest) == 2
    assert 'foo' not in dest
    # multiple excludes
    dest = dict()
    excludes = 'b.z', '.*x'
    copy_store(source, dest, excludes=excludes)
    assert len(dest) == 1
    assert 'foo' in dest
    assert 'bar/baz' not in dest
    assert 'bar/qux' not in dest
    # excludes and includes
    dest = dict()
    excludes = 'b.*'
    includes = '.*x'
    copy_store(source, dest, excludes=excludes, includes=includes)
    assert len(dest) == 2
    assert 'foo' in dest
    assert 'bar/baz' not in dest
    assert 'bar/qux' in dest


def _test_copy(new_source, new_dest):

    source = new_source()
    foo = source.create_group('foo')
    foo.attrs['experiment'] = 'weird science'
    baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
    baz.attrs['units'] = 'metres'
    spam = source.create_dataset('spam', data=np.arange(100, 200), chunks=(30,))

    # copy array with default options
    dest = new_dest()
    copy(source['foo/bar/baz'], dest)
    a = dest['baz']  # defaults to use source name
    assert a.dtype == baz.dtype
    assert a.shape == baz.shape
    assert a.chunks == baz.chunks
    if hasattr(a, 'compressor') and hasattr(baz, 'compressor'):
        assert a.compressor == baz.compressor
    assert_array_equal(a[:], baz[:])
    assert a.attrs['units'] == 'metres'

    # copy array with name
    dest = new_dest()
    copy(source['foo/bar/baz'], dest, name='qux')
    assert 'baz' not in dest
    a = dest['qux']
    assert a.dtype == baz.dtype
    assert a.shape == baz.shape
    assert a.chunks == baz.chunks
    if hasattr(a, 'compressor') and hasattr(baz, 'compressor'):
        assert a.compressor == baz.compressor
    assert_array_equal(a[:], baz[:])
    assert a.attrs['units'] == 'metres'

    # copy array, provide creation options
    dest = new_dest()
    compressor = Zlib(9)
    if isinstance(dest, Group):
        copy(source['foo/bar/baz'], dest, without_attrs=True, compressor=compressor,
             chunks=True)
    else:
        copy(source['foo/bar/baz'], dest, without_attrs=True, compression='gzip',
             compression_opts=9, chunks=True)
    a = dest['baz']
    assert a.dtype == baz.dtype
    assert a.shape == baz.shape
    assert a.chunks != baz.chunks  # autochunking was requested
    if hasattr(a, 'compressor'):
        assert compressor == a.compressor
        if hasattr(baz, 'compressor'):
            assert a.compressor != baz.compressor
    else:
        assert a.compression == 'gzip'
        assert a.compression_opts == 9
    assert_array_equal(a[:], baz[:])
    assert 'units' not in a.attrs

    # copy group, default options
    dest = new_dest()
    copy(source['foo'], dest)
    g = dest['foo']  # defaults to use source name
    assert g.attrs['experiment'] == 'weird science'
    a = g['bar/baz']
    assert a.dtype == baz.dtype
    assert a.shape == baz.shape
    assert a.chunks == baz.chunks
    if hasattr(a, 'compressor') and hasattr(baz, 'compressor'):
        assert a.compressor == baz.compressor
    assert_array_equal(a[:], baz[:])
    assert a.attrs['units'] == 'metres'

    # copy group, non-default options
    dest = new_dest()
    copy(source['foo'], dest, name='qux', without_attrs=True)
    assert 'foo' not in dest
    g = dest['qux']
    assert 'experiment' not in g.attrs
    a = g['bar/baz']
    assert a.dtype == baz.dtype
    assert a.shape == baz.shape
    assert a.chunks == baz.chunks
    if hasattr(a, 'compressor') and hasattr(baz, 'compressor'):
        assert a.compressor == baz.compressor
    assert_array_equal(a[:], baz[:])
    assert 'units' not in a.attrs

    # copy group, shallow
    dest = new_dest()
    copy(source, dest, name='eggs', shallow=True)
    assert 'eggs' in dest
    eggs = dest['eggs']
    assert 'spam' in eggs
    a = eggs['spam']
    assert a.dtype == spam.dtype
    assert a.shape == spam.shape
    assert a.chunks == spam.chunks
    if hasattr(a, 'compressor') and hasattr(spam, 'compressor'):
        assert a.compressor == spam.compressor
    assert_array_equal(a[:], spam[:])
    assert 'foo' not in eggs
    assert 'bar' not in eggs


def test_copy_zarr_zarr():
    # zarr -> zarr
    _test_copy(group, group)


try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False


def temp_h5f():
    fn = tempfile.mktemp()
    atexit.register(os.remove, fn)
    h5f = h5py.File(fn, mode='w')
    return h5f


@pytest.mark.skipif(not have_h5py, reason='h5py not installed')
def test_copy_h5py_zarr():
    # h5py -> zarr
    _test_copy(temp_h5f, group)


@pytest.mark.skipif(not have_h5py, reason='h5py not installed')
def test_copy_zarr_h5py():
    # zarr -> h5py
    _test_copy(group, temp_h5f)
