# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit


from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_equal


from zarr.convenience import open, save, save_group, load, copy_store
from zarr.storage import atexit_rmtree
from zarr.core import Array
from zarr.hierarchy import Group


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
