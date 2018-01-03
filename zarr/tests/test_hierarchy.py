# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import tempfile
import atexit
import shutil
import textwrap
import os
import pickle
import warnings


import numpy as np
from numpy.testing import assert_array_equal
import pytest


from zarr.storage import (DictStore, DirectoryStore, ZipStore, init_group, init_array,
                          array_meta_key, group_meta_key, atexit_rmtree,
                          NestedDirectoryStore, DBMStore, LMDBStore, atexit_rmglob,
                          LRUStoreCache)
from zarr.core import Array
from zarr.compat import PY2, text_type
from zarr.hierarchy import Group, group, open_group
from zarr.attrs import Attributes
from zarr.errors import PermissionError
from zarr.creation import open_array
from zarr.util import InfoReporter
from numcodecs import Zlib


# needed for PY2/PY3 consistent behaviour
if PY2:  # pragma: py3 no cover
    warnings.resetwarnings()
    warnings.simplefilter('always')


# noinspection PyStatementEffect
class TestGroup(unittest.TestCase):

    @staticmethod
    def create_store():
        # can be overridden in sub-classes
        return dict(), None

    def create_group(self, store=None, path=None, read_only=False,
                     chunk_store=None, synchronizer=None):
        # can be overridden in sub-classes
        if store is None:
            store, chunk_store = self.create_store()
        init_group(store, path=path, chunk_store=chunk_store)
        g = Group(store, path=path, read_only=read_only,
                  chunk_store=chunk_store, synchronizer=synchronizer)
        return g

    def test_group_init_1(self):
        store, chunk_store = self.create_store()
        g = self.create_group(store, chunk_store=chunk_store)
        assert store is g.store
        if chunk_store is None:
            assert store is g.chunk_store
        else:
            assert chunk_store is g.chunk_store
        assert not g.read_only
        assert '' == g.path
        assert '/' == g.name
        assert '' == g.basename
        assert isinstance(g.attrs, Attributes)
        g.attrs['foo'] = 'bar'
        assert g.attrs['foo'] == 'bar'
        assert isinstance(g.info, InfoReporter)
        assert isinstance(repr(g.info), str)
        assert isinstance(g.info._repr_html_(), str)

    def test_group_init_2(self):
        store, chunk_store = self.create_store()
        g = self.create_group(store, chunk_store=chunk_store,
                              path='/foo/bar/', read_only=True)
        assert store is g.store
        assert g.read_only
        assert 'foo/bar' == g.path
        assert '/foo/bar' == g.name
        assert 'bar' == g.basename
        assert isinstance(g.attrs, Attributes)

    def test_group_init_errors_1(self):
        store, chunk_store = self.create_store()
        # group metadata not initialized
        with pytest.raises(ValueError):
            Group(store, chunk_store=chunk_store)

    def test_group_init_errors_2(self):
        store, chunk_store = self.create_store()
        init_array(store, shape=1000, chunks=100, chunk_store=chunk_store)
        # array blocks group
        with pytest.raises(ValueError):
            Group(store, chunk_store=chunk_store)

    def test_create_group(self):
        g1 = self.create_group()

        # check root group
        assert '' == g1.path
        assert '/' == g1.name

        # create level 1 child group
        g2 = g1.create_group('foo')
        assert isinstance(g2, Group)
        assert 'foo' == g2.path
        assert '/foo' == g2.name

        # create level 2 child group
        g3 = g2.create_group('bar')
        assert isinstance(g3, Group)
        assert 'foo/bar' == g3.path
        assert '/foo/bar' == g3.name

        # create level 3 child group
        g4 = g1.create_group('foo/bar/baz')
        assert isinstance(g4, Group)
        assert 'foo/bar/baz' == g4.path
        assert '/foo/bar/baz' == g4.name

        # create level 3 group via root
        g5 = g4.create_group('/a/b/c/')
        assert isinstance(g5, Group)
        assert 'a/b/c' == g5.path
        assert '/a/b/c' == g5.name

        # test non-str keys
        class Foo(object):

            def __init__(self, s):
                self.s = s

            def __str__(self):
                return self.s

        o = Foo('test/object')
        go = g1.create_group(o)
        assert isinstance(go, Group)
        assert 'test/object' == go.path
        go = g1.create_group(b'test/bytes')
        assert isinstance(go, Group)
        assert 'test/bytes' == go.path

        # test bad keys
        with pytest.raises(ValueError):
            g1.create_group('foo')  # already exists
        with pytest.raises(ValueError):
            g1.create_group('a/b/c')  # already exists
        with pytest.raises(ValueError):
            g4.create_group('/a/b/c')  # already exists
        with pytest.raises(ValueError):
            g1.create_group('')
        with pytest.raises(ValueError):
            g1.create_group('/')
        with pytest.raises(ValueError):
            g1.create_group('//')

        # multi
        g6, g7 = g1.create_groups('y', 'z')
        assert isinstance(g6, Group)
        assert g6.path == 'y'
        assert isinstance(g7, Group)
        assert g7.path == 'z'

    def test_require_group(self):
        g1 = self.create_group()

        # test creation
        g2 = g1.require_group('foo')
        assert isinstance(g2, Group)
        assert 'foo' == g2.path
        g3 = g2.require_group('bar')
        assert isinstance(g3, Group)
        assert 'foo/bar' == g3.path
        g4 = g1.require_group('foo/bar/baz')
        assert isinstance(g4, Group)
        assert 'foo/bar/baz' == g4.path
        g5 = g4.require_group('/a/b/c/')
        assert isinstance(g5, Group)
        assert 'a/b/c' == g5.path

        # test when already created
        g2a = g1.require_group('foo')
        assert g2 == g2a
        assert g2.store is g2a.store
        g3a = g2a.require_group('bar')
        assert g3 == g3a
        assert g3.store is g3a.store
        g4a = g1.require_group('foo/bar/baz')
        assert g4 == g4a
        assert g4.store is g4a.store
        g5a = g4a.require_group('/a/b/c/')
        assert g5 == g5a
        assert g5.store is g5a.store

        # test path normalization
        assert g1.require_group('quux') == g1.require_group('/quux/')

        # multi
        g6, g7 = g1.require_groups('y', 'z')
        assert isinstance(g6, Group)
        assert g6.path == 'y'
        assert isinstance(g7, Group)
        assert g7.path == 'z'

    def test_create_dataset(self):
        g = self.create_group()

        # create as immediate child
        d1 = g.create_dataset('foo', shape=1000, chunks=100)
        assert isinstance(d1, Array)
        assert (1000,) == d1.shape
        assert (100,) == d1.chunks
        assert 'foo' == d1.path
        assert '/foo' == d1.name
        assert g.store is d1.store

        # create as descendant
        d2 = g.create_dataset('/a/b/c/', shape=2000, chunks=200, dtype='i1',
                              compression='zlib', compression_opts=9,
                              fill_value=42, order='F')
        assert isinstance(d2, Array)
        assert (2000,) == d2.shape
        assert (200,) == d2.chunks
        assert np.dtype('i1') == d2.dtype
        assert 'zlib' == d2.compressor.codec_id
        assert 9 == d2.compressor.level
        assert 42 == d2.fill_value
        assert 'F' == d2.order
        assert 'a/b/c' == d2.path
        assert '/a/b/c' == d2.name
        assert g.store is d2.store

        # create with data
        data = np.arange(3000, dtype='u2')
        d3 = g.create_dataset('bar', data=data, chunks=300)
        assert isinstance(d3, Array)
        assert (3000,) == d3.shape
        assert (300,) == d3.chunks
        assert np.dtype('u2') == d3.dtype
        assert_array_equal(data, d3[:])
        assert 'bar' == d3.path
        assert '/bar' == d3.name
        assert g.store is d3.store

        # compression arguments handling follows...

        # compression_opts as dict
        d = g.create_dataset('aaa', shape=1000, dtype='u1',
                             compression='blosc',
                             compression_opts=dict(cname='zstd', clevel=1, shuffle=2))
        assert d.compressor.codec_id == 'blosc'
        assert 'zstd' == d.compressor.cname
        assert 1 == d.compressor.clevel
        assert 2 == d.compressor.shuffle

        # compression_opts as sequence
        d = g.create_dataset('bbb', shape=1000, dtype='u1',
                             compression='blosc',
                             compression_opts=('zstd', 1, 2))
        assert d.compressor.codec_id == 'blosc'
        assert 'zstd' == d.compressor.cname
        assert 1 == d.compressor.clevel
        assert 2 == d.compressor.shuffle

        # None compression_opts
        d = g.create_dataset('ccc', shape=1000, dtype='u1', compression='zlib')
        assert d.compressor.codec_id == 'zlib'
        assert 1 == d.compressor.level

        # None compression
        d = g.create_dataset('ddd', shape=1000, dtype='u1', compression=None)
        assert d.compressor is None

        # compressor as compression
        d = g.create_dataset('eee', shape=1000, dtype='u1', compression=Zlib(1))
        assert d.compressor.codec_id == 'zlib'
        assert 1 == d.compressor.level

    def test_require_dataset(self):
        g = self.create_group()

        # create
        d1 = g.require_dataset('foo', shape=1000, chunks=100, dtype='f4')
        d1[:] = np.arange(1000)
        assert isinstance(d1, Array)
        assert (1000,) == d1.shape
        assert (100,) == d1.chunks
        assert np.dtype('f4') == d1.dtype
        assert 'foo' == d1.path
        assert '/foo' == d1.name
        assert g.store is d1.store
        assert_array_equal(np.arange(1000), d1[:])

        # require
        d2 = g.require_dataset('foo', shape=1000, chunks=100, dtype='f4')
        assert isinstance(d2, Array)
        assert (1000,) == d2.shape
        assert (100,) == d2.chunks
        assert np.dtype('f4') == d2.dtype
        assert 'foo' == d2.path
        assert '/foo' == d2.name
        assert g.store is d2.store
        assert_array_equal(np.arange(1000), d2[:])
        assert d1 == d2

        # bad shape - use TypeError for h5py compatibility
        with pytest.raises(TypeError):
            g.require_dataset('foo', shape=2000, chunks=100, dtype='f4')

        # dtype matching
        # can cast
        d3 = g.require_dataset('foo', shape=1000, chunks=100, dtype='i2')
        assert np.dtype('f4') == d3.dtype
        assert d1 == d3
        with pytest.raises(TypeError):
            # cannot cast
            g.require_dataset('foo', shape=1000, chunks=100, dtype='i4')
        with pytest.raises(TypeError):
            # can cast but not exact match
            g.require_dataset('foo', shape=1000, chunks=100, dtype='i2',
                              exact=True)

    def test_create_errors(self):
        g = self.create_group()

        # array obstructs group, array
        g.create_dataset('foo', shape=100, chunks=10)
        with pytest.raises(ValueError):
            g.create_group('foo/bar')
        with pytest.raises(ValueError):
            g.require_group('foo/bar')
        with pytest.raises(ValueError):
            g.create_dataset('foo/bar', shape=100, chunks=10)
        with pytest.raises(ValueError):
            g.require_dataset('foo/bar', shape=100, chunks=10)

        # array obstructs group, array
        g.create_dataset('a/b', shape=100, chunks=10)
        with pytest.raises(ValueError):
            g.create_group('a/b')
        with pytest.raises(ValueError):
            g.require_group('a/b')
        with pytest.raises(ValueError):
            g.create_dataset('a/b', shape=100, chunks=10)

        # group obstructs array
        g.create_group('c/d')
        with pytest.raises(ValueError):
            g.create_dataset('c', shape=100, chunks=10)
        with pytest.raises(ValueError):
            g.require_dataset('c', shape=100, chunks=10)
        with pytest.raises(ValueError):
            g.create_dataset('c/d', shape=100, chunks=10)
        with pytest.raises(ValueError):
            g.require_dataset('c/d', shape=100, chunks=10)

        # h5py compatibility, accept 'fillvalue'
        d = g.create_dataset('x', shape=100, chunks=10, fillvalue=42)
        assert 42 == d.fill_value

        # h5py compatibility, ignore 'shuffle'
        with pytest.warns(UserWarning, match="ignoring keyword argument 'shuffle'"):
            g.create_dataset('y', shape=100, chunks=10, shuffle=True)

        # read-only
        g = self.create_group(read_only=True)
        with pytest.raises(PermissionError):
            g.create_group('zzz')
        with pytest.raises(PermissionError):
            g.require_group('zzz')
        with pytest.raises(PermissionError):
            g.create_dataset('zzz', shape=100, chunks=10)
        with pytest.raises(PermissionError):
            g.require_dataset('zzz', shape=100, chunks=10)

    def test_create_overwrite(self):
        try:
            for method_name in 'create_dataset', 'create', 'empty', 'zeros', \
                               'ones':
                g = self.create_group()
                getattr(g, method_name)('foo', shape=100, chunks=10)

                # overwrite array with array
                d = getattr(g, method_name)('foo', shape=200, chunks=20,
                                            overwrite=True)
                assert (200,) == d.shape
                # overwrite array with group
                g2 = g.create_group('foo', overwrite=True)
                assert 0 == len(g2)
                # overwrite group with array
                d = getattr(g, method_name)('foo', shape=300, chunks=30,
                                            overwrite=True)
                assert (300,) == d.shape
                # overwrite array with group
                d = getattr(g, method_name)('foo/bar', shape=400, chunks=40,
                                            overwrite=True)
                assert (400,) == d.shape
                assert isinstance(g['foo'], Group)
        except NotImplementedError:
            pass

    def test_getitem_contains_iterators(self):
        # setup
        g1 = self.create_group()
        g2 = g1.create_group('foo/bar')
        d1 = g2.create_dataset('/a/b/c', shape=1000, chunks=100)
        d1[:] = np.arange(1000)
        d2 = g1.create_dataset('foo/baz', shape=3000, chunks=300)
        d2[:] = np.arange(3000)

        # test __getitem__
        assert isinstance(g1['foo'], Group)
        assert isinstance(g1['foo']['bar'], Group)
        assert isinstance(g1['foo/bar'], Group)
        assert isinstance(g1['/foo/bar/'], Group)
        assert isinstance(g1['foo/baz'], Array)
        assert g2 == g1['foo/bar']
        assert g1['foo']['bar'] == g1['foo/bar']
        assert d2 == g1['foo/baz']
        assert_array_equal(d2[:], g1['foo/baz'])
        assert isinstance(g1['a'], Group)
        assert isinstance(g1['a']['b'], Group)
        assert isinstance(g1['a/b'], Group)
        assert isinstance(g1['a']['b']['c'], Array)
        assert isinstance(g1['a/b/c'], Array)
        assert d1 == g1['a/b/c']
        assert g1['a']['b']['c'] == g1['a/b/c']
        assert_array_equal(d1[:], g1['a/b/c'][:])

        # test __contains__
        assert 'foo' in g1
        assert 'foo/bar' in g1
        assert 'foo/baz' in g1
        assert 'bar' in g1['foo']
        assert 'a' in g1
        assert 'a/b' in g1
        assert 'a/b/c' in g1
        assert 'baz' not in g1
        assert 'a/b/c/d' not in g1
        assert 'a/z' not in g1
        assert 'quux' not in g1['foo']

        # test key errors
        with pytest.raises(KeyError):
            g1['baz']
        with pytest.raises(KeyError):
            g1['x/y/z']

        # test __len__
        assert 2 == len(g1)
        assert 2 == len(g1['foo'])
        assert 0 == len(g1['foo/bar'])
        assert 1 == len(g1['a'])
        assert 1 == len(g1['a/b'])

        # test __iter__, keys()
        # currently assumes sorted by key

        assert ['a', 'foo'] == list(g1)
        assert ['a', 'foo'] == list(g1.keys())
        assert ['bar', 'baz'] == list(g1['foo'])
        assert ['bar', 'baz'] == list(g1['foo'].keys())
        assert [] == sorted(g1['foo/bar'])
        assert [] == sorted(g1['foo/bar'].keys())

        # test items(), values()
        # currently assumes sorted by key

        items = list(g1.items())
        values = list(g1.values())
        assert 'a' == items[0][0]
        assert g1['a'] == items[0][1]
        assert g1['a'] == values[0]
        assert 'foo' == items[1][0]
        assert g1['foo'] == items[1][1]
        assert g1['foo'] == values[1]

        items = list(g1['foo'].items())
        values = list(g1['foo'].values())
        assert 'bar' == items[0][0]
        assert g1['foo']['bar'] == items[0][1]
        assert g1['foo']['bar'] == values[0]
        assert 'baz' == items[1][0]
        assert g1['foo']['baz'] == items[1][1]
        assert g1['foo']['baz'] == values[1]

        # test array_keys(), arrays(), group_keys(), groups()
        # currently assumes sorted by key

        assert ['a', 'foo'] == list(g1.group_keys())
        groups = list(g1.groups())
        arrays = list(g1.arrays())
        assert 'a' == groups[0][0]
        assert g1['a'] == groups[0][1]
        assert 'foo' == groups[1][0]
        assert g1['foo'] == groups[1][1]
        assert [] == list(g1.array_keys())
        assert [] == arrays

        assert ['bar'] == list(g1['foo'].group_keys())
        assert ['baz'] == list(g1['foo'].array_keys())
        groups = list(g1['foo'].groups())
        arrays = list(g1['foo'].arrays())
        assert 'bar' == groups[0][0]
        assert g1['foo']['bar'] == groups[0][1]
        assert 'baz' == arrays[0][0]
        assert g1['foo']['baz'] == arrays[0][1]

        # visitor collection tests
        items = []

        def visitor2(obj):
            items.append(obj.path)

        # noinspection PyUnusedLocal
        def visitor3(name, obj=None):
            items.append(name)

        def visitor4(name, obj):
            items.append((name, obj))

        del items[:]
        g1.visitvalues(visitor2)
        assert [
            "a",
            "a/b",
            "a/b/c",
            "foo",
            "foo/bar",
            "foo/baz",
        ] == items

        del items[:]
        g1["foo"].visitvalues(visitor2)
        assert [
            "foo/bar",
            "foo/baz",
        ] == items

        del items[:]
        g1.visit(visitor3)
        assert [
            "a",
            "a/b",
            "a/b/c",
            "foo",
            "foo/bar",
            "foo/baz",
        ] == items

        del items[:]
        g1["foo"].visit(visitor3)
        assert [
            "bar",
            "baz",
        ] == items

        del items[:]
        g1.visitkeys(visitor3)
        assert [
            "a",
            "a/b",
            "a/b/c",
            "foo",
            "foo/bar",
            "foo/baz",
        ] == items

        del items[:]
        g1["foo"].visitkeys(visitor3)
        assert [
            "bar",
            "baz",
        ] == items

        del items[:]
        g1.visititems(visitor3)
        assert [
            "a",
            "a/b",
            "a/b/c",
            "foo",
            "foo/bar",
            "foo/baz",
        ] == items

        del items[:]
        g1["foo"].visititems(visitor3)
        assert [
            "bar",
            "baz",
        ] == items

        del items[:]
        g1.visititems(visitor4)
        for n, o in items:
            assert g1[n] == o

        del items[:]
        g1["foo"].visititems(visitor4)
        for n, o in items:
            assert g1["foo"][n] == o

        # visitor filter tests
        # noinspection PyUnusedLocal
        def visitor0(val, *args):
            name = getattr(val, "path", val)
            if name == "a/b/c/d":
                return True  # pragma: no cover

        # noinspection PyUnusedLocal
        def visitor1(val, *args):
            name = getattr(val, "path", val)
            if name == "a/b/c":
                return True

        assert g1.visit(visitor0) is None
        assert g1.visitkeys(visitor0) is None
        assert g1.visitvalues(visitor0) is None
        assert g1.visititems(visitor0) is None
        assert g1.visit(visitor1) is True
        assert g1.visitkeys(visitor1) is True
        assert g1.visitvalues(visitor1) is True
        assert g1.visititems(visitor1) is True

    def test_empty_getitem_contains_iterators(self):
        # setup
        g = self.create_group()

        # test
        assert [] == list(g)
        assert [] == list(g.keys())
        assert 0 == len(g)
        assert 'foo' not in g

    def test_getattr(self):
        # setup
        g1 = self.create_group()
        g2 = g1.create_group('foo')
        g2.create_dataset('bar', shape=100)

        # test
        assert g1['foo'] == g1.foo
        assert g2['bar'] == g2.bar
        # test that hasattr returns False instead of an exception (issue #88)
        assert not hasattr(g1, 'unexistingattribute')

    def test_setitem(self):
        g = self.create_group()
        try:
            data = np.arange(100)
            g['foo'] = data
            assert_array_equal(data, g['foo'])
            data = np.arange(200)
            g['foo'] = data
            assert_array_equal(data, g['foo'])
            # 0d array
            g['foo'] = 42
            assert () == g['foo'].shape
            assert 42 == g['foo'][()]
        except NotImplementedError:
            pass

    def test_delitem(self):
        g = self.create_group()
        g.create_group('foo')
        g.create_dataset('bar/baz', shape=100, chunks=10)
        assert 'foo' in g
        assert 'bar' in g
        assert 'bar/baz' in g
        try:
            del g['bar']
            with pytest.raises(KeyError):
                del g['xxx']
        except NotImplementedError:
            pass
        else:
            assert 'foo' in g
            assert 'bar' not in g
            assert 'bar/baz' not in g

    def test_move(self):
        g = self.create_group()

        data = np.arange(100)
        g['boo'] = data

        data = np.arange(100)
        g['foo'] = data

        try:
            g.move('foo', 'bar')
            assert 'foo' not in g
            assert 'bar' in g
            assert_array_equal(data, g['bar'])

            g.move('bar', 'foo/bar')
            assert 'bar' not in g
            assert 'foo' in g
            assert 'foo/bar' in g
            assert isinstance(g['foo'], Group)
            assert_array_equal(data, g['foo/bar'])

            g.move('foo', 'foo2')
            assert 'foo' not in g
            assert 'foo/bar' not in g
            assert 'foo2' in g
            assert 'foo2/bar' in g
            assert isinstance(g['foo2'], Group)
            assert_array_equal(data, g['foo2/bar'])

            g2 = g['foo2']
            g2.move('bar', '/bar')
            assert 'foo2' in g
            assert 'foo2/bar' not in g
            assert 'bar' in g
            assert isinstance(g['foo2'], Group)
            assert_array_equal(data, g['bar'])

            with pytest.raises(ValueError):
                g2.move('bar', 'bar2')

            with pytest.raises(ValueError):
                g.move('bar', 'boo')
        except NotImplementedError:
            pass

    def test_array_creation(self):
        grp = self.create_group()

        a = grp.create('a', shape=100, chunks=10)
        assert isinstance(a, Array)
        b = grp.empty('b', shape=100, chunks=10)
        assert isinstance(b, Array)
        assert b.fill_value is None
        c = grp.zeros('c', shape=100, chunks=10)
        assert isinstance(c, Array)
        assert 0 == c.fill_value
        d = grp.ones('d', shape=100, chunks=10)
        assert isinstance(d, Array)
        assert 1 == d.fill_value
        e = grp.full('e', shape=100, chunks=10, fill_value=42)
        assert isinstance(e, Array)
        assert 42 == e.fill_value

        f = grp.empty_like('f', a)
        assert isinstance(f, Array)
        assert f.fill_value is None
        g = grp.zeros_like('g', a)
        assert isinstance(g, Array)
        assert 0 == g.fill_value
        h = grp.ones_like('h', a)
        assert isinstance(h, Array)
        assert 1 == h.fill_value
        i = grp.full_like('i', e)
        assert isinstance(i, Array)
        assert 42 == i.fill_value

        j = grp.array('j', data=np.arange(100), chunks=10)
        assert isinstance(j, Array)
        assert_array_equal(np.arange(100), j[:])

        grp = self.create_group(read_only=True)
        with pytest.raises(PermissionError):
            grp.create('aa', shape=100, chunks=10)
        with pytest.raises(PermissionError):
            grp.empty('aa', shape=100, chunks=10)
        with pytest.raises(PermissionError):
            grp.zeros('aa', shape=100, chunks=10)
        with pytest.raises(PermissionError):
            grp.ones('aa', shape=100, chunks=10)
        with pytest.raises(PermissionError):
            grp.full('aa', shape=100, chunks=10, fill_value=42)
        with pytest.raises(PermissionError):
            grp.array('aa', data=np.arange(100), chunks=10)
        with pytest.raises(PermissionError):
            grp.create('aa', shape=100, chunks=10)
        with pytest.raises(PermissionError):
            grp.empty_like('aa', a)
        with pytest.raises(PermissionError):
            grp.zeros_like('aa', a)
        with pytest.raises(PermissionError):
            grp.ones_like('aa', a)
        with pytest.raises(PermissionError):
            grp.full_like('aa', a)

    def test_paths(self):
        g1 = self.create_group()
        g2 = g1.create_group('foo/bar')

        assert g1 == g1['/']
        assert g1 == g1['//']
        assert g1 == g1['///']
        assert g1 == g2['/']
        assert g1 == g2['//']
        assert g1 == g2['///']
        assert g2 == g1['foo/bar']
        assert g2 == g1['/foo/bar']
        assert g2 == g1['foo/bar/']
        assert g2 == g1['//foo/bar']
        assert g2 == g1['//foo//bar//']
        assert g2 == g1['///foo///bar///']
        assert g2 == g2['/foo/bar']

        with pytest.raises(ValueError):
            g1['.']
        with pytest.raises(ValueError):
            g1['..']
        with pytest.raises(ValueError):
            g1['foo/.']
        with pytest.raises(ValueError):
            g1['foo/..']
        with pytest.raises(ValueError):
            g1['foo/./bar']
        with pytest.raises(ValueError):
            g1['foo/../bar']

    def test_pickle(self):
        # setup
        g = self.create_group()
        d = g.create_dataset('foo/bar', shape=100, chunks=10)
        d[:] = np.arange(100)

        # needed for zip store
        if hasattr(g.store, 'flush'):
            g.store.flush()

        # pickle round trip
        g2 = pickle.loads(pickle.dumps(g))
        assert g.path == g2.path
        assert g.name == g2.name
        assert len(g) == len(g2)
        assert list(g) == list(g2)
        assert g['foo'] == g2['foo']
        assert g['foo/bar'] == g2['foo/bar']


class TestGroupWithDictStore(TestGroup):

    @staticmethod
    def create_store():
        return DictStore(), None


class TestGroupWithDirectoryStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStore(path)
        return store, None


class TestGroupWithNestedDirectoryStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = NestedDirectoryStore(path)
        return store, None


class TestGroupWithZipStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStore(path)
        return store, None


class TestGroupWithDBMStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        store = DBMStore(path, flag='n')
        return store, None


try:
    import bsddb3
except ImportError:  # pragma: no cover
    bsddb3 = None


@unittest.skipIf(bsddb3 is None, 'bsddb3 is not installed')
class TestGroupWithDBMStoreBerkeleyDB(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStore(path, flag='n', open=bsddb3.btopen)
        return store, None


try:
    import lmdb
except ImportError:  # pragma: no cover
    lmdb = None


@unittest.skipIf(lmdb is None, 'lmdb is not installed')
class TestGroupWithLMDBStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        store = LMDBStore(path)
        return store, None


class TestGroupWithChunkStore(TestGroup):

    @staticmethod
    def create_store():
        return dict(), dict()

    def test_chunk_store(self):
        # setup
        store, chunk_store = self.create_store()
        g = self.create_group(store, chunk_store=chunk_store)

        # check attributes
        assert store is g.store
        assert chunk_store is g.chunk_store

        # create array
        a = g.zeros('foo', shape=100, chunks=10)
        assert store is a.store
        assert chunk_store is a.chunk_store
        a[:] = np.arange(100)
        assert_array_equal(np.arange(100), a[:])

        # check store keys
        expect = sorted([group_meta_key, 'foo/' + array_meta_key])
        actual = sorted(store.keys())
        assert expect == actual
        expect = ['foo/' + str(i) for i in range(10)]
        actual = sorted(chunk_store.keys())
        assert expect == actual


class TestGroupWithStoreCache(TestGroup):

    @staticmethod
    def create_store():
        store = LRUStoreCache(dict(), max_size=None)
        return store, None


def test_group():
    # test the group() convenience function

    # basic usage
    g = group()
    assert isinstance(g, Group)
    assert '' == g.path
    assert '/' == g.name

    # usage with custom store
    store = dict()
    g = group(store=store)
    assert isinstance(g, Group)
    assert store is g.store

    # overwrite behaviour
    store = dict()
    init_array(store, shape=100, chunks=10)
    with pytest.raises(ValueError):
        group(store)
    g = group(store, overwrite=True)
    assert isinstance(g, Group)
    assert store is g.store


def test_open_group():
    # test the open_group() convenience function

    store = 'data/group.zarr'

    # mode == 'w'
    g = open_group(store, mode='w')
    assert isinstance(g, Group)
    assert isinstance(g.store, DirectoryStore)
    assert 0 == len(g)
    g.create_groups('foo', 'bar')
    assert 2 == len(g)

    # mode in 'r', 'r+'
    open_array('data/array.zarr', shape=100, chunks=10, mode='w')
    for mode in 'r', 'r+':
        with pytest.raises(ValueError):
            open_group('doesnotexist', mode=mode)
        with pytest.raises(ValueError):
            open_group('data/array.zarr', mode=mode)
    g = open_group(store, mode='r')
    assert isinstance(g, Group)
    assert 2 == len(g)
    with pytest.raises(PermissionError):
        g.create_group('baz')
    g = open_group(store, mode='r+')
    assert isinstance(g, Group)
    assert 2 == len(g)
    g.create_groups('baz', 'quux')
    assert 4 == len(g)

    # mode == 'a'
    shutil.rmtree(store)
    g = open_group(store, mode='a')
    assert isinstance(g, Group)
    assert isinstance(g.store, DirectoryStore)
    assert 0 == len(g)
    g.create_groups('foo', 'bar')
    assert 2 == len(g)
    with pytest.raises(ValueError):
        open_group('data/array.zarr', mode='a')

    # mode in 'w-', 'x'
    for mode in 'w-', 'x':
        shutil.rmtree(store)
        g = open_group(store, mode=mode)
        assert isinstance(g, Group)
        assert isinstance(g.store, DirectoryStore)
        assert 0 == len(g)
        g.create_groups('foo', 'bar')
        assert 2 == len(g)
        with pytest.raises(ValueError):
            open_group(store, mode=mode)
        with pytest.raises(ValueError):
            open_group('data/array.zarr', mode=mode)

    # open with path
    g = open_group(store, path='foo/bar')
    assert isinstance(g, Group)
    assert 'foo/bar' == g.path


def test_group_completions():
    g = group()
    d = dir(g)
    assert 'foo' not in d
    assert 'bar' not in d
    assert 'baz' not in d
    assert 'qux' not in d
    assert 'xxx' not in d
    assert 'yyy' not in d
    assert 'zzz' not in d
    assert '123' not in d
    assert '456' not in d
    g.create_groups('foo', 'bar', 'baz/qux', '123')
    g.zeros('xxx', shape=100)
    g.zeros('yyy', shape=100)
    g.zeros('zzz', shape=100)
    g.zeros('456', shape=100)
    d = dir(g)
    assert 'foo' in d
    assert 'bar' in d
    assert 'baz' in d
    assert 'qux' not in d
    assert 'xxx' in d
    assert 'yyy' in d
    assert 'zzz' in d
    assert '123' not in d  # not valid identifier
    assert '456' not in d  # not valid identifier


def test_group_key_completions():
    g = group()
    d = dir(g)
    # noinspection PyProtectedMember
    k = g._ipython_key_completions_()

    # none of these names should be an attribute
    assert 'foo' not in d
    assert 'bar' not in d
    assert 'baz' not in d
    assert 'qux' not in d
    assert 'xxx' not in d
    assert 'yyy' not in d
    assert 'zzz' not in d
    assert '123' not in d
    assert '456' not in d
    assert 'asdf;' not in d

    # none of these names should be an item
    assert 'foo' not in k
    assert 'bar' not in k
    assert 'baz' not in k
    assert 'qux' not in k
    assert 'xxx' not in k
    assert 'yyy' not in k
    assert 'zzz' not in k
    assert '123' not in k
    assert '456' not in k
    assert 'asdf;' not in k

    g.create_groups('foo', 'bar', 'baz/qux', '123')
    g.zeros('xxx', shape=100)
    g.zeros('yyy', shape=100)
    g.zeros('zzz', shape=100)
    g.zeros('456', shape=100)
    g.zeros('asdf;', shape=100)

    d = dir(g)
    # noinspection PyProtectedMember
    k = g._ipython_key_completions_()

    assert 'foo' in d
    assert 'bar' in d
    assert 'baz' in d
    assert 'qux' not in d
    assert 'xxx' in d
    assert 'yyy' in d
    assert 'zzz' in d
    assert '123' not in d  # not valid identifier
    assert '456' not in d  # not valid identifier
    assert 'asdf;' not in d  # not valid identifier

    assert 'foo' in k
    assert 'bar' in k
    assert 'baz' in k
    assert 'qux' not in k
    assert 'xxx' in k
    assert 'yyy' in k
    assert 'zzz' in k
    assert '123' in k
    assert '456' in k
    assert 'asdf;' in k


def _check_tree(g, expect_bytes, expect_text):
    assert expect_bytes == bytes(g.tree())
    assert expect_text == text_type(g.tree())
    expect_repr = expect_text
    if PY2:  # pragma: py3 no cover
        expect_repr = expect_bytes
    assert expect_repr == repr(g.tree())
    # test _repr_html_ lightly
    # noinspection PyProtectedMember
    html = g.tree()._repr_html_().strip()
    assert html.startswith('<link')
    assert html.endswith('</script>')


def test_tree():
    # setup
    g1 = group()
    g2 = g1.create_group('foo')
    g3 = g1.create_group('bar')
    g3.create_group('baz')
    g5 = g3.create_group('quux')
    g5.create_dataset('baz', shape=100, chunks=10)

    # test root group
    expect_bytes = textwrap.dedent(u"""\
    /
     +-- bar
     |   +-- baz
     |   +-- quux
     |       +-- baz (100,) float64
     +-- foo""").encode()
    expect_text = textwrap.dedent(u"""\
    /
     ├── bar
     │   ├── baz
     │   └── quux
     │       └── baz (100,) float64
     └── foo""")
    _check_tree(g1, expect_bytes, expect_text)

    # test different group
    expect_bytes = textwrap.dedent(u"""\
    foo""").encode()
    expect_text = textwrap.dedent(u"""\
    foo""")
    _check_tree(g2, expect_bytes, expect_text)

    # test different group
    expect_bytes = textwrap.dedent(u"""\
    bar
     +-- baz
     +-- quux
         +-- baz (100,) float64""").encode()
    expect_text = textwrap.dedent(u"""\
    bar
     ├── baz
     └── quux
         └── baz (100,) float64""")
    _check_tree(g3, expect_bytes, expect_text)
