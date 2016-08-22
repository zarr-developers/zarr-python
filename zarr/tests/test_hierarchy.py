# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import tempfile
import atexit
import shutil
import os


from nose.tools import assert_raises, eq_ as eq, assert_is, assert_true, \
    assert_is_instance, assert_is_none, assert_false
import numpy as np
from numpy.testing import assert_array_equal


from zarr.storage import DictStore, DirectoryStore, ZipStore, init_group, \
    init_array
from zarr.core import Array
from zarr.hierarchy import Group
from zarr.attrs import Attributes


# noinspection PyStatementEffect
class TestGroup(unittest.TestCase):

    @staticmethod
    def create_store():
        """
        :rtype: MutableMapping
        """
        # override in sub-classes
        return dict()

    def test_group_init_1(self):
        store = self.create_store()
        init_group(store)
        g = Group(store)
        assert_is(store, g.store)
        assert_false(g.readonly)
        eq('', g.path)
        eq('/', g.name)
        assert_is_instance(g.attrs, Attributes)

    def test_group_init_2(self):
        store = self.create_store()
        init_group(store, path='/foo/bar/')
        g = Group(store, path='/foo/bar/', readonly=True)
        assert_is(store, g.store)
        assert_true(g.readonly)
        eq('foo/bar', g.path)
        eq('/foo/bar', g.name)
        assert_is_instance(g.attrs, Attributes)

    def test_group_init_errors_1(self):
        store = self.create_store()
        with assert_raises(ValueError):
            Group(store)

    def test_group_init_errors_2(self):
        store = self.create_store()
        init_array(store, shape=1000, chunks=100)
        with assert_raises(ValueError):
            Group(store)

    def test_create_group(self):
        store = self.create_store()
        init_group(store)
        g1 = Group(store=store)

        # check root group
        eq('', g1.path)
        eq('/', g1.name)

        # create level 1 child group
        g2 = g1.create_group('foo')
        assert_is_instance(g2, Group)
        eq('foo', g2.path)
        eq('/foo', g2.name)

        # create level 2 child group
        g3 = g2.create_group('bar')
        assert_is_instance(g3, Group)
        eq('foo/bar', g3.path)
        eq('/foo/bar', g3.name)

        # create level 3 child group
        g4 = g1.create_group('foo/bar/baz')
        assert_is_instance(g4, Group)
        eq('foo/bar/baz', g4.path)
        eq('/foo/bar/baz', g4.name)

        # create level 3 group via root
        g5 = g4.create_group('/a/b/c/')
        assert_is_instance(g5, Group)
        eq('a/b/c', g5.path)
        eq('/a/b/c', g5.name)

        # test bad keys
        with assert_raises(KeyError):
            g1.create_group('foo')  # already exists
        with assert_raises(KeyError):
            g1.create_group('a/b/c')  # already exists
        with assert_raises(KeyError):
            g4.create_group('/a/b/c')  # already exists
        with assert_raises(KeyError):
            g1.create_group('')
        with assert_raises(KeyError):
            g1.create_group('/')
        with assert_raises(KeyError):
            g1.create_group('//')

    def test_require_group(self):
        store = self.create_store()
        init_group(store)
        g1 = Group(store=store)

        # test creation
        g2 = g1.require_group('foo')
        assert_is_instance(g2, Group)
        eq('foo', g2.path)
        g3 = g2.require_group('bar')
        assert_is_instance(g3, Group)
        eq('foo/bar', g3.path)
        g4 = g1.require_group('foo/bar/baz')
        assert_is_instance(g4, Group)
        eq('foo/bar/baz', g4.path)
        g5 = g4.require_group('/a/b/c/')
        assert_is_instance(g5, Group)
        eq('a/b/c', g5.path)

        # test when already created
        g2a = g1.require_group('foo')
        eq(g2, g2a)
        assert_is(g2.store, g2a.store)
        g3a = g2a.require_group('bar')
        eq(g3, g3a)
        assert_is(g3.store, g3a.store)
        g4a = g1.require_group('foo/bar/baz')
        eq(g4, g4a)
        assert_is(g4.store, g4a.store)
        g5a = g4a.require_group('/a/b/c/')
        eq(g5, g5a)
        assert_is(g5.store, g5a.store)

        # test path normalization
        eq(g1.require_group('quux'), g1.require_group('/quux/'))

    def test_create_dataset(self):
        store = self.create_store()
        init_group(store)
        g = Group(store=store)

        # create as immediate child
        d1 = g.create_dataset('foo', shape=1000, chunks=100)
        assert_is_instance(d1, Array)
        eq((1000,), d1.shape)
        eq((100,), d1.chunks)
        eq('foo', d1.path)
        eq('/foo', d1.name)
        assert_is(store, d1.store)

        # create as descendant
        d2 = g.create_dataset('/a/b/c/', shape=2000, chunks=200, dtype='i1',
                              compression='zlib', compression_opts=9,
                              fill_value=42, order='F')
        assert_is_instance(d2, Array)
        eq((2000,), d2.shape)
        eq((200,), d2.chunks)
        eq(np.dtype('i1'), d2.dtype)
        eq('zlib', d2.compression)
        eq(9, d2.compression_opts)
        eq(42, d2.fill_value)
        eq('F', d2.order)
        eq('a/b/c', d2.path)
        eq('/a/b/c', d2.name)
        assert_is(store, d2.store)

        # create with data
        data = np.arange(3000, dtype='u2')
        d3 = g.create_dataset('bar', data=data, chunks=300)
        assert_is_instance(d3, Array)
        eq((3000,), d3.shape)
        eq((300,), d3.chunks)
        eq(np.dtype('u2'), d3.dtype)
        assert_array_equal(data, d3[:])
        eq('bar', d3.path)
        eq('/bar', d3.name)
        assert_is(store, d3.store)

    def test_getitem_contains_iterators(self):
        # setup
        store = self.create_store()
        init_group(store)
        g1 = Group(store=store)
        g2 = g1.create_group('foo/bar')
        d1 = g2.create_dataset('/a/b/c', shape=1000, chunks=100)
        d1[:] = np.arange(1000)
        d2 = g1.create_dataset('foo/baz', shape=3000, chunks=300)
        d2[:] = np.arange(3000)

        # test __getitem__
        assert_is_instance(g1['foo'], Group)
        assert_is_instance(g1['foo']['bar'], Group)
        assert_is_instance(g1['foo/bar'], Group)
        assert_is_instance(g1['/foo/bar/'], Group)
        assert_is_instance(g1['foo/baz'], Array)
        eq(g2, g1['foo/bar'])
        eq(g1['foo']['bar'], g1['foo/bar'])
        eq(d2, g1['foo/baz'])
        assert_array_equal(d2[:], g1['foo/baz'])
        assert_is_instance(g1['a'], Group)
        assert_is_instance(g1['a']['b'], Group)
        assert_is_instance(g1['a/b'], Group)
        assert_is_instance(g1['a']['b']['c'], Array)
        assert_is_instance(g1['a/b/c'], Array)
        eq(d1, g1['a/b/c'])
        eq(g1['a']['b']['c'], g1['a/b/c'])
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
        with assert_raises(KeyError):
            g1['baz']
        with assert_raises(KeyError):
            g1['x/y/z']

        # test __len__
        eq(2, len(g1))
        eq(2, len(g1['foo']))
        eq(0, len(g1['foo/bar']))
        eq(1, len(g1['a']))
        eq(1, len(g1['a/b']))

        # test keys()
        eq(['a', 'foo'], sorted(g1.keys()))
        eq(['bar', 'baz'], sorted(g1['foo'].keys()))
        eq([], sorted(g1['foo/bar'].keys()))

    def test_empty_getitem_contains_iterators(self):
        # setup
        store = self.create_store()
        init_group(store)
        g = Group(store=store)

        # test
        eq([], list(g.keys()))
        eq(0, len(g))
        assert 'foo' not in g

    def test_group_repr(self):
        store = self.create_store()
        init_group(store)
        g = Group(store=store)
        expect = 'zarr.hierarchy.Group(/, 0)\n  store: builtins.dict'
        actual = repr(g)
        eq(expect, actual)


class TestGroupDictStore(TestGroup):

    @staticmethod
    def create_store():
        return DictStore()

    def test_group_repr(self):
        store = self.create_store()
        init_group(store)
        g = Group(store=store)
        expect = 'zarr.hierarchy.Group(/, 0)\n  store: zarr.storage.DictStore'
        actual = repr(g)
        eq(expect, actual)


def rmtree_if_exists(path, rmtree=shutil.rmtree, isdir=os.path.isdir):
    if isdir(path):
        rmtree(path)


class TestGroupDirectoryStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(rmtree_if_exists, path)
        store = DirectoryStore(path)
        return store

    def test_group_repr(self):
        store = self.create_store()
        init_group(store)
        g = Group(store=store)
        expect = 'zarr.hierarchy.Group(/, 0)\n' \
                 '  store: zarr.storage.DirectoryStore'
        actual = repr(g)
        eq(expect, actual)


class TestGroupZipStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStore(path)
        return store

    def test_group_repr(self):
        store = self.create_store()
        init_group(store)
        g = Group(store=store)
        expect = 'zarr.hierarchy.Group(/, 0)\n' \
                 '  store: zarr.storage.ZipStore'
        actual = repr(g)
        eq(expect, actual)
