# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import unittest
import tempfile
import atexit
import shutil
import os
import pickle


from nose.tools import assert_raises, eq_ as eq, assert_is, assert_true, \
    assert_is_instance, assert_false, assert_is_none
import numpy as np
from numpy.testing import assert_array_equal


from zarr.storage import DictStore, DirectoryStore, ZipStore, init_group, \
    init_array, attrs_key, array_meta_key, group_meta_key
from zarr.core import Array
from zarr.hierarchy import Group, group, open_group
from zarr.attrs import Attributes
from zarr.errors import PermissionError
from zarr.creation import open_array
from zarr.compat import PY2
from zarr.codecs import Zlib


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
        assert_is(store, g.store)
        assert_false(g.read_only)
        eq('', g.path)
        eq('/', g.name)
        assert_is_instance(g.attrs, Attributes)

    def test_group_init_2(self):
        store, chunk_store = self.create_store()
        g = self.create_group(store, chunk_store=chunk_store,
                              path='/foo/bar/', read_only=True)
        assert_is(store, g.store)
        assert_true(g.read_only)
        eq('foo/bar', g.path)
        eq('/foo/bar', g.name)
        assert_is_instance(g.attrs, Attributes)

    def test_group_init_errors_1(self):
        store, chunk_store = self.create_store()
        # group metadata not initialized
        with assert_raises(KeyError):
            Group(store, chunk_store=chunk_store)

    def test_group_init_errors_2(self):
        store, chunk_store = self.create_store()
        init_array(store, shape=1000, chunks=100, chunk_store=chunk_store)
        # array blocks group
        with assert_raises(KeyError):
            Group(store, chunk_store=chunk_store)

    def test_create_group(self):
        g1 = self.create_group()

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

        # multi
        g6, g7 = g1.create_groups('y', 'z')
        assert_is_instance(g6, Group)
        eq(g6.path, 'y')
        assert_is_instance(g7, Group)
        eq(g7.path, 'z')

    def test_require_group(self):
        g1 = self.create_group()

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

        # multi
        g6, g7 = g1.require_groups('y', 'z')
        assert_is_instance(g6, Group)
        eq(g6.path, 'y')
        assert_is_instance(g7, Group)
        eq(g7.path, 'z')

    def test_create_dataset(self):
        g = self.create_group()

        # create as immediate child
        d1 = g.create_dataset('foo', shape=1000, chunks=100)
        assert_is_instance(d1, Array)
        eq((1000,), d1.shape)
        eq((100,), d1.chunks)
        eq('foo', d1.path)
        eq('/foo', d1.name)
        assert_is(g.store, d1.store)

        # create as descendant
        d2 = g.create_dataset('/a/b/c/', shape=2000, chunks=200, dtype='i1',
                              compression='zlib', compression_opts=9,
                              fill_value=42, order='F')
        assert_is_instance(d2, Array)
        eq((2000,), d2.shape)
        eq((200,), d2.chunks)
        eq(np.dtype('i1'), d2.dtype)
        eq('zlib', d2.compressor.codec_id)
        eq(9, d2.compressor.level)
        eq(42, d2.fill_value)
        eq('F', d2.order)
        eq('a/b/c', d2.path)
        eq('/a/b/c', d2.name)
        assert_is(g.store, d2.store)

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
        assert_is(g.store, d3.store)

        # compression arguments handling follows...

        # compression_opts as dict
        d = g.create_dataset('aaa', shape=1000, dtype='u1',
                             compression='blosc',
                             compression_opts=dict(cname='zstd', clevel=1,
                                                   shuffle=2))
        eq(d.compressor.codec_id, 'blosc')
        eq(b'zstd', d.compressor.cname)
        eq(1, d.compressor.clevel)
        eq(2, d.compressor.shuffle)

        # compression_opts as sequence
        d = g.create_dataset('bbb', shape=1000, dtype='u1',
                             compression='blosc',
                             compression_opts=('zstd', 1, 2))
        eq(d.compressor.codec_id, 'blosc')
        eq(b'zstd', d.compressor.cname)
        eq(1, d.compressor.clevel)
        eq(2, d.compressor.shuffle)

        # None compression_opts
        d = g.create_dataset('ccc', shape=1000, dtype='u1',
                             compression='zlib')
        eq(d.compressor.codec_id, 'zlib')
        eq(1, d.compressor.level)

        # None compression
        d = g.create_dataset('ddd', shape=1000, dtype='u1', compression=None)
        assert_is_none(d.compressor)

        # compressor as compression
        d = g.create_dataset('eee', shape=1000, dtype='u1',
                             compression=Zlib(1))
        eq(d.compressor.codec_id, 'zlib')
        eq(1, d.compressor.level)

    def test_require_dataset(self):
        g = self.create_group()

        # create
        d1 = g.require_dataset('foo', shape=1000, chunks=100, dtype='f4')
        d1[:] = np.arange(1000)
        assert_is_instance(d1, Array)
        eq((1000,), d1.shape)
        eq((100,), d1.chunks)
        eq(np.dtype('f4'), d1.dtype)
        eq('foo', d1.path)
        eq('/foo', d1.name)
        assert_is(g.store, d1.store)
        assert_array_equal(np.arange(1000), d1[:])

        # require
        d2 = g.require_dataset('foo', shape=1000, chunks=100, dtype='f4')
        assert_is_instance(d2, Array)
        eq((1000,), d2.shape)
        eq((100,), d2.chunks)
        eq(np.dtype('f4'), d2.dtype)
        eq('foo', d2.path)
        eq('/foo', d2.name)
        assert_is(g.store, d2.store)
        assert_array_equal(np.arange(1000), d2[:])
        eq(d1, d2)

        # bad shape - use TypeError for h5py compatibility
        with assert_raises(TypeError):
            g.require_dataset('foo', shape=2000, chunks=100, dtype='f4')

        # dtype matching
        # can cast
        d3 = g.require_dataset('foo', shape=1000, chunks=100, dtype='i2')
        eq(np.dtype('f4'), d3.dtype)
        eq(d1, d3)
        with assert_raises(TypeError):
            # cannot cast
            g.require_dataset('foo', shape=1000, chunks=100, dtype='i4')
        with assert_raises(TypeError):
            # can cast but not exact match
            g.require_dataset('foo', shape=1000, chunks=100, dtype='i2',
                              exact=True)

    def test_create_errors(self):
        g = self.create_group()

        # array obstructs group, array
        g.create_dataset('foo', shape=100, chunks=10)
        with assert_raises(KeyError):
            g.create_group('foo/bar')
        with assert_raises(KeyError):
            g.require_group('foo/bar')
        with assert_raises(KeyError):
            g.create_dataset('foo/bar', shape=100, chunks=10)
        with assert_raises(KeyError):
            g.require_dataset('foo/bar', shape=100, chunks=10)

        # array obstructs group, array
        g.create_dataset('a/b', shape=100, chunks=10)
        with assert_raises(KeyError):
            g.create_group('a/b')
        with assert_raises(KeyError):
            g.require_group('a/b')
        with assert_raises(KeyError):
            g.create_dataset('a/b', shape=100, chunks=10)

        # group obstructs array
        g.create_group('c/d')
        with assert_raises(KeyError):
            g.create_dataset('c', shape=100, chunks=10)
        with assert_raises(KeyError):
            g.require_dataset('c', shape=100, chunks=10)
        with assert_raises(KeyError):
            g.create_dataset('c/d', shape=100, chunks=10)
        with assert_raises(KeyError):
            g.require_dataset('c/d', shape=100, chunks=10)

        # h5py compatibility
        d = g.create_dataset('x', shape=100, chunks=10, fillvalue=42)
        eq(42, d.fill_value)
        # ignore 'shuffle'
        d = g.create_dataset('y', shape=100, chunks=10, shuffle=True)
        assert not hasattr(d, 'shuffle')

        # read-only
        g = self.create_group(read_only=True)
        with assert_raises(PermissionError):
            g.create_group('zzz')
        with assert_raises(PermissionError):
            g.require_group('zzz')
        with assert_raises(PermissionError):
            g.create_dataset('zzz', shape=100, chunks=10)
        with assert_raises(PermissionError):
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
                eq((200,), d.shape)
                # overwrite array with group
                g2 = g.create_group('foo', overwrite=True)
                eq(0, len(g2))
                # overwrite group with array
                d = getattr(g, method_name)('foo', shape=300, chunks=30,
                                            overwrite=True)
                eq((300,), d.shape)
                # overwrite array with group
                d = getattr(g, method_name)('foo/bar', shape=400, chunks=40,
                                            overwrite=True)
                assert_is_instance(g['foo'], Group)
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

        # test __iter__, keys()
        # currently assumes sorted by key

        eq(['a', 'foo'], list(g1))
        eq(['a', 'foo'], list(g1.keys()))
        eq(['bar', 'baz'], list(g1['foo']))
        eq(['bar', 'baz'], list(g1['foo'].keys()))
        eq([], sorted(g1['foo/bar']))
        eq([], sorted(g1['foo/bar'].keys()))

        # test items(), values()
        # currently assumes sorted by key

        items = list(g1.items())
        values = list(g1.values())
        eq('a', items[0][0])
        eq(g1['a'], items[0][1])
        eq(g1['a'], values[0])
        eq('foo', items[1][0])
        eq(g1['foo'], items[1][1])
        eq(g1['foo'], values[1])

        items = list(g1['foo'].items())
        values = list(g1['foo'].values())
        eq('bar', items[0][0])
        eq(g1['foo']['bar'], items[0][1])
        eq(g1['foo']['bar'], values[0])
        eq('baz', items[1][0])
        eq(g1['foo']['baz'], items[1][1])
        eq(g1['foo']['baz'], values[1])

        # test array_keys(), arrays(), group_keys(), groups()
        # currently assumes sorted by key

        eq(['a', 'foo'], list(g1.group_keys()))
        groups = list(g1.groups())
        arrays = list(g1.arrays())
        eq('a', groups[0][0])
        eq(g1['a'], groups[0][1])
        eq('foo', groups[1][0])
        eq(g1['foo'], groups[1][1])
        eq([], list(g1.array_keys()))
        eq([], arrays)

        eq(['bar'], list(g1['foo'].group_keys()))
        eq(['baz'], list(g1['foo'].array_keys()))
        groups = list(g1['foo'].groups())
        arrays = list(g1['foo'].arrays())
        eq('bar', groups[0][0])
        eq(g1['foo']['bar'], groups[0][1])
        eq('baz', arrays[0][0])
        eq(g1['foo']['baz'], arrays[0][1])

    def test_empty_getitem_contains_iterators(self):
        # setup
        g = self.create_group()

        # test
        eq([], list(g))
        eq([], list(g.keys()))
        eq(0, len(g))
        assert 'foo' not in g

    def test_getattr(self):
        # setup
        g1 = self.create_group()
        g2 = g1.create_group('foo')
        g2.create_dataset('bar', shape=100)

        # test
        eq(g1['foo'], g1.foo)
        eq(g2['bar'], g2.bar)
        # test that hasattr returns False instead of an exception (issue #88)
        assert_false(hasattr(g1, 'unexistingattribute'))

    def test_group_repr(self):
        g = self.create_group()
        expect = 'Group(/, 0)\n  store: dict'
        actual = repr(g)
        eq(expect, actual)
        g.create_group('foo')
        g.create_group('bar')
        g.create_group('y'*80)
        g.create_dataset('baz', shape=100, chunks=10)
        g.create_dataset('quux', shape=100, chunks=10)
        g.create_dataset('z'*80, shape=100, chunks=10)
        expect = \
            'Group(/, 6)\n' \
            '  arrays: 3; baz, quux, ' \
            'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz...\n' \
            '  groups: 3; bar, foo, ' \
            'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy...\n' \
            '  store: dict'
        actual = repr(g)
        eq(expect, actual)

    def test_setitem(self):
        g = self.create_group()
        try:
            data = np.arange(100)
            g['foo'] = data
            assert_array_equal(data, g['foo'])
            data = np.arange(200)
            g['foo'] = data
            assert_array_equal(data, g['foo'])
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
            with assert_raises(KeyError):
                del g['xxx']
        except NotImplementedError:
            pass
        else:
            assert 'foo' in g
            assert 'bar' not in g
            assert 'bar/baz' not in g

    def test_array_creation(self):
        grp = self.create_group()

        a = grp.create('a', shape=100, chunks=10)
        assert_is_instance(a, Array)
        b = grp.empty('b', shape=100, chunks=10)
        assert_is_instance(b, Array)
        assert_is_none(b.fill_value)
        c = grp.zeros('c', shape=100, chunks=10)
        assert_is_instance(c, Array)
        eq(0, c.fill_value)
        d = grp.ones('d', shape=100, chunks=10)
        assert_is_instance(d, Array)
        eq(1, d.fill_value)
        e = grp.full('e', shape=100, chunks=10, fill_value=42)
        assert_is_instance(e, Array)
        eq(42, e.fill_value)

        f = grp.empty_like('f', a)
        assert_is_instance(f, Array)
        assert_is_none(f.fill_value)
        g = grp.zeros_like('g', a)
        assert_is_instance(g, Array)
        eq(0, g.fill_value)
        h = grp.ones_like('h', a)
        assert_is_instance(h, Array)
        eq(1, h.fill_value)
        i = grp.full_like('i', e)
        assert_is_instance(i, Array)
        eq(42, i.fill_value)

        j = grp.array('j', data=np.arange(100), chunks=10)
        assert_is_instance(j, Array)
        assert_array_equal(np.arange(100), j[:])

        grp = self.create_group(read_only=True)
        with assert_raises(PermissionError):
            grp.create('aa', shape=100, chunks=10)
        with assert_raises(PermissionError):
            grp.empty('aa', shape=100, chunks=10)
        with assert_raises(PermissionError):
            grp.zeros('aa', shape=100, chunks=10)
        with assert_raises(PermissionError):
            grp.ones('aa', shape=100, chunks=10)
        with assert_raises(PermissionError):
            grp.full('aa', shape=100, chunks=10, fill_value=42)
        with assert_raises(PermissionError):
            grp.array('aa', data=np.arange(100), chunks=10)
        with assert_raises(PermissionError):
            grp.create('aa', shape=100, chunks=10)
        with assert_raises(PermissionError):
            grp.empty_like('aa', a)
        with assert_raises(PermissionError):
            grp.zeros_like('aa', a)
        with assert_raises(PermissionError):
            grp.ones_like('aa', a)
        with assert_raises(PermissionError):
            grp.full_like('aa', a)

    def test_paths(self):
        g1 = self.create_group()
        g2 = g1.create_group('foo/bar')

        eq(g1, g1['/'])
        eq(g1, g1['//'])
        eq(g1, g1['///'])
        eq(g1, g2['/'])
        eq(g1, g2['//'])
        eq(g1, g2['///'])
        eq(g2, g1['foo/bar'])
        eq(g2, g1['/foo/bar'])
        eq(g2, g1['foo/bar/'])
        eq(g2, g1['//foo/bar'])
        eq(g2, g1['//foo//bar//'])
        eq(g2, g1['///foo///bar///'])
        eq(g2, g2['/foo/bar'])

        with assert_raises(ValueError):
            g1['.']
        with assert_raises(ValueError):
            g1['..']
        with assert_raises(ValueError):
            g1['foo/.']
        with assert_raises(ValueError):
            g1['foo/..']
        with assert_raises(ValueError):
            g1['foo/./bar']
        with assert_raises(ValueError):
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
        eq(g.path, g2.path)
        eq(g.name, g2.name)
        eq(len(g), len(g2))
        eq(list(g), list(g2))
        eq(g['foo'], g2['foo'])
        eq(g['foo/bar'], g2['foo/bar'])


class TestGroupWithDictStore(TestGroup):

    @staticmethod
    def create_store():
        return DictStore(), None

    def test_group_repr(self):
        g = self.create_group()
        expect = 'Group(/, 0)\n  store: DictStore'
        actual = repr(g)
        for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
            eq(l1, l2)


def rmtree(p, f=shutil.rmtree, g=os.path.isdir):  # pragma: no cover
    """Version of rmtree that will work atexit and only remove if directory."""
    if g(p):
        f(p)


class TestGroupWithDirectoryStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(rmtree, path)
        store = DirectoryStore(path)
        return store, None

    def test_group_repr(self):
        g = self.create_group()
        expect = 'Group(/, 0)\n' \
                 '  store: DirectoryStore'
        actual = repr(g)
        for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
            eq(l1, l2)


class TestGroupWithZipStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStore(path)
        return store, None

    def test_group_repr(self):
        g = self.create_group()
        expect = 'Group(/, 0)\n' \
                 '  store: ZipStore'
        actual = repr(g)
        for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
            eq(l1, l2)


class TestGroupWithChunkStore(TestGroup):

    @staticmethod
    def create_store():
        return dict(), dict()

    def test_group_repr(self):
        if not PY2:
            g = self.create_group()
            expect = 'Group(/, 0)\n' \
                     '  store: dict; chunk_store: dict'
            actual = repr(g)
            for l1, l2 in zip(expect.split('\n'), actual.split('\n')):
                eq(l1, l2)

    def test_chunk_store(self):
        # setup
        store, chunk_store = self.create_store()
        g = self.create_group(store, chunk_store=chunk_store)

        # check attributes
        assert_is(store, g.store)
        assert_is(chunk_store, g.chunk_store)

        # create array
        a = g.zeros('foo', shape=100, chunks=10)
        assert_is(store, a.store)
        assert_is(chunk_store, a.chunk_store)
        a[:] = np.arange(100)
        assert_array_equal(np.arange(100), a[:])

        # check store keys
        expect = sorted([attrs_key, group_meta_key, 'foo/' + attrs_key,
                         'foo/' + array_meta_key])
        actual = sorted(store.keys())
        eq(expect, actual)
        expect = ['foo/' + str(i) for i in range(10)]
        actual = sorted(chunk_store.keys())
        eq(expect, actual)


def test_group():
    # test the group() convenience function

    # basic usage
    g = group()
    assert_is_instance(g, Group)
    eq('', g.path)
    eq('/', g.name)

    # usage with custom store
    store = dict()
    g = group(store=store)
    assert_is_instance(g, Group)
    assert_is(store, g.store)

    # overwrite behaviour
    store = dict()
    init_array(store, shape=100, chunks=10)
    with assert_raises(KeyError):
        group(store)
    g = group(store, overwrite=True)
    assert_is_instance(g, Group)
    assert_is(store, g.store)


def test_open_group():
    # test the open_group() convenience function

    store = 'example'

    # mode == 'w'
    g = open_group(store, mode='w')
    assert_is_instance(g, Group)
    assert_is_instance(g.store, DirectoryStore)
    eq(0, len(g))
    g.create_groups('foo', 'bar')
    eq(2, len(g))

    # mode in 'r', 'r+'
    open_array('example_array', shape=100, chunks=10, mode='w')
    for mode in 'r', 'r+':
        with assert_raises(KeyError):
            open_group('doesnotexist', mode=mode)
        with assert_raises(KeyError):
            open_group('example_array', mode=mode)
    g = open_group(store, mode='r')
    assert_is_instance(g, Group)
    eq(2, len(g))
    with assert_raises(PermissionError):
        g.create_group('baz')
    g = open_group(store, mode='r+')
    assert_is_instance(g, Group)
    eq(2, len(g))
    g.create_groups('baz', 'quux')
    eq(4, len(g))

    # mode == 'a'
    shutil.rmtree(store)
    g = open_group(store, mode='a')
    assert_is_instance(g, Group)
    assert_is_instance(g.store, DirectoryStore)
    eq(0, len(g))
    g.create_groups('foo', 'bar')
    eq(2, len(g))
    with assert_raises(KeyError):
        open_group('example_array', mode='a')

    # mode in 'w-', 'x'
    for mode in 'w-', 'x':
        shutil.rmtree(store)
        g = open_group(store, mode=mode)
        assert_is_instance(g, Group)
        assert_is_instance(g.store, DirectoryStore)
        eq(0, len(g))
        g.create_groups('foo', 'bar')
        eq(2, len(g))
        with assert_raises(KeyError):
            open_group(store, mode=mode)
        with assert_raises(KeyError):
            open_group('example_array', mode=mode)

    # open with path
    g = open_group(store, path='foo/bar')
    assert_is_instance(g, Group)
    eq('foo/bar', g.path)
