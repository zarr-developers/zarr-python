import atexit
import os
import sys
import pickle
import shutil
import tempfile
import textwrap
import unittest

import numpy as np
import pytest

try:
    import ipytree
except ImportError:  # pragma: no cover
    ipytree = None

from numcodecs import Zlib
from numpy.testing import assert_array_equal

from zarr._storage.store import _get_metadata_suffix, v3_api_available
from zarr.attrs import Attributes
from zarr.core import Array
from zarr.creation import open_array
from zarr.hierarchy import Group, group, open_group
from zarr.storage import (ABSStore, DBMStore, KVStore, DirectoryStore, FSStore,
                          LMDBStore, LRUStoreCache, MemoryStore,
                          NestedDirectoryStore, SQLiteStore, ZipStore,
                          array_meta_key, atexit_rmglob, atexit_rmtree, data_root,
                          group_meta_key, init_array, init_group, meta_root)
from zarr._storage.v3 import (ABSStoreV3, KVStoreV3, DirectoryStoreV3, MemoryStoreV3,
                              FSStoreV3, ZipStoreV3, DBMStoreV3, LMDBStoreV3, SQLiteStoreV3,
                              LRUStoreCacheV3)
from zarr.util import InfoReporter, buffer_size
from zarr.tests.util import skip_test_env_var, have_fsspec, abs_container, mktemp


_VERSIONS = ((2, 3) if v3_api_available else (2, ))

# noinspection PyStatementEffect


class TestGroup(unittest.TestCase):

    @staticmethod
    def create_store():
        # can be overridden in sub-classes
        return KVStore(dict()), None

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
        store.close()

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
        store.close()

    def test_group_init_errors_1(self):
        store, chunk_store = self.create_store()
        # group metadata not initialized
        with pytest.raises(ValueError):
            Group(store, chunk_store=chunk_store)
        store.close()

    def test_group_repr(self):
        store, chunk_store = self.create_store()
        g = self.create_group(store, chunk_store=chunk_store)
        assert g.name in repr(g)

    def test_group_init_errors_2(self):
        store, chunk_store = self.create_store()
        init_array(store, shape=1000, chunks=100, chunk_store=chunk_store)
        # array blocks group
        with pytest.raises(ValueError):
            Group(store, chunk_store=chunk_store)
        store.close()

    def _subgroup_path(self, group, path):
        path = path.rstrip('/')
        group_path = '/'.join([group.path, path])
        group_path = group_path.lstrip('/')
        group_name = '/' + group_path
        return group_path, group_name

    def test_create_group(self):
        g1 = self.create_group()

        if g1._version == 2:
            path, name = '', '/'
        else:
            path, name = 'group', '/group'
        # check root group
        assert path == g1.path
        assert name == g1.name

        # create level 1 child group
        g2 = g1.create_group('foo')
        # check with relative path
        path, name = self._subgroup_path(g1, 'foo')
        assert isinstance(g2, Group)
        assert path == g2.path
        assert name == g2.name

        # create level 2 child group
        g3 = g2.create_group('bar')
        path, name = self._subgroup_path(g2, 'bar')
        assert isinstance(g3, Group)
        assert path == g3.path
        assert name == g3.name

        # create level 3 child group
        g4 = g1.create_group('foo/bar/baz')
        path, name = self._subgroup_path(g1, 'foo/bar/baz')
        assert isinstance(g4, Group)
        assert path == g4.path
        assert name == g4.name

        # create level 3 group via root
        g5 = g4.create_group('/a/b/c/')
        assert isinstance(g5, Group)
        assert 'a/b/c' == g5.path
        assert '/a/b/c' == g5.name

        # test non-str keys
        class Foo:

            def __init__(self, s):
                self.s = s

            def __str__(self):
                return self.s

        o = Foo('test/object')
        go = g1.create_group(o)
        path, name = self._subgroup_path(g1, str(o))
        assert isinstance(go, Group)
        assert path == go.path
        go = g1.create_group(b'test/bytes')
        path, name = self._subgroup_path(g1, 'test/bytes')
        assert isinstance(go, Group)
        assert path == go.path

        # test bad keys
        with pytest.raises(ValueError):
            g1.create_group('foo')  # already exists
        if g1._version == 2:
            with pytest.raises(ValueError):
                g1.create_group('a/b/c')  # already exists
        elif g1._version == 3:
            # for v3 'group/a/b/c' does not already exist
            g1.create_group('a/b/c')
        with pytest.raises(ValueError):
            g4.create_group('/a/b/c')  # already exists
        with pytest.raises(ValueError):
            g1.create_group('')

        # multi
        g6, g7 = g1.create_groups('y', 'z')
        assert isinstance(g6, Group)
        assert g6.path == self._subgroup_path(g1, 'y')[0]
        assert isinstance(g7, Group)
        assert g7.path == self._subgroup_path(g1, 'z')[0]

        g1.store.close()

    def test_require_group(self):
        g1 = self.create_group()

        # test creation
        g2 = g1.require_group('foo')
        path, name = self._subgroup_path(g1, 'foo')
        assert isinstance(g2, Group)
        assert path == g2.path
        g3 = g2.require_group('bar')
        path, name = self._subgroup_path(g2, 'bar')
        assert isinstance(g3, Group)
        assert path == g3.path
        g4 = g1.require_group('foo/bar/baz')
        path, name = self._subgroup_path(g1, 'foo/bar/baz')
        assert isinstance(g4, Group)
        assert path == g4.path
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
        if g1._version == 2:
            assert g1.require_group('quux') == g1.require_group('/quux/')
        elif g1._version:
            # These are not equal in v3!
            # 'quux' will be within the group:
            #      meta/root/group/quux.group.json
            # '/quux/' will be outside of the group at:
            #      meta/root/quux.group.json
            assert g1.require_group('quux') != g1.require_group('/quux/')

        # multi
        g6, g7 = g1.require_groups('y', 'z')
        assert isinstance(g6, Group)
        assert g6.path == self._subgroup_path(g1, 'y')[0]
        assert isinstance(g7, Group)
        assert g7.path == self._subgroup_path(g1, 'z')[0]

        g1.store.close()

    def test_rmdir_group_and_array_metadata_files(self):
        """Test group.store's rmdir method.

        This test case was added to complete test coverage of `ABSStore.rmdir`.
        """
        g1 = self.create_group()
        # create a dataset
        g1.create_dataset('arr1', shape=(100,), chunks=(10,), dtype=np.uint8)

        # create level 1 child group
        g2 = g1.create_group('foo')
        g1.create_dataset('arr2', shape=(100,), chunks=(10,), dtype=np.uint8)

        if g1._version > 2 and g1.store.is_erasable():
            arr_path = g1.path + '/arr1'
            sfx = _get_metadata_suffix(g1.store)
            array_meta_file = meta_root + arr_path + '.array' + sfx
            assert array_meta_file in g1.store
            group_meta_file = meta_root + g2.path + '.group' + sfx
            assert group_meta_file in g1.store

            # rmdir on the array path should also remove the metadata file
            g1.store.rmdir(arr_path)
            assert array_meta_file not in g1.store
            # rmdir on the group path should also remove its metadata file
            g1.store.rmdir(g2.path)
            assert group_meta_file not in g1.store

    def _dataset_path(self, group, path):
        path = path.rstrip('/')
        absolute = path.startswith('/')
        if absolute:
            dataset_path = path
        else:
            dataset_path = '/'.join([group.path, path])
        dataset_path = dataset_path.lstrip('/')
        dataset_name = '/' + dataset_path
        return dataset_path, dataset_name

    def test_create_dataset(self):
        g = self.create_group()

        # create as immediate child
        dpath = 'foo'
        d1 = g.create_dataset(dpath, shape=1000, chunks=100)
        path, name = self._dataset_path(g, dpath)
        assert isinstance(d1, Array)
        assert (1000,) == d1.shape
        assert (100,) == d1.chunks
        assert path == d1.path
        assert name == d1.name
        assert g.store is d1.store

        # create as descendant
        dpath = '/a/b/c/'
        d2 = g.create_dataset(dpath, shape=2000, chunks=200, dtype='i1',
                              compression='zlib', compression_opts=9,
                              fill_value=42, order='F')
        path, name = self._dataset_path(g, dpath)
        assert isinstance(d2, Array)
        assert (2000,) == d2.shape
        assert (200,) == d2.chunks
        assert np.dtype('i1') == d2.dtype
        assert 'zlib' == d2.compressor.codec_id
        assert 9 == d2.compressor.level
        assert 42 == d2.fill_value
        assert 'F' == d2.order
        assert path == d2.path
        assert name == d2.name
        assert g.store is d2.store

        # create with data
        data = np.arange(3000, dtype='u2')
        dpath = 'bar'
        d3 = g.create_dataset(dpath, data=data, chunks=300)
        path, name = self._dataset_path(g, dpath)
        assert isinstance(d3, Array)
        assert (3000,) == d3.shape
        assert (300,) == d3.chunks
        assert np.dtype('u2') == d3.dtype
        assert_array_equal(data, d3[:])
        assert path == d3.path
        assert name == d3.name
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

        g.store.close()

    def test_require_dataset(self):
        g = self.create_group()

        # create
        dpath = 'foo'
        d1 = g.require_dataset(dpath, shape=1000, chunks=100, dtype='f4')
        d1[:] = np.arange(1000)
        path, name = self._dataset_path(g, dpath)
        assert isinstance(d1, Array)
        assert (1000,) == d1.shape
        assert (100,) == d1.chunks
        assert np.dtype('f4') == d1.dtype
        assert path == d1.path
        assert name == d1.name
        assert g.store is d1.store
        assert_array_equal(np.arange(1000), d1[:])

        # require
        d2 = g.require_dataset(dpath, shape=1000, chunks=100, dtype='f4')
        assert isinstance(d2, Array)
        assert (1000,) == d2.shape
        assert (100,) == d2.chunks
        assert np.dtype('f4') == d2.dtype
        assert path == d2.path
        assert name == d2.name
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

        g.store.close()

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

        g.store.close()

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

                g.store.close()
        except NotImplementedError:
            pass

    def test_getitem_contains_iterators(self):
        # setup
        g1 = self.create_group()
        g2 = g1.create_group('foo/bar')
        if g1._version == 2:
            d1 = g2.create_dataset('/a/b/c', shape=1000, chunks=100)
        else:
            # v3: cannot create a dataset at the root by starting with /
            #     instead, need to create the dataset on g1 directly
            d1 = g1.create_dataset('a/b/c', shape=1000, chunks=100)
        d1[:] = np.arange(1000)
        d2 = g1.create_dataset('foo/baz', shape=3000, chunks=300)
        d2[:] = np.arange(3000)

        # test __getitem__
        assert isinstance(g1['foo'], Group)
        assert isinstance(g1['foo']['bar'], Group)
        assert isinstance(g1['foo/bar'], Group)
        if g1._version == 2:
            assert isinstance(g1['/foo/bar/'], Group)
        else:
            # start or end with / raises KeyError
            # TODO: should we allow stripping of these on v3?
            with pytest.raises(KeyError):
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

        if g1._version == 2:
            # currently assumes sorted by key
            assert ['a', 'foo'] == list(g1)
            assert ['a', 'foo'] == list(g1.keys())
            assert ['bar', 'baz'] == list(g1['foo'])
            assert ['bar', 'baz'] == list(g1['foo'].keys())
        else:
            # v3 is not necessarily sorted by key
            assert ['a', 'foo'] == sorted(list(g1))
            assert ['a', 'foo'] == sorted(list(g1.keys()))
            assert ['bar', 'baz'] == sorted(list(g1['foo']))
            assert ['bar', 'baz'] == sorted(list(g1['foo'].keys()))
        assert [] == sorted(g1['foo/bar'])
        assert [] == sorted(g1['foo/bar'].keys())

        # test items(), values()
        # currently assumes sorted by key

        items = list(g1.items())
        values = list(g1.values())
        if g1._version == 3:
            # v3 are not automatically sorted by key
            items, values = zip(*sorted(zip(items, values), key=lambda x: x[0]))
        assert 'a' == items[0][0]
        assert g1['a'] == items[0][1]
        assert g1['a'] == values[0]
        assert 'foo' == items[1][0]
        assert g1['foo'] == items[1][1]
        assert g1['foo'] == values[1]

        items = list(g1['foo'].items())
        values = list(g1['foo'].values())
        if g1._version == 3:
            # v3 are not automatically sorted by key
            items, values = zip(*sorted(zip(items, values), key=lambda x: x[0]))
        assert 'bar' == items[0][0]
        assert g1['foo']['bar'] == items[0][1]
        assert g1['foo']['bar'] == values[0]
        assert 'baz' == items[1][0]
        assert g1['foo']['baz'] == items[1][1]
        assert g1['foo']['baz'] == values[1]

        # test array_keys(), arrays(), group_keys(), groups()

        groups = list(g1.groups())
        arrays = list(g1.arrays())
        if g1._version == 2:
            # currently assumes sorted by key
            assert ['a', 'foo'] == list(g1.group_keys())
        else:
            assert ['a', 'foo'] == sorted(list(g1.group_keys()))
            groups = sorted(groups)
            arrays = sorted(arrays)
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
        if g1._version == 3:
            groups = sorted(groups)
            arrays = sorted(arrays)
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
        expected_items = [
            "a",
            "a/b",
            "a/b/c",
            "foo",
            "foo/bar",
            "foo/baz",
        ]
        if g1._version == 3:
            expected_items = [g1.path + '/' + i for i in expected_items]
        assert expected_items == items

        del items[:]
        g1["foo"].visitvalues(visitor2)
        expected_items = [
            "foo/bar",
            "foo/baz",
        ]
        if g1._version == 3:
            expected_items = [g1.path + '/' + i for i in expected_items]
        assert expected_items == items

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
            if name.startswith('group/'):
                # strip the group path for v3
                name = name[6:]
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

        g1.store.close()

    def test_empty_getitem_contains_iterators(self):
        # setup
        g = self.create_group()

        # test
        assert [] == list(g)
        assert [] == list(g.keys())
        assert 0 == len(g)
        assert 'foo' not in g

        g.store.close()

    def test_iterators_recurse(self):
        # setup
        g1 = self.create_group()
        g2 = g1.create_group('foo/bar')
        d1 = g2.create_dataset('/a/b/c', shape=1000, chunks=100)
        d1[:] = np.arange(1000)
        d2 = g1.create_dataset('foo/baz', shape=3000, chunks=300)
        d2[:] = np.arange(3000)
        d3 = g2.create_dataset('zab', shape=2000, chunks=200)
        d3[:] = np.arange(2000)

        # test recursive array_keys
        array_keys = list(g1['foo'].array_keys(recurse=False))
        array_keys_recurse = list(g1['foo'].array_keys(recurse=True))
        assert len(array_keys_recurse) > len(array_keys)
        assert sorted(array_keys_recurse) == ['baz', 'zab']

        # test recursive arrays
        arrays = list(g1['foo'].arrays(recurse=False))
        arrays_recurse = list(g1['foo'].arrays(recurse=True))
        assert len(arrays_recurse) > len(arrays)
        assert 'zab' == arrays_recurse[0][0]
        assert g1['foo']['bar']['zab'] == arrays_recurse[0][1]

        g1.store.close()

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

        g1.store.close()

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
        g.store.close()

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
        g.store.close()

    def test_move(self):
        g = self.create_group()

        data = np.arange(100)
        g['boo'] = data

        data = np.arange(100)
        g['foo'] = data

        g.move("foo", "bar")
        assert "foo" not in g
        assert "bar" in g
        assert_array_equal(data, g["bar"])

        g.move("bar", "foo/bar")
        assert "bar" not in g
        assert "foo" in g
        assert "foo/bar" in g
        assert isinstance(g["foo"], Group)
        assert_array_equal(data, g["foo/bar"])

        g.move("foo", "foo2")
        assert "foo" not in g
        assert "foo/bar" not in g
        assert "foo2" in g
        assert "foo2/bar" in g
        assert isinstance(g["foo2"], Group)
        assert_array_equal(data, g["foo2/bar"])

        g2 = g["foo2"]
        g2.move("bar", "/bar")
        assert "foo2" in g
        assert "foo2/bar" not in g
        if g2._version == 2:
            assert "bar" in g
        else:
            # The `g2.move` call above moved bar to meta/root/bar and
            # meta/data/bar. This is outside the `g` group located at
            # /meta/root/group, so bar is no longer within `g`.
            assert "bar" not in g
            assert 'meta/root/bar.array.json' in g._store
            if g._chunk_store:
                assert 'data/root/bar/c0' in g._chunk_store
            else:
                assert 'data/root/bar/c0' in g._store
        assert isinstance(g["foo2"], Group)
        if g2._version == 2:
            assert_array_equal(data, g["bar"])
        else:
            # TODO: How to access element created outside of group.path in v3?
            #       One option is to make a Hierarchy class representing the
            #       root. Currently Group requires specification of `path`,
            #       but the path of the root would be just '' which is not
            #       currently allowed.
            pass

        with pytest.raises(ValueError):
            g2.move("bar", "bar2")

        with pytest.raises(ValueError):
            g.move("bar", "boo")

        g.store.close()

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

        grp.store.close()

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

        grp.store.close()

    def test_paths(self):
        g1 = self.create_group()
        g2 = g1.create_group('foo/bar')

        if g1._version == 2:
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
        else:
            # the expected key format gives a match
            assert g2 == g1['foo/bar']

            # TODO: Should presence of a trailing slash raise KeyError?
            # The spec says "the final character is not a / character"
            # but we currently strip trailing '/' as done for v2.
            assert g2 == g1['foo/bar/']

            # double slash also currently works (spec doesn't mention this
            # case, but have kept it for v2 behavior compatibility)
            assert g2 == g1['foo//bar']

            # TODO, root: fix these cases
            # v3: leading / implies we are at the root, not within a group,
            # so these all raise KeyError
            for path in ['/foo/bar', '//foo/bar', '//foo//bar//',
                         '///fooo///bar///']:
                with pytest.raises(KeyError):
                    g1[path]

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

        g1.store.close()

    def test_pickle(self):

        # setup group
        g = self.create_group()
        d = g.create_dataset('foo/bar', shape=100, chunks=10)
        d[:] = np.arange(100)
        path = g.path
        name = g.name
        n = len(g)
        keys = list(g)

        # round-trip through pickle
        dump = pickle.dumps(g)
        # some stores cannot be opened twice at the same time, need to close
        # store before can round-trip through pickle
        g.store.close()
        g2 = pickle.loads(dump)

        # verify
        assert path == g2.path
        assert name == g2.name
        assert n == len(g2)
        assert keys == list(g2)
        assert isinstance(g2['foo'], Group)
        assert isinstance(g2['foo/bar'], Array)

        g2.store.close()

    def test_context_manager(self):

        with self.create_group() as g:
            d = g.create_dataset('foo/bar', shape=100, chunks=10)
            d[:] = np.arange(100)


@pytest.mark.parametrize('chunk_dict', [False, True])
def test_group_init_from_dict(chunk_dict):
    if chunk_dict:
        store, chunk_store = dict(), dict()
    else:
        store, chunk_store = dict(), None
    init_group(store, path=None, chunk_store=chunk_store)
    g = Group(store, path=None, read_only=False, chunk_store=chunk_store)
    assert store is not g.store
    assert isinstance(g.store, KVStore)
    if chunk_store is None:
        assert g.store is g.chunk_store
    else:
        assert chunk_store is not g.chunk_store


# noinspection PyStatementEffect
@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3(TestGroup, unittest.TestCase):

    @staticmethod
    def create_store():
        # can be overridden in sub-classes
        return KVStoreV3(dict()), None

    def create_group(self, store=None, path='group', read_only=False,
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
        # different path/name in v3 case
        assert 'group' == g.path
        assert '/group' == g.name
        assert 'group' == g.basename

        assert isinstance(g.attrs, Attributes)
        g.attrs['foo'] = 'bar'
        assert g.attrs['foo'] == 'bar'

        assert isinstance(g.info, InfoReporter)
        assert isinstance(repr(g.info), str)
        assert isinstance(g.info._repr_html_(), str)
        store.close()

    def test_group_init_errors_2(self):
        store, chunk_store = self.create_store()
        path = 'tmp'
        init_array(store, path=path, shape=1000, chunks=100, chunk_store=chunk_store)
        # array blocks group
        with pytest.raises(ValueError):
            Group(store, path=path, chunk_store=chunk_store)
        store.close()


class TestGroupWithMemoryStore(TestGroup):

    @staticmethod
    def create_store():
        return MemoryStore(), None


# noinspection PyStatementEffect
@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithMemoryStore(TestGroupWithMemoryStore, TestGroupV3):

    @staticmethod
    def create_store():
        return MemoryStoreV3(), None


class TestGroupWithDirectoryStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStore(path)
        return store, None


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithDirectoryStore(TestGroupWithDirectoryStore, TestGroupV3):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = DirectoryStoreV3(path)
        return store, None


@skip_test_env_var("ZARR_TEST_ABS")
class TestGroupWithABSStore(TestGroup):

    @staticmethod
    def create_store():
        container_client = abs_container()
        store = ABSStore(client=container_client)
        store.rmdir()
        return store, None

    @pytest.mark.skipif(sys.version_info < (3, 7), reason="attr not serializable in py36")
    def test_pickle(self):
        # internal attribute on ContainerClient isn't serializable for py36 and earlier
        super().test_pickle()


@skip_test_env_var("ZARR_TEST_ABS")
@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithABSStore(TestGroupV3):

    @staticmethod
    def create_store():
        container_client = abs_container()
        store = ABSStoreV3(client=container_client)
        store.rmdir()
        return store, None

    @pytest.mark.skipif(sys.version_info < (3, 7), reason="attr not serializable in py36")
    def test_pickle(self):
        # internal attribute on ContainerClient isn't serializable for py36 and earlier
        super().test_pickle()


class TestGroupWithNestedDirectoryStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = NestedDirectoryStore(path)
        return store, None


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestGroupWithFSStore(TestGroup):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = FSStore(path)
        return store, None

    def test_round_trip_nd(self):
        data = np.arange(1000).reshape(10, 10, 10)
        name = 'raw'

        store, _ = self.create_store()
        f = open_group(store, mode='w')
        f.create_dataset(name, data=data, chunks=(5, 5, 5),
                         compressor=None)
        assert name in f
        h = open_group(store, mode='r')
        np.testing.assert_array_equal(h[name][:], data)


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithFSStore(TestGroupWithFSStore, TestGroupV3):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = FSStoreV3(path)
        return store, None

    def test_round_trip_nd(self):
        data = np.arange(1000).reshape(10, 10, 10)
        name = 'raw'

        store, _ = self.create_store()
        f = open_group(store, path='group', mode='w')
        f.create_dataset(name, data=data, chunks=(5, 5, 5),
                         compressor=None)
        h = open_group(store, path='group', mode='r')
        np.testing.assert_array_equal(h[name][:], data)

        f = open_group(store, path='group2', mode='w')

        data_size = data.nbytes
        group_meta_size = buffer_size(store[meta_root + 'group.group.json'])
        group2_meta_size = buffer_size(store[meta_root + 'group2.group.json'])
        array_meta_size = buffer_size(store[meta_root + 'group/raw.array.json'])
        assert store.getsize() == data_size + group_meta_size + group2_meta_size + array_meta_size
        # added case with path to complete coverage
        assert store.getsize('group') == data_size + group_meta_size + array_meta_size
        assert store.getsize('group2') == group2_meta_size
        assert store.getsize('group/raw') == data_size + array_meta_size


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
class TestGroupWithNestedFSStore(TestGroupWithFSStore):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = FSStore(path, key_separator='/', auto_mkdir=True)
        return store, None

    def test_inconsistent_dimension_separator(self):
        data = np.arange(1000).reshape(10, 10, 10)
        name = 'raw'

        store, _ = self.create_store()
        f = open_group(store, mode='w')

        # cannot specify dimension_separator that conflicts with the store
        with pytest.raises(ValueError):
            f.create_dataset(name, data=data, chunks=(5, 5, 5),
                             compressor=None, dimension_separator='.')


@pytest.mark.skipif(have_fsspec is False, reason="needs fsspec")
@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithNestedFSStore(TestGroupV3WithFSStore):

    @staticmethod
    def create_store():
        path = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, path)
        store = FSStoreV3(path, key_separator='/', auto_mkdir=True)
        return store, None

    def test_inconsistent_dimension_separator(self):
        data = np.arange(1000).reshape(10, 10, 10)
        name = 'raw'

        store, _ = self.create_store()
        f = open_group(store, path='group', mode='w')

        # cannot specify dimension_separator that conflicts with the store
        with pytest.raises(ValueError):
            f.create_dataset(name, data=data, chunks=(5, 5, 5),
                             compressor=None, dimension_separator='.')


class TestGroupWithZipStore(TestGroup):

    @staticmethod
    def create_store():
        path = mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStore(path)
        return store, None

    def test_context_manager(self):

        with self.create_group() as g:
            store = g.store
            d = g.create_dataset('foo/bar', shape=100, chunks=10)
            d[:] = np.arange(100)

        # Check that exiting the context manager closes the store,
        # and therefore the underlying ZipFile.
        with pytest.raises(ValueError):
            store.zf.extractall()

    def test_move(self):
        # zip store is not erasable (can so far only append to a zip
        # so we can't test for move.
        pass


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithZipStore(TestGroupWithZipStore, TestGroupV3):

    @staticmethod
    def create_store():
        path = mktemp(suffix='.zip')
        atexit.register(os.remove, path)
        store = ZipStoreV3(path)
        return store, None


class TestGroupWithDBMStore(TestGroup):

    @staticmethod
    def create_store():
        path = mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        store = DBMStore(path, flag='n')
        return store, None


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithDBMStore(TestGroupWithDBMStore, TestGroupV3):

    @staticmethod
    def create_store():
        path = mktemp(suffix='.anydbm')
        atexit.register(atexit_rmglob, path + '*')
        store = DBMStoreV3(path, flag='n')
        return store, None


class TestGroupWithDBMStoreBerkeleyDB(TestGroup):

    @staticmethod
    def create_store():
        bsddb3 = pytest.importorskip("bsddb3")
        path = mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStore(path, flag='n', open=bsddb3.btopen)
        return store, None


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithDBMStoreBerkeleyDB(TestGroupWithDBMStoreBerkeleyDB, TestGroupV3):

    @staticmethod
    def create_store():
        bsddb3 = pytest.importorskip("bsddb3")
        path = mktemp(suffix='.dbm')
        atexit.register(os.remove, path)
        store = DBMStoreV3(path, flag='n', open=bsddb3.btopen)
        return store, None


class TestGroupWithLMDBStore(TestGroup):

    @staticmethod
    def create_store():
        pytest.importorskip("lmdb")
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        store = LMDBStore(path)
        return store, None


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithLMDBStore(TestGroupWithLMDBStore, TestGroupV3):

    @staticmethod
    def create_store():
        pytest.importorskip("lmdb")
        path = mktemp(suffix='.lmdb')
        atexit.register(atexit_rmtree, path)
        store = LMDBStoreV3(path)
        return store, None


class TestGroupWithSQLiteStore(TestGroup):

    def create_store(self):
        pytest.importorskip("sqlite3")
        path = mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStore(path)
        return store, None


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithSQLiteStore(TestGroupWithSQLiteStore, TestGroupV3):

    def create_store(self):
        pytest.importorskip("sqlite3")
        path = mktemp(suffix='.db')
        atexit.register(atexit_rmtree, path)
        store = SQLiteStoreV3(path)
        return store, None


class TestGroupWithChunkStore(TestGroup):

    @staticmethod
    def create_store():
        return KVStore(dict()), KVStore(dict())

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


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithChunkStore(TestGroupWithChunkStore, TestGroupV3):

    @staticmethod
    def create_store():
        return KVStoreV3(dict()), KVStoreV3(dict())

    def test_chunk_store(self):
        # setup
        store, chunk_store = self.create_store()
        path = 'group1'
        g = self.create_group(store, path=path, chunk_store=chunk_store)

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
        group_key = meta_root + path + '.group.json'
        array_key = meta_root + path + '/foo' + '.array.json'
        expect = sorted([group_key, array_key, 'zarr.json'])
        actual = sorted(store.keys())
        assert expect == actual
        expect = [data_root + path + '/foo/c' + str(i) for i in range(10)]
        expect += ['zarr.json']
        actual = sorted(chunk_store.keys())
        assert expect == actual


class TestGroupWithStoreCache(TestGroup):

    @staticmethod
    def create_store():
        store = LRUStoreCache(dict(), max_size=None)
        return store, None


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestGroupV3WithStoreCache(TestGroupWithStoreCache, TestGroupV3):

    @staticmethod
    def create_store():
        store = LRUStoreCacheV3(dict(), max_size=None)
        return store, None


@pytest.mark.parametrize('zarr_version', _VERSIONS)
def test_group(zarr_version):
    # test the group() convenience function

    # basic usage
    if zarr_version == 2:
        g = group()
        assert '' == g.path
        assert '/' == g.name
    else:
        g = group(path='group1', zarr_version=zarr_version)
        assert 'group1' == g.path
        assert '/group1' == g.name
    assert isinstance(g, Group)

    # usage with custom store
    if zarr_version == 2:
        store = KVStore(dict())
        path = None
    else:
        store = KVStoreV3(dict())
        path = 'foo'
    g = group(store=store, path=path)
    assert isinstance(g, Group)
    assert store is g.store

    # overwrite behaviour
    if zarr_version == 2:
        store = KVStore(dict())
        path = None
    else:
        store = KVStoreV3(dict())
        path = 'foo'
    init_array(store, path=path, shape=100, chunks=10)
    with pytest.raises(ValueError):
        group(store, path=path)
    g = group(store, path=path, overwrite=True)
    assert isinstance(g, Group)
    assert store is g.store


@pytest.mark.parametrize('zarr_version', _VERSIONS)
def test_open_group(zarr_version):
    # test the open_group() convenience function

    store = 'data/group.zarr'

    expected_store_type = DirectoryStore if zarr_version == 2 else DirectoryStoreV3

    # mode == 'w'
    path = None if zarr_version == 2 else 'group1'
    g = open_group(store, path=path, mode='w', zarr_version=zarr_version)
    assert isinstance(g, Group)
    assert isinstance(g.store, expected_store_type)
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
    g = open_group(store, path=path, mode='a', zarr_version=zarr_version)
    assert isinstance(g, Group)
    assert isinstance(g.store, expected_store_type)
    assert 0 == len(g)
    g.create_groups('foo', 'bar')
    assert 2 == len(g)
    if zarr_version == 2:
        with pytest.raises(ValueError):
            open_group('data/array.zarr', mode='a', zarr_version=zarr_version)
    else:
        # TODO, root: should this raise an error?
        open_group('data/array.zarr', mode='a', zarr_version=zarr_version)

    # mode in 'w-', 'x'
    for mode in 'w-', 'x':
        shutil.rmtree(store)
        g = open_group(store, path=path, mode=mode, zarr_version=zarr_version)
        assert isinstance(g, Group)
        assert isinstance(g.store, expected_store_type)
        assert 0 == len(g)
        g.create_groups('foo', 'bar')
        assert 2 == len(g)
        with pytest.raises(ValueError):
            open_group(store, path=path, mode=mode, zarr_version=zarr_version)
        if zarr_version == 2:
            with pytest.raises(ValueError):
                open_group('data/array.zarr', mode=mode)

    # open with path
    g = open_group(store, path='foo/bar', zarr_version=zarr_version)
    assert isinstance(g, Group)
    assert 'foo/bar' == g.path


@pytest.mark.parametrize('zarr_version', _VERSIONS)
def test_group_completions(zarr_version):
    path = None if zarr_version == 2 else 'group1'
    g = group(path=path, zarr_version=zarr_version)
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


@pytest.mark.parametrize('zarr_version', _VERSIONS)
def test_group_key_completions(zarr_version):
    path = None if zarr_version == 2 else 'group1'
    g = group(path=path, zarr_version=zarr_version)
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
    if zarr_version == 2:
        g.zeros('asdf;', shape=100)
    else:
        # cannot have ; in key name for v3
        with pytest.raises(ValueError):
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
    if zarr_version == 2:
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
    if zarr_version == 2:
        assert 'asdf;' in k


def _check_tree(g, expect_bytes, expect_text):
    assert expect_bytes == bytes(g.tree())
    assert expect_text == str(g.tree())
    expect_repr = expect_text
    assert expect_repr == repr(g.tree())
    if ipytree:
        # noinspection PyProtectedMember
        widget = g.tree()._ipython_display_()
        isinstance(widget, ipytree.Tree)


@pytest.mark.parametrize('zarr_version', _VERSIONS)
@pytest.mark.parametrize('at_root', [False, True])
def test_tree(zarr_version, at_root):
    # setup
    path = None if at_root else 'group1'
    g1 = group(path=path, zarr_version=zarr_version)
    g2 = g1.create_group('foo')
    g3 = g1.create_group('bar')
    g3.create_group('baz')
    g5 = g3.create_group('quux')
    g5.create_dataset('baz', shape=100, chunks=10)

    tree_path = '/' if at_root else path
    # test root group
    if zarr_version == 2:
        expect_bytes = textwrap.dedent(f"""\
        {tree_path}
         +-- bar
         |   +-- baz
         |   +-- quux
         |       +-- baz (100,) float64
         +-- foo""").encode()
        expect_text = textwrap.dedent(f"""\
        {tree_path}
         ├── bar
         │   ├── baz
         │   └── quux
         │       └── baz (100,) float64
         └── foo""")
    else:
        # Almost the same as for v2, but has a path name and the
        # subgroups are not necessarily sorted alphabetically.
        expect_bytes = textwrap.dedent(f"""\
        {tree_path}
         +-- foo
         +-- bar
             +-- baz
             +-- quux
                 +-- baz (100,) float64""").encode()
        expect_text = textwrap.dedent(f"""\
        {tree_path}
         ├── foo
         └── bar
             ├── baz
             └── quux
                 └── baz (100,) float64""")
    _check_tree(g1, expect_bytes, expect_text)

    # test different group
    expect_bytes = textwrap.dedent("""\
    foo""").encode()
    expect_text = textwrap.dedent("""\
    foo""")
    _check_tree(g2, expect_bytes, expect_text)

    # test different group
    expect_bytes = textwrap.dedent("""\
    bar
     +-- baz
     +-- quux
         +-- baz (100,) float64""").encode()
    expect_text = textwrap.dedent("""\
    bar
     ├── baz
     └── quux
         └── baz (100,) float64""")
    _check_tree(g3, expect_bytes, expect_text)


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
def test_group_mismatched_store_versions():
    store_v3 = KVStoreV3(dict())
    store_v2 = KVStore(dict())

    # separate chunk store
    chunk_store_v2 = KVStore(dict())
    chunk_store_v3 = KVStoreV3(dict())

    init_group(store_v2, path='group1', chunk_store=chunk_store_v2)
    init_group(store_v3, path='group1', chunk_store=chunk_store_v3)

    g1_v3 = Group(store_v3, path='group1', read_only=True, chunk_store=chunk_store_v3)
    assert isinstance(g1_v3._store, KVStoreV3)
    g1_v2 = Group(store_v2, path='group1', read_only=True, chunk_store=chunk_store_v2)
    assert isinstance(g1_v2._store, KVStore)

    # store and chunk_store must have the same zarr protocol version
    with pytest.raises(ValueError):
        Group(store_v3, path='group1', read_only=False, chunk_store=chunk_store_v2)
    with pytest.raises(ValueError):
        Group(store_v2, path='group1', read_only=False, chunk_store=chunk_store_v3)
    with pytest.raises(ValueError):
        open_group(store_v2, path='group1', chunk_store=chunk_store_v3)
    with pytest.raises(ValueError):
        open_group(store_v3, path='group1', chunk_store=chunk_store_v2)

    # raises Value if read_only and path is not a pre-existing group
    with pytest.raises(ValueError):
        Group(store_v3, path='group2', read_only=True, chunk_store=chunk_store_v3)
    with pytest.raises(ValueError):
        Group(store_v3, path='group2', read_only=True, chunk_store=chunk_store_v3)


@pytest.mark.parametrize('zarr_version', _VERSIONS)
def test_open_group_from_paths(zarr_version):
    """Verify zarr_version is applied to both the store and chunk_store."""
    store = tempfile.mkdtemp()
    chunk_store = tempfile.mkdtemp()
    atexit.register(atexit_rmtree, store)
    atexit.register(atexit_rmtree, chunk_store)
    path = 'g1'
    g = open_group(store, path=path, chunk_store=chunk_store, zarr_version=zarr_version)
    assert g._store._store_version == g._chunk_store._store_version == zarr_version
