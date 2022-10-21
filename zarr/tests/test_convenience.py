import atexit
import tempfile
import unittest
from numbers import Integral

import numpy as np
import pytest
from numcodecs import Adler32, Zlib
from numpy.testing import assert_array_equal

import zarr
from zarr.convenience import (
    consolidate_metadata,
    copy,
    copy_store,
    load,
    open,
    open_consolidated,
    save,
    save_group,
    save_array,
    copy_all,
)
from zarr.core import Array
from zarr.errors import CopyError
from zarr.hierarchy import Group, group
from zarr.storage import (
    ConsolidatedMetadataStore,
    FSStore,
    KVStore,
    MemoryStore,
    atexit_rmtree,
    data_root,
    meta_root,
    getsize,
)
from zarr._storage.store import v3_api_available
from zarr._storage.v3 import (
    ConsolidatedMetadataStoreV3,
    DirectoryStoreV3,
    FSStoreV3,
    KVStoreV3,
    MemoryStoreV3,
    SQLiteStoreV3,
)
from zarr.tests.util import have_fsspec

_VERSIONS = ((2, 3) if v3_api_available else (2, ))


def _init_creation_kwargs(zarr_version):
    kwargs = {'zarr_version': zarr_version}
    if zarr_version == 3:
        kwargs['path'] = 'dataset'
    return kwargs


@pytest.mark.parametrize('zarr_version', _VERSIONS)
def test_open_array(path_type, zarr_version):

    store = tempfile.mkdtemp()
    atexit.register(atexit_rmtree, store)
    store = path_type(store)
    kwargs = _init_creation_kwargs(zarr_version)

    # open array, create if doesn't exist
    z = open(store, mode='a', shape=100, **kwargs)
    assert isinstance(z, Array)
    assert z.shape == (100,)

    # open array, overwrite
    z = open(store, mode='w', shape=200, **kwargs)
    assert isinstance(z, Array)
    assert z.shape == (200,)

    # open array, read-only
    z = open(store, mode='r', **kwargs)
    assert isinstance(z, Array)
    assert z.shape == (200,)
    assert z.read_only

    # path not found
    with pytest.raises(ValueError):
        open('doesnotexist', mode='r')


@pytest.mark.parametrize("zarr_version", _VERSIONS)
def test_open_group(path_type, zarr_version):

    store = tempfile.mkdtemp()
    atexit.register(atexit_rmtree, store)
    store = path_type(store)
    kwargs = _init_creation_kwargs(zarr_version)

    # open group, create if doesn't exist
    g = open(store, mode='a', **kwargs)
    g.create_group('foo')
    assert isinstance(g, Group)
    assert 'foo' in g

    # open group, overwrite
    g = open(store, mode='w', **kwargs)
    assert isinstance(g, Group)
    assert 'foo' not in g

    # open group, read-only
    g = open(store, mode='r', **kwargs)
    assert isinstance(g, Group)
    assert g.read_only


@pytest.mark.parametrize("zarr_version", _VERSIONS)
def test_save_errors(zarr_version):
    with pytest.raises(ValueError):
        # no arrays provided
        save_group('data/group.zarr', zarr_version=zarr_version)
    with pytest.raises(TypeError):
        # no array provided
        save_array('data/group.zarr', zarr_version=zarr_version)
    with pytest.raises(ValueError):
        # no arrays provided
        save('data/group.zarr', zarr_version=zarr_version)


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
def test_zarr_v3_save_multiple_unnamed():
    x = np.ones(8)
    y = np.zeros(8)
    store = KVStoreV3(dict())
    # no path provided
    save_group(store, x, y, path='dataset', zarr_version=3)
    # names become arr_{i} for unnamed *args
    assert data_root + 'dataset/arr_0/c0' in store
    assert data_root + 'dataset/arr_1/c0' in store
    assert meta_root + 'dataset/arr_0.array.json' in store
    assert meta_root + 'dataset/arr_1.array.json' in store


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
def test_zarr_v3_save_errors():
    x = np.ones(8)
    with pytest.raises(ValueError):
        # no path provided
        save_group('data/group.zr3', x, zarr_version=3)
    with pytest.raises(ValueError):
        # no path provided
        save_array('data/group.zr3', x, zarr_version=3)
    with pytest.raises(ValueError):
        # no path provided
        save('data/group.zr3', x, zarr_version=3)


@pytest.mark.parametrize("zarr_version", _VERSIONS)
def test_lazy_loader(zarr_version):
    foo = np.arange(100)
    bar = np.arange(100, 0, -1)
    store = 'data/group.zarr' if zarr_version == 2 else 'data/group.zr3'
    kwargs = _init_creation_kwargs(zarr_version)
    save(store, foo=foo, bar=bar, **kwargs)
    loader = load(store, **kwargs)
    assert 'foo' in loader
    assert 'bar' in loader
    assert 'baz' not in loader
    assert len(loader) == 2
    assert sorted(loader) == ['bar', 'foo']
    assert_array_equal(foo, loader['foo'])
    assert_array_equal(bar, loader['bar'])
    assert 'LazyLoader: ' in repr(loader)


@pytest.mark.parametrize("zarr_version", _VERSIONS)
def test_load_array(zarr_version):
    foo = np.arange(100)
    bar = np.arange(100, 0, -1)
    store = 'data/group.zarr' if zarr_version == 2 else 'data/group.zr3'
    kwargs = _init_creation_kwargs(zarr_version)
    save(store, foo=foo, bar=bar, **kwargs)

    # can also load arrays directly into a numpy array
    for array_name in ['foo', 'bar']:
        array_path = 'dataset/' + array_name if zarr_version == 3 else array_name
        array = load(store, path=array_path, zarr_version=zarr_version)
        assert isinstance(array, np.ndarray)
        if array_name == 'foo':
            assert_array_equal(foo, array)
        else:
            assert_array_equal(bar, array)


@pytest.mark.parametrize("zarr_version", _VERSIONS)
def test_tree(zarr_version):
    kwargs = _init_creation_kwargs(zarr_version)
    g1 = zarr.group(**kwargs)
    g1.create_group('foo')
    g3 = g1.create_group('bar')
    g3.create_group('baz')
    g5 = g3.create_group('qux')
    g5.create_dataset('baz', shape=100, chunks=10)
    assert repr(zarr.tree(g1)) == repr(g1.tree())
    assert str(zarr.tree(g1)) == str(g1.tree())


@pytest.mark.parametrize('zarr_version', _VERSIONS)
@pytest.mark.parametrize('stores_from_path', [False, True])
@pytest.mark.parametrize(
    'with_chunk_store,listable',
    [(False, True), (True, True), (False, False)],
    ids=['default-listable', 'with_chunk_store-listable', 'default-unlistable']
)
def test_consolidate_metadata(with_chunk_store,
                              zarr_version,
                              listable,
                              monkeypatch,
                              stores_from_path):

    # setup initial data
    if stores_from_path:
        store = tempfile.mkdtemp()
        atexit.register(atexit_rmtree, store)
        if with_chunk_store:
            chunk_store = tempfile.mkdtemp()
            atexit.register(atexit_rmtree, chunk_store)
        else:
            chunk_store = None
        version_kwarg = {'zarr_version': zarr_version}
    else:
        if zarr_version == 2:
            store = MemoryStore()
            chunk_store = MemoryStore() if with_chunk_store else None
        elif zarr_version == 3:
            store = MemoryStoreV3()
            chunk_store = MemoryStoreV3() if with_chunk_store else None
        version_kwarg = {}
    path = 'dataset' if zarr_version == 3 else None
    z = group(store, chunk_store=chunk_store, path=path, **version_kwarg)

    # Reload the actual store implementation in case str
    store_to_copy = z.store

    z.create_group('g1')
    g2 = z.create_group('g2')
    g2.attrs['hello'] = 'world'
    arr = g2.create_dataset('arr', shape=(20, 20), chunks=(5, 5), dtype='f8')
    assert 16 == arr.nchunks
    assert 0 == arr.nchunks_initialized
    arr.attrs['data'] = 1
    arr[:] = 1.0
    assert 16 == arr.nchunks_initialized

    if stores_from_path:
        # get the actual store class for use with consolidate_metadata
        store_class = z._store
    else:
        store_class = store

    if zarr_version == 3:
        # error on v3 if path not provided
        with pytest.raises(ValueError):
            consolidate_metadata(store_class, path=None)

        with pytest.raises(ValueError):
            consolidate_metadata(store_class, path='')

    # perform consolidation
    out = consolidate_metadata(store_class, path=path)
    assert isinstance(out, Group)
    assert ['g1', 'g2'] == list(out)
    if not stores_from_path:
        if zarr_version == 2:
            assert isinstance(out._store, ConsolidatedMetadataStore)
            assert '.zmetadata' in store
            meta_keys = ['.zgroup',
                         'g1/.zgroup',
                         'g2/.zgroup',
                         'g2/.zattrs',
                         'g2/arr/.zarray',
                         'g2/arr/.zattrs']
        else:
            assert isinstance(out._store, ConsolidatedMetadataStoreV3)
            assert 'meta/root/consolidated/.zmetadata' in store
            meta_keys = ['zarr.json',
                         meta_root + 'dataset.group.json',
                         meta_root + 'dataset/g1.group.json',
                         meta_root + 'dataset/g2.group.json',
                         meta_root + 'dataset/g2/arr.array.json',
                         'meta/root/consolidated.group.json']
        for key in meta_keys:
            del store[key]

    # https://github.com/zarr-developers/zarr-python/issues/993
    # Make sure we can still open consolidated on an unlistable store:
    if not listable:
        fs_memory = pytest.importorskip("fsspec.implementations.memory")
        monkeypatch.setattr(fs_memory.MemoryFileSystem, "isdir", lambda x, y: False)
        monkeypatch.delattr(fs_memory.MemoryFileSystem, "ls")
        fs = fs_memory.MemoryFileSystem()
        if zarr_version == 2:
            store_to_open = FSStore("", fs=fs)
        else:
            store_to_open = FSStoreV3("", fs=fs)

        # copy original store to new unlistable store
        store_to_open.update(store_to_copy)

    else:
        store_to_open = store

    # open consolidated
    z2 = open_consolidated(store_to_open, chunk_store=chunk_store, path=path, **version_kwarg)
    assert ['g1', 'g2'] == list(z2)
    assert 'world' == z2.g2.attrs['hello']
    assert 1 == z2.g2.arr.attrs['data']
    assert (z2.g2.arr[:] == 1.0).all()
    assert 16 == z2.g2.arr.nchunks
    if listable:
        assert 16 == z2.g2.arr.nchunks_initialized
    else:
        with pytest.raises(NotImplementedError):
            _ = z2.g2.arr.nchunks_initialized

    if stores_from_path:
        # path string is note a BaseStore subclass so cannot be used to
        # initialize a ConsolidatedMetadataStore.
        if zarr_version == 2:
            with pytest.raises(ValueError):
                cmd = ConsolidatedMetadataStore(store)
        elif zarr_version == 3:
            with pytest.raises(ValueError):
                cmd = ConsolidatedMetadataStoreV3(store)
    else:
        # tests del/write on the store
        if zarr_version == 2:
            cmd = ConsolidatedMetadataStore(store)
            with pytest.raises(PermissionError):
                del cmd['.zgroup']
            with pytest.raises(PermissionError):
                cmd['.zgroup'] = None
        else:
            cmd = ConsolidatedMetadataStoreV3(store)
            with pytest.raises(PermissionError):
                del cmd[meta_root + 'dataset.group.json']
            with pytest.raises(PermissionError):
                cmd[meta_root + 'dataset.group.json'] = None

        # test getsize on the store
        assert isinstance(getsize(cmd), Integral)

    # test new metadata are not writeable
    with pytest.raises(PermissionError):
        z2.create_group('g3')
    with pytest.raises(PermissionError):
        z2.create_dataset('spam', shape=42, chunks=7, dtype='i4')
    with pytest.raises(PermissionError):
        del z2['g2']

    # test consolidated metadata are not writeable
    with pytest.raises(PermissionError):
        z2.g2.attrs['hello'] = 'universe'
    with pytest.raises(PermissionError):
        z2.g2.arr.attrs['foo'] = 'bar'

    # test the data are writeable
    z2.g2.arr[:] = 2
    assert (z2.g2.arr[:] == 2).all()

    # test invalid modes
    with pytest.raises(ValueError):
        open_consolidated(store, chunk_store=chunk_store, mode='a', path=path)
    with pytest.raises(ValueError):
        open_consolidated(store, chunk_store=chunk_store, mode='w', path=path)
    with pytest.raises(ValueError):
        open_consolidated(store, chunk_store=chunk_store, mode='w-', path=path)

    # make sure keyword arguments are passed through without error
    open_consolidated(
        store, chunk_store=chunk_store, path=path, cache_attrs=True, synchronizer=None,
        **version_kwarg,
    )


@pytest.mark.parametrize("options", (
    {"dimension_separator": "/"},
    {"dimension_separator": "."},
    {"dimension_separator": None},
))
def test_save_array_separator(tmpdir, options):
    data = np.arange(6).reshape((3, 2))
    url = tmpdir.join("test.zarr")
    save_array(url, data, **options)


class TestCopyStore(unittest.TestCase):

    _version = 2

    def setUp(self):
        source = dict()
        source['foo'] = b'xxx'
        source['bar/baz'] = b'yyy'
        source['bar/qux'] = b'zzz'
        self.source = source

    def _get_dest_store(self):
        return dict()

    def test_no_paths(self):
        source = self.source
        dest = self._get_dest_store()
        copy_store(source, dest)
        assert len(source) == len(dest)
        for key in source:
            assert source[key] == dest[key]

    def test_source_path(self):
        source = self.source
        # paths should be normalized
        for source_path in 'bar', 'bar/', '/bar', '/bar/':
            dest = self._get_dest_store()
            copy_store(source, dest, source_path=source_path)
            assert 2 == len(dest)
            for key in source:
                if key.startswith('bar/'):
                    dest_key = key.split('bar/')[1]
                    assert source[key] == dest[dest_key]
                else:
                    assert key not in dest

    def test_dest_path(self):
        source = self.source
        # paths should be normalized
        for dest_path in 'new', 'new/', '/new', '/new/':
            dest = self._get_dest_store()
            copy_store(source, dest, dest_path=dest_path)
            assert len(source) == len(dest)
            for key in source:
                if self._version == 3:
                    dest_key = key[:10] + 'new/' + key[10:]
                else:
                    dest_key = 'new/' + key
                assert source[key] == dest[dest_key]

    def test_source_dest_path(self):
        source = self.source
        # paths should be normalized
        for source_path in 'bar', 'bar/', '/bar', '/bar/':
            for dest_path in 'new', 'new/', '/new', '/new/':
                dest = self._get_dest_store()
                copy_store(source, dest, source_path=source_path,
                           dest_path=dest_path)
                assert 2 == len(dest)
                for key in source:
                    if key.startswith('bar/'):
                        dest_key = 'new/' + key.split('bar/')[1]
                        assert source[key] == dest[dest_key]
                    else:
                        assert key not in dest
                        assert ('new/' + key) not in dest

    def test_excludes_includes(self):
        source = self.source

        # single excludes
        dest = self._get_dest_store()
        excludes = 'f.*'
        copy_store(source, dest, excludes=excludes)
        assert len(dest) == 2

        root = '' if self._version == 2 else meta_root
        assert root + 'foo' not in dest

        # multiple excludes
        dest = self._get_dest_store()
        excludes = 'b.z', '.*x'
        copy_store(source, dest, excludes=excludes)
        assert len(dest) == 1
        assert root + 'foo' in dest
        assert root + 'bar/baz' not in dest
        assert root + 'bar/qux' not in dest

        # excludes and includes
        dest = self._get_dest_store()
        excludes = 'b.*'
        includes = '.*x'
        copy_store(source, dest, excludes=excludes, includes=includes)
        assert len(dest) == 2
        assert root + 'foo' in dest
        assert root + 'bar/baz' not in dest
        assert root + 'bar/qux' in dest

    def test_dry_run(self):
        source = self.source
        dest = self._get_dest_store()
        copy_store(source, dest, dry_run=True)
        assert 0 == len(dest)

    def test_if_exists(self):
        source = self.source
        dest = self._get_dest_store()
        root = '' if self._version == 2 else meta_root
        dest[root + 'bar/baz'] = b'mmm'

        # default ('raise')
        with pytest.raises(CopyError):
            copy_store(source, dest)

        # explicit 'raise'
        with pytest.raises(CopyError):
            copy_store(source, dest, if_exists='raise')

        # skip
        copy_store(source, dest, if_exists='skip')
        assert 3 == len(dest)
        assert dest[root + 'foo'] == b'xxx'
        assert dest[root + 'bar/baz'] == b'mmm'
        assert dest[root + 'bar/qux'] == b'zzz'

        # replace
        copy_store(source, dest, if_exists='replace')
        assert 3 == len(dest)
        assert dest[root + 'foo'] == b'xxx'
        assert dest[root + 'bar/baz'] == b'yyy'
        assert dest[root + 'bar/qux'] == b'zzz'

        # invalid option
        with pytest.raises(ValueError):
            copy_store(source, dest, if_exists='foobar')


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestCopyStoreV3(TestCopyStore):

    _version = 3

    def setUp(self):
        source = KVStoreV3(dict())
        source['meta/root/foo'] = b'xxx'
        source['meta/root/bar/baz'] = b'yyy'
        source['meta/root/bar/qux'] = b'zzz'
        self.source = source

    def _get_dest_store(self):
        return KVStoreV3(dict())

    def test_mismatched_store_versions(self):
        # cannot copy between stores of mixed Zarr versions
        dest = KVStore(dict())
        with pytest.raises(ValueError):
            copy_store(self.source, dest)


def check_copied_array(original, copied, without_attrs=False,
                       expect_props=None):

    # setup
    source_h5py = original.__module__.startswith('h5py.')
    dest_h5py = copied.__module__.startswith('h5py.')
    zarr_to_zarr = not (source_h5py or dest_h5py)
    h5py_to_h5py = source_h5py and dest_h5py
    zarr_to_h5py = not source_h5py and dest_h5py
    h5py_to_zarr = source_h5py and not dest_h5py
    if expect_props is None:
        expect_props = dict()
    else:
        expect_props = expect_props.copy()

    # common properties in zarr and h5py
    for p in 'dtype', 'shape', 'chunks':
        expect_props.setdefault(p, getattr(original, p))

    # zarr-specific properties
    if zarr_to_zarr:
        for p in 'compressor', 'filters', 'order', 'fill_value':
            expect_props.setdefault(p, getattr(original, p))

    # h5py-specific properties
    if h5py_to_h5py:
        for p in ('maxshape', 'compression', 'compression_opts', 'shuffle',
                  'scaleoffset', 'fletcher32', 'fillvalue'):
            expect_props.setdefault(p, getattr(original, p))

    # common properties with some name differences
    if h5py_to_zarr:
        expect_props.setdefault('fill_value', original.fillvalue)
    if zarr_to_h5py:
        expect_props.setdefault('fillvalue', original.fill_value)

    # compare properties
    for k, v in expect_props.items():
        assert v == getattr(copied, k)

    # compare data
    assert_array_equal(original[:], copied[:])

    # compare attrs
    if without_attrs:
        for k in original.attrs.keys():
            assert k not in copied.attrs
    else:
        if dest_h5py and 'filters' in original.attrs:
            # special case in v3 (storing filters metadata under attributes)
            # we explicitly do not copy this info over to HDF5
            original_attrs = original.attrs.asdict().copy()
            original_attrs.pop('filters')
        else:
            original_attrs = original.attrs
        assert sorted(original_attrs.items()) == sorted(copied.attrs.items())


def check_copied_group(original, copied, without_attrs=False, expect_props=None,
                       shallow=False):

    # setup
    if expect_props is None:
        expect_props = dict()
    else:
        expect_props = expect_props.copy()

    # compare children
    for k, v in original.items():
        if hasattr(v, 'shape'):
            assert k in copied
            check_copied_array(v, copied[k], without_attrs=without_attrs,
                               expect_props=expect_props)
        elif shallow:
            assert k not in copied
        else:
            assert k in copied
            check_copied_group(v, copied[k], without_attrs=without_attrs,
                               shallow=shallow, expect_props=expect_props)

    # compare attrs
    if without_attrs:
        for k in original.attrs.keys():
            assert k not in copied.attrs
    else:
        assert sorted(original.attrs.items()) == sorted(copied.attrs.items())


def test_copy_all():
    """
    https://github.com/zarr-developers/zarr-python/issues/269

    copy_all used to not copy attributes as `.keys()` does not return hidden `.zattrs`.

    """
    original_group = zarr.group(store=MemoryStore(), overwrite=True)
    original_group.attrs["info"] = "group attrs"
    original_subgroup = original_group.create_group("subgroup")
    original_subgroup.attrs["info"] = "sub attrs"

    destination_group = zarr.group(store=MemoryStore(), overwrite=True)

    # copy from memory to directory store
    copy_all(
        original_group,
        destination_group,
        dry_run=False,
    )

    assert 'subgroup' in destination_group
    assert destination_group.attrs["info"] == "group attrs"
    assert destination_group.subgroup.attrs["info"] == "sub attrs"


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
def test_copy_all_v3():
    """
    https://github.com/zarr-developers/zarr-python/issues/269

    copy_all used to not copy attributes as `.keys()`

    """
    original_group = zarr.group(store=MemoryStoreV3(), path='group1', overwrite=True)
    original_group.create_group("subgroup")

    destination_group = zarr.group(store=MemoryStoreV3(), path='group2', overwrite=True)

    # copy from memory to directory store
    copy_all(
        original_group,
        destination_group,
        dry_run=False,
    )
    assert 'subgroup' in destination_group


class TestCopy:
    @pytest.fixture(params=[False, True], ids=['zarr', 'hdf5'])
    def source(self, request, tmpdir):
        def prep_source(source):
            foo = source.create_group('foo')
            foo.attrs['experiment'] = 'weird science'
            baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
            baz.attrs['units'] = 'metres'
            if request.param:
                extra_kws = dict(compression='gzip', compression_opts=3, fillvalue=84,
                                 shuffle=True, fletcher32=True)
            else:
                extra_kws = dict(compressor=Zlib(3), order='F', fill_value=42, filters=[Adler32()])
            source.create_dataset('spam', data=np.arange(100, 200).reshape(20, 5),
                                  chunks=(10, 2), dtype='i2', **extra_kws)
            return source

        if request.param:
            h5py = pytest.importorskip('h5py')
            fn = tmpdir.join('source.h5')
            with h5py.File(str(fn), mode='w') as h5f:
                yield prep_source(h5f)
        else:
            yield prep_source(group())

    @pytest.fixture(params=[False, True], ids=['zarr', 'hdf5'])
    def dest(self, request, tmpdir):
        if request.param:
            h5py = pytest.importorskip('h5py')
            fn = tmpdir.join('dest.h5')
            with h5py.File(str(fn), mode='w') as h5f:
                yield h5f
        else:
            yield group()

    def test_copy_array(self, source, dest):
        # copy array with default options
        copy(source['foo/bar/baz'], dest)
        check_copied_array(source['foo/bar/baz'], dest['baz'])
        copy(source['spam'], dest)
        check_copied_array(source['spam'], dest['spam'])

    def test_copy_bad_dest(self, source, dest):
        # try to copy to an array, dest must be a group
        dest = dest.create_dataset('eggs', shape=(100,))
        with pytest.raises(ValueError):
            copy(source['foo/bar/baz'], dest)

    def test_copy_array_name(self, source, dest):
        # copy array with name
        copy(source['foo/bar/baz'], dest, name='qux')
        assert 'baz' not in dest
        check_copied_array(source['foo/bar/baz'], dest['qux'])

    def test_copy_array_create_options(self, source, dest):
        dest_h5py = dest.__module__.startswith('h5py.')

        # copy array, provide creation options
        compressor = Zlib(9)
        create_kws = dict(chunks=(10,))
        if dest_h5py:
            create_kws.update(compression='gzip', compression_opts=9,
                              shuffle=True, fletcher32=True, fillvalue=42)
        else:
            create_kws.update(compressor=compressor, fill_value=42, order='F',
                              filters=[Adler32()])
        copy(source['foo/bar/baz'], dest, without_attrs=True, **create_kws)
        check_copied_array(source['foo/bar/baz'], dest['baz'],
                           without_attrs=True, expect_props=create_kws)

    def test_copy_array_exists_array(self, source, dest):
        # copy array, dest array in the way
        dest.create_dataset('baz', shape=(10,))

        # raise
        with pytest.raises(CopyError):
            # should raise by default
            copy(source['foo/bar/baz'], dest)
        assert (10,) == dest['baz'].shape
        with pytest.raises(CopyError):
            copy(source['foo/bar/baz'], dest, if_exists='raise')
        assert (10,) == dest['baz'].shape

        # skip
        copy(source['foo/bar/baz'], dest, if_exists='skip')
        assert (10,) == dest['baz'].shape

        # replace
        copy(source['foo/bar/baz'], dest, if_exists='replace')
        check_copied_array(source['foo/bar/baz'], dest['baz'])

        # invalid option
        with pytest.raises(ValueError):
            copy(source['foo/bar/baz'], dest, if_exists='foobar')

    def test_copy_array_exists_group(self, source, dest):
        # copy array, dest group in the way
        dest.create_group('baz')

        # raise
        with pytest.raises(CopyError):
            copy(source['foo/bar/baz'], dest)
        assert not hasattr(dest['baz'], 'shape')
        with pytest.raises(CopyError):
            copy(source['foo/bar/baz'], dest, if_exists='raise')
        assert not hasattr(dest['baz'], 'shape')

        # skip
        copy(source['foo/bar/baz'], dest, if_exists='skip')
        assert not hasattr(dest['baz'], 'shape')

        # replace
        copy(source['foo/bar/baz'], dest, if_exists='replace')
        check_copied_array(source['foo/bar/baz'], dest['baz'])

    def test_copy_array_skip_initialized(self, source, dest):
        dest_h5py = dest.__module__.startswith('h5py.')

        dest.create_dataset('baz', shape=(100,), chunks=(10,), dtype='i8')
        assert not np.all(source['foo/bar/baz'][:] == dest['baz'][:])

        if dest_h5py:
            with pytest.raises(ValueError):
                # not available with copy to h5py
                copy(source['foo/bar/baz'], dest, if_exists='skip_initialized')

        else:
            # copy array, dest array exists but not yet initialized
            copy(source['foo/bar/baz'], dest, if_exists='skip_initialized')
            check_copied_array(source['foo/bar/baz'], dest['baz'])

            # copy array, dest array exists and initialized, will be skipped
            dest['baz'][:] = np.arange(100, 200)
            copy(source['foo/bar/baz'], dest, if_exists='skip_initialized')
            assert_array_equal(np.arange(100, 200), dest['baz'][:])
            assert not np.all(source['foo/bar/baz'][:] == dest['baz'][:])

    def test_copy_group(self, source, dest):
        # copy group, default options
        copy(source['foo'], dest)
        check_copied_group(source['foo'], dest['foo'])

    def test_copy_group_no_name(self, source, dest):
        with pytest.raises(TypeError):
            # need a name if copy root
            copy(source, dest)

        copy(source, dest, name='root')
        check_copied_group(source, dest['root'])

    def test_copy_group_options(self, source, dest):
        # copy group, non-default options
        copy(source['foo'], dest, name='qux', without_attrs=True)
        assert 'foo' not in dest
        check_copied_group(source['foo'], dest['qux'], without_attrs=True)

    def test_copy_group_shallow(self, source, dest):
        # copy group, shallow
        copy(source, dest, name='eggs', shallow=True)
        check_copied_group(source, dest['eggs'], shallow=True)

    def test_copy_group_exists_group(self, source, dest):
        # copy group, dest groups exist
        dest.create_group('foo/bar')
        copy(source['foo'], dest)
        check_copied_group(source['foo'], dest['foo'])

    def test_copy_group_exists_array(self, source, dest):
        # copy group, dest array in the way
        dest.create_dataset('foo/bar', shape=(10,))

        # raise
        with pytest.raises(CopyError):
            copy(source['foo'], dest)
        assert dest['foo/bar'].shape == (10,)
        with pytest.raises(CopyError):
            copy(source['foo'], dest, if_exists='raise')
        assert dest['foo/bar'].shape == (10,)

        # skip
        copy(source['foo'], dest, if_exists='skip')
        assert dest['foo/bar'].shape == (10,)

        # replace
        copy(source['foo'], dest, if_exists='replace')
        check_copied_group(source['foo'], dest['foo'])

    def test_copy_group_dry_run(self, source, dest):
        # dry run, empty destination
        n_copied, n_skipped, n_bytes_copied = \
            copy(source['foo'], dest, dry_run=True, return_stats=True)
        assert 0 == len(dest)
        assert 3 == n_copied
        assert 0 == n_skipped
        assert 0 == n_bytes_copied

        # dry run, array exists in destination
        baz = np.arange(100, 200)
        dest.create_dataset('foo/bar/baz', data=baz)
        assert not np.all(source['foo/bar/baz'][:] == dest['foo/bar/baz'][:])
        assert 1 == len(dest)

        # raise
        with pytest.raises(CopyError):
            copy(source['foo'], dest, dry_run=True)
        assert 1 == len(dest)

        # skip
        n_copied, n_skipped, n_bytes_copied = \
            copy(source['foo'], dest, dry_run=True, if_exists='skip',
                 return_stats=True)
        assert 1 == len(dest)
        assert 2 == n_copied
        assert 1 == n_skipped
        assert 0 == n_bytes_copied
        assert_array_equal(baz, dest['foo/bar/baz'])

        # replace
        n_copied, n_skipped, n_bytes_copied = \
            copy(source['foo'], dest, dry_run=True, if_exists='replace',
                 return_stats=True)
        assert 1 == len(dest)
        assert 3 == n_copied
        assert 0 == n_skipped
        assert 0 == n_bytes_copied
        assert_array_equal(baz, dest['foo/bar/baz'])

    def test_logging(self, source, dest, tmpdir):
        # callable log
        copy(source['foo'], dest, dry_run=True, log=print)

        # file name
        fn = str(tmpdir.join('log_name'))
        copy(source['foo'], dest, dry_run=True, log=fn)

        # file
        with tmpdir.join('log_file').open(mode='w') as f:
            copy(source['foo'], dest, dry_run=True, log=f)

        # bad option
        with pytest.raises(TypeError):
            copy(source['foo'], dest, dry_run=True, log=True)


@pytest.mark.skipif(not v3_api_available, reason="V3 is disabled")
class TestCopyV3(TestCopy):

    @pytest.fixture(params=['zarr', 'hdf5'])
    def source(self, request, tmpdir):
        def prep_source(source):
            foo = source.create_group('foo')
            foo.attrs['experiment'] = 'weird science'
            baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
            baz.attrs['units'] = 'metres'
            if request.param == 'hdf5':
                extra_kws = dict(compression='gzip', compression_opts=3, fillvalue=84,
                                 shuffle=True, fletcher32=True)
            else:
                extra_kws = dict(compressor=Zlib(3), order='F', fill_value=42, filters=[Adler32()])
            source.create_dataset('spam', data=np.arange(100, 200).reshape(20, 5),
                                  chunks=(10, 2), dtype='i2', **extra_kws)
            return source

        if request.param == 'hdf5':
            h5py = pytest.importorskip('h5py')
            fn = tmpdir.join('source.h5')
            with h5py.File(str(fn), mode='w') as h5f:
                yield prep_source(h5f)
        elif request.param == 'zarr':
            yield prep_source(group(path='group1', zarr_version=3))

    # Test with various destination StoreV3 types as TestCopyV3 covers rmdir
    destinations = ['hdf5', 'zarr', 'zarr_kvstore', 'zarr_directorystore', 'zarr_sqlitestore']
    if have_fsspec:
        destinations += ['zarr_fsstore']

    @pytest.fixture(params=destinations)
    def dest(self, request, tmpdir):
        if request.param == 'hdf5':
            h5py = pytest.importorskip('h5py')
            fn = tmpdir.join('dest.h5')
            with h5py.File(str(fn), mode='w') as h5f:
                yield h5f
        elif request.param == 'zarr':
            yield group(path='group2', zarr_version=3)
        elif request.param == 'zarr_kvstore':
            store = KVStoreV3(dict())
            yield group(store, path='group2', zarr_version=3)
        elif request.param == 'zarr_fsstore':
            fn = tmpdir.join('dest.zr3')
            store = FSStoreV3(str(fn), auto_mkdir=True)
            yield group(store, path='group2', zarr_version=3)
        elif request.param == 'zarr_directorystore':
            fn = tmpdir.join('dest.zr3')
            store = DirectoryStoreV3(str(fn))
            yield group(store, path='group2', zarr_version=3)
        elif request.param == 'zarr_sqlitestore':
            fn = tmpdir.join('dest.db')
            store = SQLiteStoreV3(str(fn))
            yield group(store, path='group2', zarr_version=3)

    def test_copy_array_create_options(self, source, dest):
        dest_h5py = dest.__module__.startswith('h5py.')

        # copy array, provide creation options
        compressor = Zlib(9)
        create_kws = dict(chunks=(10,))
        if dest_h5py:
            create_kws.update(compression='gzip', compression_opts=9,
                              shuffle=True, fletcher32=True, fillvalue=42)
        else:
            # v3 case has no filters argument in zarr create_kws
            create_kws.update(compressor=compressor, fill_value=42, order='F')
        copy(source['foo/bar/baz'], dest, without_attrs=True, **create_kws)
        check_copied_array(source['foo/bar/baz'], dest['baz'],
                           without_attrs=True, expect_props=create_kws)

    def test_copy_group_no_name(self, source, dest):
        if source.__module__.startswith('h5py'):
            with pytest.raises(TypeError):
                copy(source, dest)
        else:
            # For v3, dest.name will be inferred from source.name
            copy(source, dest)
            check_copied_group(source, dest[source.name.lstrip('/')])

        copy(source, dest, name='root')
        check_copied_group(source, dest['root'])
