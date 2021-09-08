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
    copy_all,
)
from zarr.core import Array
from zarr.errors import CopyError
from zarr.hierarchy import Group, group
from zarr.storage import (ConsolidatedMetadataStore, MemoryStore,
                          atexit_rmtree, getsize)


def test_open_array(path_type):

    store = tempfile.mkdtemp()
    atexit.register(atexit_rmtree, store)
    store = path_type(store)

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
    with pytest.raises(ValueError):
        open('doesnotexist', mode='r')


def test_open_group(path_type):

    store = tempfile.mkdtemp()
    atexit.register(atexit_rmtree, store)
    store = path_type(store)

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
    with pytest.raises(ValueError):
        # no arrays provided
        save_group('data/group.zarr')
    with pytest.raises(ValueError):
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


def test_consolidate_metadata():

    # setup initial data
    store = MemoryStore()
    z = group(store)
    z.create_group('g1')
    g2 = z.create_group('g2')
    g2.attrs['hello'] = 'world'
    arr = g2.create_dataset('arr', shape=(20, 20), chunks=(5, 5), dtype='f8')
    assert 16 == arr.nchunks
    assert 0 == arr.nchunks_initialized
    arr.attrs['data'] = 1
    arr[:] = 1.0
    assert 16 == arr.nchunks_initialized

    # perform consolidation
    out = consolidate_metadata(store)
    assert isinstance(out, Group)
    assert '.zmetadata' in store
    for key in ['.zgroup',
                'g1/.zgroup',
                'g2/.zgroup',
                'g2/.zattrs',
                'g2/arr/.zarray',
                'g2/arr/.zattrs']:
        del store[key]

    # open consolidated
    z2 = open_consolidated(store)
    assert ['g1', 'g2'] == list(z2)
    assert 'world' == z2.g2.attrs['hello']
    assert 1 == z2.g2.arr.attrs['data']
    assert (z2.g2.arr[:] == 1.0).all()
    assert 16 == z2.g2.arr.nchunks
    assert 16 == z2.g2.arr.nchunks_initialized

    # tests del/write on the store
    cmd = ConsolidatedMetadataStore(store)
    with pytest.raises(PermissionError):
        del cmd['.zgroup']
    with pytest.raises(PermissionError):
        cmd['.zgroup'] = None

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
        open_consolidated(store, mode='a')
    with pytest.raises(ValueError):
        open_consolidated(store, mode='w')

    # make sure keyword arguments are passed through without error
    open_consolidated(store, cache_attrs=True, synchronizer=None)


def test_consolidated_with_chunk_store():
    # setup initial data
    store = MemoryStore()
    chunk_store = MemoryStore()
    z = group(store, chunk_store=chunk_store)
    z.create_group('g1')
    g2 = z.create_group('g2')
    g2.attrs['hello'] = 'world'
    arr = g2.create_dataset('arr', shape=(20, 20), chunks=(5, 5), dtype='f8')
    assert 16 == arr.nchunks
    assert 0 == arr.nchunks_initialized
    arr.attrs['data'] = 1
    arr[:] = 1.0
    assert 16 == arr.nchunks_initialized

    # perform consolidation
    out = consolidate_metadata(store)
    assert isinstance(out, Group)
    assert '.zmetadata' in store
    for key in ['.zgroup',
                'g1/.zgroup',
                'g2/.zgroup',
                'g2/.zattrs',
                'g2/arr/.zarray',
                'g2/arr/.zattrs']:
        del store[key]
    # open consolidated
    z2 = open_consolidated(store, chunk_store=chunk_store)
    assert ['g1', 'g2'] == list(z2)
    assert 'world' == z2.g2.attrs['hello']
    assert 1 == z2.g2.arr.attrs['data']
    assert (z2.g2.arr[:] == 1.0).all()
    assert 16 == z2.g2.arr.nchunks
    assert 16 == z2.g2.arr.nchunks_initialized

    # test the data are writeable
    z2.g2.arr[:] = 2
    assert (z2.g2.arr[:] == 2).all()

    # test invalid modes
    with pytest.raises(ValueError):
        open_consolidated(store, mode='a', chunk_store=chunk_store)
    with pytest.raises(ValueError):
        open_consolidated(store, mode='w', chunk_store=chunk_store)

    # make sure keyword arguments are passed through without error
    open_consolidated(store, cache_attrs=True, synchronizer=None,
                      chunk_store=chunk_store)


class TestCopyStore(unittest.TestCase):

    def setUp(self):
        source = dict()
        source['foo'] = b'xxx'
        source['bar/baz'] = b'yyy'
        source['bar/qux'] = b'zzz'
        self.source = source

    def test_no_paths(self):
        source = self.source
        dest = dict()
        copy_store(source, dest)
        assert len(source) == len(dest)
        for key in source:
            assert source[key] == dest[key]

    def test_source_path(self):
        source = self.source
        # paths should be normalized
        for source_path in 'bar', 'bar/', '/bar', '/bar/':
            dest = dict()
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
            dest = dict()
            copy_store(source, dest, dest_path=dest_path)
            assert len(source) == len(dest)
            for key in source:
                dest_key = 'new/' + key
                assert source[key] == dest[dest_key]

    def test_source_dest_path(self):
        source = self.source
        # paths should be normalized
        for source_path in 'bar', 'bar/', '/bar', '/bar/':
            for dest_path in 'new', 'new/', '/new', '/new/':
                dest = dict()
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

    def test_dry_run(self):
        source = self.source
        dest = dict()
        copy_store(source, dest, dry_run=True)
        assert 0 == len(dest)

    def test_if_exists(self):
        source = self.source
        dest = dict()
        dest['bar/baz'] = b'mmm'

        # default ('raise')
        with pytest.raises(CopyError):
            copy_store(source, dest)

        # explicit 'raise'
        with pytest.raises(CopyError):
            copy_store(source, dest, if_exists='raise')

        # skip
        copy_store(source, dest, if_exists='skip')
        assert 3 == len(dest)
        assert dest['foo'] == b'xxx'
        assert dest['bar/baz'] == b'mmm'
        assert dest['bar/qux'] == b'zzz'

        # replace
        copy_store(source, dest, if_exists='replace')
        assert 3 == len(dest)
        assert dest['foo'] == b'xxx'
        assert dest['bar/baz'] == b'yyy'
        assert dest['bar/qux'] == b'zzz'

        # invalid option
        with pytest.raises(ValueError):
            copy_store(source, dest, if_exists='foobar')


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
        assert sorted(original.attrs.items()) == sorted(copied.attrs.items())


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

    assert destination_group.attrs["info"] == "group attrs"
    assert destination_group.subgroup.attrs["info"] == "sub attrs"


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
