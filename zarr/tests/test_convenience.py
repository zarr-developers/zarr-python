# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit
import os
import unittest


import numpy as np
from numpy.testing import assert_array_equal
from numcodecs import Zlib, Adler32
import pytest


from zarr.convenience import open, save, save_group, load, copy_store, copy
from zarr.storage import atexit_rmtree
from zarr.core import Array
from zarr.hierarchy import Group, group
from zarr.errors import CopyError


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
    with pytest.raises(ValueError):
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


# noinspection PyAttributeOutsideInit
class TestCopy(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCopy, self).__init__(*args, **kwargs)
        self.source_h5py = False
        self.dest_h5py = False
        self.new_source = group
        self.new_dest = group

    def setUp(self):
        source = self.new_source()
        foo = source.create_group('foo')
        foo.attrs['experiment'] = 'weird science'
        baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
        baz.attrs['units'] = 'metres'
        if self.source_h5py:
            extra_kws = dict(compression='gzip', compression_opts=3, fillvalue=84,
                             shuffle=True, fletcher32=True)
        else:
            extra_kws = dict(compressor=Zlib(3), order='F', fill_value=42,
                             filters=[Adler32()])
        source.create_dataset('spam', data=np.arange(100, 200).reshape(20, 5),
                              chunks=(10, 2), dtype='i2', **extra_kws)
        self.source = source

    def test_copy_array(self):
        source = self.source
        dest = self.new_dest()

        # copy array with default options
        copy(source['foo/bar/baz'], dest)
        check_copied_array(source['foo/bar/baz'], dest['baz'])
        copy(source['spam'], dest)
        check_copied_array(source['spam'], dest['spam'])

    def test_copy_bad_dest(self):
        source = self.source

        # try to copy to an array, dest must be a group
        dest = self.new_dest().create_dataset('eggs', shape=(100,))
        with pytest.raises(ValueError):
            copy(source['foo/bar/baz'], dest)

    def test_copy_array_name(self):
        source = self.source
        dest = self.new_dest()

        # copy array with name
        copy(source['foo/bar/baz'], dest, name='qux')
        assert 'baz' not in dest
        check_copied_array(source['foo/bar/baz'], dest['qux'])

    def test_copy_array_create_options(self):
        source = self.source
        dest = self.new_dest()

        # copy array, provide creation options
        compressor = Zlib(9)
        create_kws = dict(chunks=(10,))
        if self.dest_h5py:
            create_kws.update(compression='gzip', compression_opts=9,
                              shuffle=True, fletcher32=True, fillvalue=42)
        else:
            create_kws.update(compressor=compressor, fill_value=42, order='F',
                              filters=[Adler32()])
        copy(source['foo/bar/baz'], dest, without_attrs=True, **create_kws)
        check_copied_array(source['foo/bar/baz'], dest['baz'],
                           without_attrs=True, expect_props=create_kws)

    def test_copy_array_exists_array(self):
        source = self.source
        dest = self.new_dest()

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

    def test_copy_array_exists_group(self):
        source = self.source
        dest = self.new_dest()

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

    def test_copy_array_skip_initialized(self):
        source = self.source
        dest = self.new_dest()
        dest.create_dataset('baz', shape=(100,), chunks=(10,), dtype='i8')
        assert not np.all(source['foo/bar/baz'][:] == dest['baz'][:])

        if self.dest_h5py:
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

    def test_copy_group(self):
        source = self.source
        dest = self.new_dest()

        # copy group, default options
        copy(source['foo'], dest)
        check_copied_group(source['foo'], dest['foo'])

    def test_copy_group_no_name(self):
        source = self.source
        dest = self.new_dest()

        with pytest.raises(TypeError):
            # need a name if copy root
            copy(source, dest)

        copy(source, dest, name='root')
        check_copied_group(source, dest['root'])

    def test_copy_group_options(self):
        source = self.source
        dest = self.new_dest()

        # copy group, non-default options
        copy(source['foo'], dest, name='qux', without_attrs=True)
        assert 'foo' not in dest
        check_copied_group(source['foo'], dest['qux'], without_attrs=True)

    def test_copy_group_shallow(self):
        source = self.source
        dest = self.new_dest()

        # copy group, shallow
        copy(source, dest, name='eggs', shallow=True)
        check_copied_group(source, dest['eggs'], shallow=True)

    def test_copy_group_exists_group(self):
        source = self.source
        dest = self.new_dest()

        # copy group, dest groups exist
        dest.create_group('foo/bar')
        copy(source['foo'], dest)
        check_copied_group(source['foo'], dest['foo'])

    def test_copy_group_exists_array(self):
        source = self.source
        dest = self.new_dest()

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

    def test_copy_group_dry_run(self):
        source = self.source
        dest = self.new_dest()

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

    def test_logging(self):
        source = self.source
        dest = self.new_dest()

        # callable log
        copy(source['foo'], dest, dry_run=True, log=print)

        # file name
        fn = tempfile.mktemp()
        atexit.register(os.remove, fn)
        copy(source['foo'], dest, dry_run=True, log=fn)

        # file
        with tempfile.TemporaryFile(mode='w') as f:
            copy(source['foo'], dest, dry_run=True, log=f)

        # bad option
        with pytest.raises(TypeError):
            copy(source['foo'], dest, dry_run=True, log=True)


try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def temp_h5f():
    fn = tempfile.mktemp()
    atexit.register(os.remove, fn)
    h5f = h5py.File(fn, mode='w')
    atexit.register(lambda v: v.close(), h5f)
    return h5f


@unittest.skipIf(h5py is None, 'h5py is not installed')
class TestCopyHDF5ToZarr(TestCopy):

    def __init__(self, *args, **kwargs):
        super(TestCopyHDF5ToZarr, self).__init__(*args, **kwargs)
        self.source_h5py = True
        self.dest_h5py = False
        self.new_source = temp_h5f
        self.new_dest = group


@unittest.skipIf(h5py is None, 'h5py is not installed')
class TestCopyZarrToHDF5(TestCopy):

    def __init__(self, *args, **kwargs):
        super(TestCopyZarrToHDF5, self).__init__(*args, **kwargs)
        self.source_h5py = False
        self.dest_h5py = True
        self.new_source = group
        self.new_dest = temp_h5f


@unittest.skipIf(h5py is None, 'h5py is not installed')
class TestCopyHDF5ToHDF5(TestCopy):

    def __init__(self, *args, **kwargs):
        super(TestCopyHDF5ToHDF5, self).__init__(*args, **kwargs)
        self.source_h5py = True
        self.dest_h5py = True
        self.new_source = temp_h5f
        self.new_dest = temp_h5f
