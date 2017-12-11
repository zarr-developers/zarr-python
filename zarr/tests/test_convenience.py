# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import tempfile
import atexit
import os


from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_equal
from numcodecs import Zlib, Adler32
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


def test_copy_store_no_paths():
    source = dict()
    source['foo'] = b'xxx'
    source['bar'] = b'yyy'
    dest = dict()
    copy_store(source, dest)
    assert len(dest) == 2
    for key in source:
        assert source[key] == dest[key]


def test_copy_store_source_path():
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


def test_copy_store_dest_path():
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


def test_copy_store_source_dest_path():
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


def test_copy_store_excludes_includes():
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


def test_copy_store_dry_run():
    source = dict()
    source['foo'] = b'xxx'
    source['bar/baz'] = b'yyy'
    source['bar/qux'] = b'zzz'
    dest = dict()
    copy_store(source, dest, dry_run=True)
    assert 0 == len(dest)


def test_copy_store_if_exists():

    # setup
    source = dict()
    source['foo'] = b'xxx'
    source['bar/baz'] = b'yyy'
    source['bar/qux'] = b'zzz'
    dest = dict()
    dest['bar/baz'] = b'mmm'

    # default ('raise')
    with pytest.raises(ValueError):
        copy_store(source, dest)

    # explicit 'raise'
    with pytest.raises(ValueError):
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


def check_copied_array(original, copied, without_attrs=False, expect_props=None):

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


def _test_copy(new_source, new_dest):

    source = new_source()
    dest = new_dest()
    # source_h5py = source.__module__.startswith('h5py.')
    dest_h5py = dest.__module__.startswith('h5py.')

    # setup source
    foo = source.create_group('foo')
    foo.attrs['experiment'] = 'weird science'
    baz = foo.create_dataset('bar/baz', data=np.arange(100), chunks=(50,))
    baz.attrs['units'] = 'metres'
    source.create_dataset('spam', data=np.arange(100, 200).reshape(20, 5),
                          chunks=(10, 2))

    # copy array with default options
    copy(source['foo/bar/baz'], dest)
    check_copied_array(source['foo/bar/baz'], dest['baz'])

    # copy array with name
    dest = new_dest()
    copy(source['foo/bar/baz'], dest, name='qux')
    assert 'baz' not in dest
    check_copied_array(source['foo/bar/baz'], dest['qux'])

    # copy array, provide creation options
    dest = new_dest()
    compressor = Zlib(9)
    create_kws = dict(chunks=(10,))
    if dest_h5py:
        create_kws.update(compression='gzip', compression_opts=9, shuffle=True,
                          fletcher32=True, fillvalue=42)
    else:
        create_kws.update(compressor=compressor, fill_value=42, order='F',
                          filters=[Adler32()])
    copy(source['foo/bar/baz'], dest, without_attrs=True, **create_kws)
    check_copied_array(source['foo/bar/baz'], dest['baz'], without_attrs=True,
                       expect_props=create_kws)

    # copy array, dest array in the way
    dest = new_dest()
    dest.create_dataset('baz', shape=(10,))
    with pytest.raises(ValueError):
        copy(source['foo/bar/baz'], dest)
    assert (10,) == dest['baz'].shape
    copy(source['foo/bar/baz'], dest, if_exists='replace')
    check_copied_array(source['foo/bar/baz'], dest['baz'])

    # copy array, dest group in the way
    dest = new_dest()
    dest.create_group('baz')
    with pytest.raises(ValueError):
        copy(source['foo/bar/baz'], dest)
    assert not hasattr(dest['baz'], 'shape')
    copy(source['foo/bar/baz'], dest, if_exists='replace')
    check_copied_array(source['foo/bar/baz'], dest['baz'])

    # copy group, default options
    dest = new_dest()
    copy(source['foo'], dest)
    check_copied_group(source['foo'], dest['foo'])

    # copy group, non-default options
    dest = new_dest()
    copy(source['foo'], dest, name='qux', without_attrs=True)
    assert 'foo' not in dest
    check_copied_group(source['foo'], dest['qux'], without_attrs=True)

    # copy group, shallow
    dest = new_dest()
    copy(source, dest, name='eggs', shallow=True)
    check_copied_group(source, dest['eggs'], shallow=True)

    # copy group, dest groups exist
    dest = new_dest()
    dest.create_group('foo/bar')
    copy(source['foo'], dest)
    check_copied_group(source['foo'], dest['foo'])

    # copy group, dest array in the way
    dest = new_dest()
    dest.create_dataset('foo/bar', shape=(10,))
    with pytest.raises(ValueError):
        copy(source['foo'], dest)
    assert dest['foo/bar'].shape == (10,)
    copy(source['foo'], dest, if_exists='replace')
    check_copied_group(source['foo'], dest['foo'])


def test_copy_zarr_to_zarr():
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
def test_copy_h5py_to_zarr():
    _test_copy(temp_h5f, group)


@pytest.mark.skipif(not have_h5py, reason='h5py not installed')
def test_copy_zarr_to_h5py():
    _test_copy(group, temp_h5f)


@pytest.mark.skipif(not have_h5py, reason='h5py not installed')
def test_copy_h5py_to_h5py():
    _test_copy(temp_h5f, temp_h5f)
