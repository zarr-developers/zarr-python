# -*- coding: utf-8 -*-
"""This module contains a storage class to support the N5 format.
"""
from __future__ import absolute_import, print_function, division
from .meta import ZARR_FORMAT
from .storage import (
        NestedDirectoryStore,
        group_meta_key as zarr_group_meta_key,
        array_meta_key as zarr_array_meta_key,
        attrs_key as zarr_attrs_key,
        _prog_ckey)
import json
import re

zarr_to_n5_keys = [
    ('chunks', 'blockSize'),
    ('dtype', 'dataType'),
    ('compressor', 'compression'),
    ('shape', 'dimensions')
]
n5_attrs_key = 'attributes.json'

class N5Store(NestedDirectoryStore):
    """Storage class using directories and files on a standard file system,
    following the N5 format (https://github.com/saalfeldlab/n5).

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.N5Store('data/array.n5')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42

    Store a group::

        >>> store = zarr.N5Store('data/group.n5')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42

    Notes
    -----

    Safe to write in multiple threads or processes.

    """


    # def __init__(self, path):
        # super(N5Store, self).__init__(path)

    def __getitem__(self, key):

        print("Getting key %s"%key)

        if key.endswith(zarr_group_meta_key):
            key = key.replace(zarr_group_meta_key, n5_attrs_key)
            value = group_metadata_to_zarr(json.loads(self[key]))
            return json.dumps(value, ensure_ascii=True).encode('ascii')

        elif key.endswith(zarr_array_meta_key):
            key = key.replace(zarr_array_meta_key, n5_attrs_key)
            value = array_metadata_to_zarr(json.loads(self[key]))
            return json.dumps(value, ensure_ascii=True).encode('ascii')

        elif key.endswith(zarr_attrs_key):
            key = key.replace(zarr_attrs_key, n5_attrs_key)

        key = invert_chunk_coords(key)

        return super(N5Store, self).__getitem__(key)

    def __setitem__(self, key, value):

        print("Setting %s to %s"%(key, value[:10]))

        if key.endswith(zarr_group_meta_key):

            key = key.replace(zarr_group_meta_key, n5_attrs_key)

            if key in self:
                attrs = json.loads(self[key])
                attrs.update(**json.loads(value))
            else:
                attrs = json.loads(value)

            value = json.dumps(group_metadata_to_n5(attrs),
                               indent=4,
                               sort_keys=True,
                               ensure_ascii=True).encode('ascii')

        elif key.endswith(zarr_array_meta_key):

            key = key.replace(zarr_array_meta_key, n5_attrs_key)

            if key in self:
                attrs = json.loads(self[key])
                attrs.update(**json.loads(value))
            else:
                attrs = json.loads(value)

            value = json.dumps(array_metadata_to_n5(attrs),
                               indent=4,
                               sort_keys=True,
                               ensure_ascii=True).encode('ascii')

        elif key.endswith(zarr_attrs_key):
            key = key.replace(zarr_attrs_key, n5_attrs_key)

        key = invert_chunk_coords(key)

        super(N5Store, self).__setitem__(key, value)

    # def __delitem__(self, key):
        # super(N5Store, self).__delitem__(key)

    def __contains__(self, key):

        print("Checking if %s exists"%key)

        if key.endswith(zarr_group_meta_key):
            key = key.replace(zarr_group_meta_key, n5_attrs_key)
            if not key in self:
                return False
            # group if not a dataset (attributes do not contain 'dimensions')
            return 'dimensions' not in json.loads(self[key])

        elif key.endswith(zarr_array_meta_key):
            key = key.replace(zarr_array_meta_key, n5_attrs_key)
            if not key in self:
                return False
            return 'dimensions' in json.loads(self[key])

        elif key.endswith(zarr_attrs_key):
            key = key.replace(zarr_array_meta_key, n5_attrs_key)

        key = invert_chunk_coords(key)

        return super(N5Store, self).__contains__(key)

    def __eq__(self, other):
        return (
            isinstance(other, N5Store) and
            self.path == other.path
        )

    # def listdir(self, path=None):
        # return super(N5Store, self).listdir(path)

def invert_chunk_coords(key):
    segments = list(key.split('/'))
    if segments:
        last_segment = segments[-1]
        if _prog_ckey.match(last_segment):
            coords = list(last_segment.split('.'))
            last_segment = '.'.join(coords[::-1])
            segments = segments[:-1] + [last_segment]
            key = '/'.join(segments)
    return key

def group_metadata_to_n5(group_metadata):
    '''Convert group metadata from zarr to N5 format.'''
    del group_metadata['zarr_format']
    group_metadata['n5'] = '2.0.0'
    return group_metadata

def group_metadata_to_zarr(group_metadata):
    '''Convert group metadata from N5 to zarr format.'''
    del group_metadata['n5']
    group_metadata['zarr_format'] = ZARR_FORMAT
    return group_metadata

def array_metadata_to_n5(array_metadata):
    '''Convert array metadata from zarr to N5 format.'''

    for f, t in zarr_to_n5_keys:
        array_metadata[t] = array_metadata[f]
        del array_metadata[f]
    del array_metadata['zarr_format']

    if 'compression' in array_metadata:
        compression = array_metadata['compression']
        compression['type'] = compression['cname']
        del compression['cname']
        if 'blocksize' in compression:
            compression['blockSize'] = compression['blocksize']
            del compression['blocksize']
        if 'clevel' in compression:
            compression['level'] = compression['clevel']
            del compression['clevel']

    return array_metadata

def array_metadata_to_zarr(array_metadata):
    '''Convert array metadata from N5 to zarr format.'''
    for t, f in zarr_to_n5_keys:
        array_metadata[t] = array_metadata[f]
        del array_metadata[f]
    array_metadata['zarr_format'] = ZARR_FORMAT

    if 'compressor' in array_metadata:
        compression = array_metadata['compressor']
        compression['cname'] = compression['type']
        del compression['type']
        if 'blockSize' in compression:
            compression['blocksize'] = compression['blockSize']
            del compression['blockSize']
        if 'level' in compression:
            compression['clevel'] = compression['level']
            del compression['level']

    return array_metadata
