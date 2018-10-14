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
from numcodecs.abc import Codec
from numcodecs.registry import register_codec, get_codec
import json

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

    assert 'compression' in array_metadata
    assert array_metadata['compression']['id'] == N5ChunkWrapper.codec_id

    compressor_config = array_metadata['compression']['compressor_config']
    if 'cname' in compressor_config:
        compressor_config['type'] = compressor_config['cname']
        del compressor_config['cname']
    if 'blocksize' in compressor_config:
        compressor_config['blockSize'] = compressor_config['blocksize']
        del compressor_config['blocksize']
    if 'clevel' in compressor_config:
        compressor_config['level'] = compressor_config['clevel']
        del compressor_config['clevel']
    array_metadata['compression'] = compressor_config

    return array_metadata

def array_metadata_to_zarr(array_metadata):
    '''Convert array metadata from N5 to zarr format.'''
    for t, f in zarr_to_n5_keys:
        array_metadata[t] = array_metadata[f]
        del array_metadata[f]
    array_metadata['zarr_format'] = ZARR_FORMAT

    assert 'compressor' in array_metadata

    compressor_config = array_metadata['compressor']
    if 'type' in compressor_config:
        compressor_config['cname'] = compressor_config['type']
        del compressor_config['type']
    if 'blockSize' in compressor_config:
        compressor_config['blocksize'] = compressor_config['blockSize']
        del compressor_config['blockSize']
    if 'level' in compressor_config:
        compressor_config['clevel'] = compressor_config['level']
        del compressor_config['level']
    array_metadata['compressor'] = {
        'id': N5ChunkWrapper.codec_id,
        'compressor_config': compressor_config
    }

    return array_metadata

class N5ChunkWrapper(Codec):

    codec_id = 'n5_wrapper'

    def __init__(self, compressor_config=None, compressor=None):

        if compressor:
            assert compressor_config is None, (
                "Only one of compressor_config or compressor should be given.")
            compressor_config = compressor.get_config()

        self.compressor_config = compressor_config
        self._compressor = get_codec(compressor_config)

    def get_config(self):
        config = {
            'id': self.codec_id,
            'compressor_config': self._compressor.get_config()
        }
        return config

    def encode(self, chunk):

        header = self._create_header(chunk)
        return header + self._compressor.encode(chunk)

    def decode(self, chunk, out=None):

        len_header, header = self._read_header(chunk)
        chunk = chunk[len_header:]
        return self._compressor.decode(chunk, out)

    def _create_header(self, chunk):

        mode = int(0).to_bytes(2, byteorder='big')
        num_dims = len(chunk.shape).to_bytes(2, byteorder='big')
        shape = b''.join(d.to_bytes(4, byteorder='big') for d in chunk.shape)

        return mode + num_dims + shape

    def _read_header(self, chunk):

        mode = int.from_bytes(chunk[0:2], byteorder='big')
        num_dims = int.from_bytes(chunk[2:4], byteorder='big')
        shape = (
            int.from_bytes(chunk[i:i+4], byteorder='big')
            for i in range(4, num_dims*4 + 4)
        )

        len_header = 4 + num_dims*4

        return len_header, {
            'mode': mode,
            'num_dims': num_dims,
            'shape': shape
        }

register_codec(N5ChunkWrapper, 'n5_wrapper')
