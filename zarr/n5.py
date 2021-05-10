"""This module contains a storage class and codec to support the N5 format.
"""
import os
import struct
import sys
import warnings

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy
from numcodecs.registry import get_codec, register_codec

from .meta import ZARR_FORMAT, json_dumps, json_loads
from .storage import NestedDirectoryStore, _prog_ckey, _prog_number
from .storage import array_meta_key as zarr_array_meta_key
from .storage import attrs_key as zarr_attrs_key
from .storage import group_meta_key as zarr_group_meta_key

zarr_to_n5_keys = [
    ('chunks', 'blockSize'),
    ('dtype', 'dataType'),
    ('compressor', 'compression'),
    ('shape', 'dimensions')
]
n5_attrs_key = 'attributes.json'
n5_keywords = ['n5', 'dataType', 'dimensions', 'blockSize', 'compression']


class N5Store(NestedDirectoryStore):
    """Storage class using directories and files on a standard file system,
    following the N5 format (https://github.com/saalfeldlab/n5).

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-senstive and
        case-insensitive file system. Default value is False.

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

    This is an experimental feature.

    Safe to write in multiple threads or processes.

    """

    def __getitem__(self, key):

        if key.endswith(zarr_group_meta_key):

            key = key.replace(zarr_group_meta_key, n5_attrs_key)
            value = group_metadata_to_zarr(self._load_n5_attrs(key))

            return json_dumps(value)

        elif key.endswith(zarr_array_meta_key):

            key = key.replace(zarr_array_meta_key, n5_attrs_key)
            value = array_metadata_to_zarr(self._load_n5_attrs(key))

            return json_dumps(value)

        elif key.endswith(zarr_attrs_key):

            key = key.replace(zarr_attrs_key, n5_attrs_key)
            value = attrs_to_zarr(self._load_n5_attrs(key))

            if len(value) == 0:
                raise KeyError(key)
            else:
                return json_dumps(value)

        elif is_chunk_key(key):

            key = invert_chunk_coords(key)

        return super().__getitem__(key)

    def __setitem__(self, key, value):

        if key.endswith(zarr_group_meta_key):

            key = key.replace(zarr_group_meta_key, n5_attrs_key)

            n5_attrs = self._load_n5_attrs(key)
            n5_attrs.update(**group_metadata_to_n5(json_loads(value)))

            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_array_meta_key):

            key = key.replace(zarr_array_meta_key, n5_attrs_key)

            n5_attrs = self._load_n5_attrs(key)
            n5_attrs.update(**array_metadata_to_n5(json_loads(value)))

            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_attrs_key):

            key = key.replace(zarr_attrs_key, n5_attrs_key)

            n5_attrs = self._load_n5_attrs(key)
            zarr_attrs = json_loads(value)

            for k in n5_keywords:
                if k in zarr_attrs.keys():
                    raise ValueError("Can not set attribute %s, this is a reserved N5 keyword" % k)

            # replace previous user attributes
            for k in list(n5_attrs.keys()):
                if k not in n5_keywords:
                    del n5_attrs[k]

            # add new user attributes
            n5_attrs.update(**zarr_attrs)

            value = json_dumps(n5_attrs)

        elif is_chunk_key(key):

            key = invert_chunk_coords(key)

        super().__setitem__(key, value)

    def __delitem__(self, key):

        if key.endswith(zarr_group_meta_key):  # pragma: no cover
            key = key.replace(zarr_group_meta_key, n5_attrs_key)
        elif key.endswith(zarr_array_meta_key):  # pragma: no cover
            key = key.replace(zarr_array_meta_key, n5_attrs_key)
        elif key.endswith(zarr_attrs_key):  # pragma: no cover
            key = key.replace(zarr_attrs_key, n5_attrs_key)
        elif is_chunk_key(key):
            key = invert_chunk_coords(key)

        super().__delitem__(key)

    def __contains__(self, key):

        if key.endswith(zarr_group_meta_key):

            key = key.replace(zarr_group_meta_key, n5_attrs_key)
            if key not in self:
                return False
            # group if not a dataset (attributes do not contain 'dimensions')
            return 'dimensions' not in self._load_n5_attrs(key)

        elif key.endswith(zarr_array_meta_key):

            key = key.replace(zarr_array_meta_key, n5_attrs_key)
            # array if attributes contain 'dimensions'
            return 'dimensions' in self._load_n5_attrs(key)

        elif key.endswith(zarr_attrs_key):

            key = key.replace(zarr_attrs_key, n5_attrs_key)
            return self._contains_attrs(key)

        elif is_chunk_key(key):

            key = invert_chunk_coords(key)

        return super().__contains__(key)

    def __eq__(self, other):
        return (
            isinstance(other, N5Store) and
            self.path == other.path
        )

    def listdir(self, path=None):

        if path is not None:
            path = invert_chunk_coords(path)

        # We can't use NestedDirectoryStore's listdir, as it requires
        # array_meta_key to be present in array directories, which this store
        # doesn't provide.
        children = super().listdir(path=path)

        if self._is_array(path):

            # replace n5 attribute file with respective zarr attribute files
            children.remove(n5_attrs_key)
            children.append(zarr_array_meta_key)
            if self._contains_attrs(path):
                children.append(zarr_attrs_key)

            # special handling of directories containing an array to map
            # inverted nested chunk keys back to standard chunk keys
            new_children = []
            root_path = self.dir_path(path)
            for entry in children:
                entry_path = os.path.join(root_path, entry)
                if _prog_number.match(entry) and os.path.isdir(entry_path):
                    for dir_path, _, file_names in os.walk(entry_path):
                        for file_name in file_names:
                            file_path = os.path.join(dir_path, file_name)
                            rel_path = file_path.split(root_path + os.path.sep)[1]
                            new_child = rel_path.replace(os.path.sep, '.')
                            new_children.append(invert_chunk_coords(new_child))
                else:
                    new_children.append(entry)

            return sorted(new_children)

        elif self._is_group(path):

            # replace n5 attribute file with respective zarr attribute files
            children.remove(n5_attrs_key)
            children.append(zarr_group_meta_key)
            if self._contains_attrs(path):  # pragma: no cover
                children.append(zarr_attrs_key)

            return sorted(children)

        else:

            return children

    def _load_n5_attrs(self, path):
        try:
            s = super().__getitem__(path)
            return json_loads(s)
        except KeyError:
            return {}

    def _is_group(self, path):

        if path is None:
            attrs_key = n5_attrs_key
        else:
            attrs_key = os.path.join(path, n5_attrs_key)

        n5_attrs = self._load_n5_attrs(attrs_key)
        return len(n5_attrs) > 0 and 'dimensions' not in n5_attrs

    def _is_array(self, path):

        if path is None:
            attrs_key = n5_attrs_key
        else:
            attrs_key = os.path.join(path, n5_attrs_key)

        return 'dimensions' in self._load_n5_attrs(attrs_key)

    def _contains_attrs(self, path):

        if path is None:
            attrs_key = n5_attrs_key
        else:
            if not path.endswith(n5_attrs_key):
                attrs_key = os.path.join(path, n5_attrs_key)
            else:  # pragma: no cover
                attrs_key = path

        attrs = attrs_to_zarr(self._load_n5_attrs(attrs_key))
        return len(attrs) > 0


def is_chunk_key(key):
    segments = list(key.split('/'))
    if segments:
        last_segment = segments[-1]
        return _prog_ckey.match(last_segment)
    return False  # pragma: no cover


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
    # TODO: This should only exist at the top-level
    group_metadata['n5'] = '2.0.0'
    return group_metadata


def group_metadata_to_zarr(group_metadata):
    '''Convert group metadata from N5 to zarr format.'''
    # This only exists at the top level
    group_metadata.pop('n5', None)
    group_metadata['zarr_format'] = ZARR_FORMAT
    return group_metadata


def array_metadata_to_n5(array_metadata):
    '''Convert array metadata from zarr to N5 format.'''

    for f, t in zarr_to_n5_keys:
        array_metadata[t] = array_metadata[f]
        del array_metadata[f]
    del array_metadata['zarr_format']

    try:
        dtype = np.dtype(array_metadata['dataType'])
    except TypeError:  # pragma: no cover
        raise TypeError(
            "data type %s not supported by N5" % array_metadata['dataType'])

    array_metadata['dataType'] = dtype.name
    array_metadata['dimensions'] = array_metadata['dimensions'][::-1]
    array_metadata['blockSize'] = array_metadata['blockSize'][::-1]

    if 'fill_value' in array_metadata:
        if array_metadata['fill_value'] != 0 and array_metadata['fill_value'] is not None:
            raise ValueError("N5 only supports fill_value == 0 (for now)")
        del array_metadata['fill_value']

    if 'order' in array_metadata:
        if array_metadata['order'] != 'C':
            raise ValueError("zarr N5 storage only stores arrays in C order (for now)")
        del array_metadata['order']

    if 'filters' in array_metadata:
        if array_metadata['filters'] != [] and array_metadata['filters'] is not None:
            raise ValueError("N5 storage does not support zarr filters")
        del array_metadata['filters']

    assert 'compression' in array_metadata
    compressor_config = array_metadata['compression']
    compressor_config = compressor_config_to_n5(compressor_config)
    array_metadata['compression'] = compressor_config

    if 'dimension_separator' in array_metadata:
        del array_metadata['dimension_separator']

    return array_metadata


def array_metadata_to_zarr(array_metadata):
    '''Convert array metadata from N5 to zarr format.'''
    for t, f in zarr_to_n5_keys:
        array_metadata[t] = array_metadata[f]
        del array_metadata[f]
    array_metadata['zarr_format'] = ZARR_FORMAT

    array_metadata['shape'] = array_metadata['shape'][::-1]
    array_metadata['chunks'] = array_metadata['chunks'][::-1]
    array_metadata['fill_value'] = 0  # also if None was requested
    array_metadata['order'] = 'C'
    array_metadata['filters'] = []

    compressor_config = array_metadata['compressor']
    compressor_config = compressor_config_to_zarr(compressor_config)
    array_metadata['compressor'] = {
        'id': N5ChunkWrapper.codec_id,
        'compressor_config': compressor_config,
        'dtype': array_metadata['dtype'],
        'chunk_shape': array_metadata['chunks']
    }

    return array_metadata


def attrs_to_zarr(attrs):
    '''Get all zarr attributes from an N5 attributes dictionary (i.e.,
    all non-keyword attributes).'''

    # remove all N5 keywords
    for n5_key in n5_keywords:
        if n5_key in attrs:
            del attrs[n5_key]

    return attrs


def compressor_config_to_n5(compressor_config):

    if compressor_config is None:
        return {'type': 'raw'}

    # peel wrapper, if present
    if compressor_config['id'] == N5ChunkWrapper.codec_id:
        compressor_config = compressor_config['compressor_config']

    codec_id = compressor_config['id']
    n5_config = {'type': codec_id}

    if codec_id == 'bz2':

        n5_config['type'] = 'bzip2'
        n5_config['blockSize'] = compressor_config['level']

    elif codec_id == 'blosc':

        warnings.warn(
            "Not all N5 implementations support blosc compression (yet). You "
            "might not be able to open the dataset with another N5 library.",
            RuntimeWarning
        )

        n5_config['cname'] = compressor_config['cname']
        n5_config['clevel'] = compressor_config['clevel']
        n5_config['shuffle'] = compressor_config['shuffle']
        n5_config['blocksize'] = compressor_config['blocksize']

    elif codec_id == 'lzma':

        # Switch to XZ for N5 if we are using the default XZ format.
        # Note: 4 is the default, which is lzma.CHECK_CRC64.
        if compressor_config['format'] == 1 and compressor_config['check'] in [-1, 4]:
            n5_config['type'] = 'xz'
        else:
            warnings.warn(
                "Not all N5 implementations support lzma compression (yet). You "
                "might not be able to open the dataset with another N5 library.",
                RuntimeWarning
            )
            n5_config['format'] = compressor_config['format']
            n5_config['check'] = compressor_config['check']
            n5_config['filters'] = compressor_config['filters']

        # The default is lzma.PRESET_DEFAULT, which is 6.
        if compressor_config['preset']:
            n5_config['preset'] = compressor_config['preset']
        else:
            n5_config['preset'] = 6

    elif codec_id == 'zlib':

        n5_config['type'] = 'gzip'
        n5_config['level'] = compressor_config['level']
        n5_config['useZlib'] = True

    elif codec_id == 'gzip':  # pragma: no cover

        n5_config['type'] = 'gzip'
        n5_config['level'] = compressor_config['level']
        n5_config['useZlib'] = False

    else:  # pragma: no cover

        n5_config.update({k: v for k, v in compressor_config.items() if k != 'type'})

    return n5_config


def compressor_config_to_zarr(compressor_config):

    codec_id = compressor_config['type']
    zarr_config = {'id': codec_id}

    if codec_id == 'bzip2':

        zarr_config['id'] = 'bz2'
        zarr_config['level'] = compressor_config['blockSize']

    elif codec_id == 'blosc':

        zarr_config['cname'] = compressor_config['cname']
        zarr_config['clevel'] = compressor_config['clevel']
        zarr_config['shuffle'] = compressor_config['shuffle']
        zarr_config['blocksize'] = compressor_config['blocksize']

    elif codec_id == 'lzma':

        zarr_config['format'] = compressor_config['format']
        zarr_config['check'] = compressor_config['check']
        zarr_config['preset'] = compressor_config['preset']
        zarr_config['filters'] = compressor_config['filters']

    elif codec_id == 'xz':

        zarr_config['id'] = 'lzma'
        zarr_config['format'] = 1  # lzma.FORMAT_XZ
        zarr_config['check'] = -1
        zarr_config['preset'] = compressor_config['preset']
        zarr_config['filters'] = None

    elif codec_id == 'gzip':

        if 'useZlib' in compressor_config and compressor_config['useZlib']:
            zarr_config['id'] = 'zlib'
            zarr_config['level'] = compressor_config['level']
        else:  # pragma: no cover
            zarr_config['id'] = 'gzip'
            zarr_config['level'] = compressor_config['level']

    elif codec_id == 'raw':

        return None

    else:  # pragma: no cover

        zarr_config.update({k: v for k, v in compressor_config.items() if k != 'type'})

    return zarr_config


class N5ChunkWrapper(Codec):

    codec_id = 'n5_wrapper'

    def __init__(self, dtype, chunk_shape, compressor_config=None, compressor=None):

        self.dtype = np.dtype(dtype)
        self.chunk_shape = tuple(chunk_shape)
        # is the dtype a little endian format?
        self._little_endian = (
            self.dtype.byteorder == '<' or
            (self.dtype.byteorder == '=' and sys.byteorder == 'little')
        )

        if compressor:  # pragma: no cover
            if compressor_config is not None:
                raise ValueError("Only one of compressor_config or compressor should be given.")
            compressor_config = compressor.get_config()

        if (
                compressor_config is None and compressor is None or
                compressor_config['id'] == 'raw'):
            self.compressor_config = None
            self._compressor = None
        else:
            self._compressor = get_codec(compressor_config)
            self.compressor_config = self._compressor.get_config()

    def get_config(self):
        config = {
            'id': self.codec_id,
            'compressor_config': self.compressor_config
        }
        return config

    def encode(self, chunk):

        assert chunk.flags.c_contiguous

        header = self._create_header(chunk)
        chunk = self._to_big_endian(chunk)

        if self._compressor:
            return header + self._compressor.encode(chunk)
        else:
            return header + chunk.tobytes(order='A')

    def decode(self, chunk, out=None):

        len_header, chunk_shape = self._read_header(chunk)
        chunk = chunk[len_header:]

        if out is not None:

            # out should only be used if we read a complete chunk
            assert chunk_shape == self.chunk_shape, (
                "Expected chunk of shape {}, found {}".format(
                    self.chunk_shape,
                    chunk_shape))

            if self._compressor:
                self._compressor.decode(chunk, out)
            else:
                ndarray_copy(chunk, out)

            # we can byteswap in-place
            if self._little_endian:
                out.byteswap(True)

            return out

        else:

            if self._compressor:
                chunk = self._compressor.decode(chunk)

            # more expensive byteswap
            chunk = self._from_big_endian(chunk)

            # read partial chunk
            if chunk_shape != self.chunk_shape:  # pragma: no cover
                chunk = np.frombuffer(chunk, dtype=self.dtype)
                chunk = chunk.reshape(chunk_shape)
                complete_chunk = np.zeros(self.chunk_shape, dtype=self.dtype)
                target_slices = tuple(slice(0, s) for s in chunk_shape)
                complete_chunk[target_slices] = chunk
                chunk = complete_chunk

            return chunk

    def _create_header(self, chunk):

        mode = struct.pack('>H', 0)
        num_dims = struct.pack('>H', len(chunk.shape))
        shape = b''.join(
            struct.pack('>I', d)
            for d in chunk.shape[::-1]
        )

        return mode + num_dims + shape

    def _read_header(self, chunk):

        num_dims = struct.unpack('>H', chunk[2:4])[0]
        shape = tuple(
            struct.unpack('>I', chunk[i:i+4])[0]
            for i in range(4, num_dims*4 + 4, 4)
        )[::-1]

        len_header = 4 + num_dims*4

        return len_header, shape

    def _to_big_endian(self, data):
        # assumes data is ndarray

        if self._little_endian:
            return data.byteswap()
        return data

    def _from_big_endian(self, data):
        # assumes data is byte array in big endian

        if not self._little_endian:
            return data

        a = np.frombuffer(data, self.dtype.newbyteorder('>'))
        return a.astype(self.dtype)


register_codec(N5ChunkWrapper, N5ChunkWrapper.codec_id)
