"""This module contains a storage class and codec to support the N5 format.
"""
import os
import struct
import sys
from typing import Any, Dict, Optional, cast
import warnings

import numpy as np
from numcodecs.abc import Codec
from numcodecs.compat import ndarray_copy
from numcodecs.registry import get_codec, register_codec

from .meta import ZARR_FORMAT, json_dumps, json_loads
from .storage import FSStore
from .storage import NestedDirectoryStore, _prog_ckey, _prog_number, normalize_storage_path
from .storage import array_meta_key as zarr_array_meta_key
from .storage import attrs_key as zarr_attrs_key
from .storage import group_meta_key as zarr_group_meta_key

N5_FORMAT = '2.0.0'

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
        useful to avoid potential discrepancies between case-sensitive and
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

    def __getitem__(self, key: str) -> bytes:
        if key.endswith(zarr_group_meta_key):

            key_new = key.replace(zarr_group_meta_key, n5_attrs_key)
            value = group_metadata_to_zarr(self._load_n5_attrs(key_new))

            return json_dumps(value)

        elif key.endswith(zarr_array_meta_key):

            key_new = key.replace(zarr_array_meta_key, n5_attrs_key)
            top_level = key == zarr_array_meta_key
            value = array_metadata_to_zarr(self._load_n5_attrs(key_new), top_level=top_level)
            return json_dumps(value)

        elif key.endswith(zarr_attrs_key):

            key_new = key.replace(zarr_attrs_key, n5_attrs_key)
            value = attrs_to_zarr(self._load_n5_attrs(key_new))

            if len(value) == 0:
                raise KeyError(key_new)
            else:
                return json_dumps(value)

        elif is_chunk_key(key):
            key_new = invert_chunk_coords(key)

        else:
            key_new = key

        return super().__getitem__(key_new)

    def __setitem__(self, key: str, value: Any):

        if key.endswith(zarr_group_meta_key):

            key_new = key.replace(zarr_group_meta_key, n5_attrs_key)

            n5_attrs = self._load_n5_attrs(key_new)
            n5_attrs.update(**group_metadata_to_n5(json_loads(value)))

            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_array_meta_key):

            key_new = key.replace(zarr_array_meta_key, n5_attrs_key)
            top_level = key == zarr_array_meta_key
            n5_attrs = self._load_n5_attrs(key_new)
            n5_attrs.update(**array_metadata_to_n5(json_loads(value), top_level=top_level))
            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_attrs_key):

            key_new = key.replace(zarr_attrs_key, n5_attrs_key)

            n5_attrs = self._load_n5_attrs(key_new)
            zarr_attrs = json_loads(value)

            for k in n5_keywords:
                if k in zarr_attrs:
                    warnings.warn(f"Attribute {k} is a reserved N5 keyword", UserWarning)

            # remove previous user attributes
            for k in list(n5_attrs.keys()):
                if k not in n5_keywords:
                    del n5_attrs[k]

            # add new user attributes
            n5_attrs.update(**zarr_attrs)

            value = json_dumps(n5_attrs)

        elif is_chunk_key(key):
            key_new = invert_chunk_coords(key)

        else:
            key_new = key

        super().__setitem__(key_new, value)

    def __delitem__(self, key: str):
        if key.endswith(zarr_group_meta_key):
            key_new = key.replace(zarr_group_meta_key, n5_attrs_key)
        elif key.endswith(zarr_array_meta_key):
            key_new = key.replace(zarr_array_meta_key, n5_attrs_key)
        elif key.endswith(zarr_attrs_key):
            key_new = key.replace(zarr_attrs_key, n5_attrs_key)
        elif is_chunk_key(key):
            key_new = invert_chunk_coords(key)
        else:
            key_new = key

        super().__delitem__(key_new)

    def __contains__(self, key):

        if key.endswith(zarr_group_meta_key):

            key_new = key.replace(zarr_group_meta_key, n5_attrs_key)
            if key_new not in self:
                return False
            # group if not a dataset (attributes do not contain 'dimensions')
            return 'dimensions' not in self._load_n5_attrs(key_new)

        elif key.endswith(zarr_array_meta_key):

            key_new = key.replace(zarr_array_meta_key, n5_attrs_key)
            # array if attributes contain 'dimensions'
            return 'dimensions' in self._load_n5_attrs(key_new)

        elif key.endswith(zarr_attrs_key):

            key_new = key.replace(zarr_attrs_key, n5_attrs_key)
            return self._contains_attrs(key_new)

        elif is_chunk_key(key):

            key_new = invert_chunk_coords(key)
        else:
            key_new = key

        return super().__contains__(key_new)

    def __eq__(self, other):
        return (
            isinstance(other, N5Store) and
            self.path == other.path
        )

    def listdir(self, path: Optional[str] = None):

        if path is not None:
            path = invert_chunk_coords(path)
        path = cast(str, path)
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
            if self._contains_attrs(path):
                children.append(zarr_attrs_key)

            return sorted(children)

        else:

            return children

    def _load_n5_attrs(self, path: str) -> Dict[str, Any]:
        try:
            s = super().__getitem__(path)
            return json_loads(s)
        except KeyError:
            return {}

    def _is_group(self, path: str):

        if path is None:
            attrs_key = n5_attrs_key
        else:
            attrs_key = os.path.join(path, n5_attrs_key)

        n5_attrs = self._load_n5_attrs(attrs_key)
        return len(n5_attrs) > 0 and 'dimensions' not in n5_attrs

    def _is_array(self, path: str):

        if path is None:
            attrs_key = n5_attrs_key
        else:
            attrs_key = os.path.join(path, n5_attrs_key)

        return 'dimensions' in self._load_n5_attrs(attrs_key)

    def _contains_attrs(self, path: str):

        if path is None:
            attrs_key = n5_attrs_key
        else:
            if not path.endswith(n5_attrs_key):
                attrs_key = os.path.join(path, n5_attrs_key)
            else:
                attrs_key = path

        attrs = attrs_to_zarr(self._load_n5_attrs(attrs_key))
        return len(attrs) > 0


class N5FSStore(FSStore):
    """Implementation of the N5 format (https://github.com/saalfeldlab/n5)
    using `fsspec`, which allows storage on a variety of filesystems. Based
    on `zarr.N5Store`.
    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-sensitive and
        case-insensitive file system. Default value is False.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.N5FSStore('data/array.n5', auto_mkdir=True)
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42

    Store a group::

        >>> store = zarr.N5FSStore('data/group.n5', auto_mkdir=True)
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42

    Notes
    -----
    This is an experimental feature.
    Safe to write in multiple threads or processes.

    Be advised that the `_dimension_separator` property of this store
    (and arrays it creates) is ".", but chunks saved by this store will
    in fact be "/" separated, as proscribed by the N5 format.

    This is counter-intuitive (to say the least), but not arbitrary.
    Chunks in N5 format are stored with reversed dimension order
    relative to Zarr chunks: a chunk of a 3D Zarr array would be stored
    on a file system as `/0/1/2`, but in N5 the same chunk would be
    stored as `/2/1/0`. Therefore, stores targeting N5 must intercept
    chunk keys and flip the order of the dimensions before writing to
    storage, and this procedure requires chunk keys with "." separated
    dimensions, hence the Zarr arrays targeting N5 have the deceptive
    "." dimension separator.
    """
    _array_meta_key = 'attributes.json'
    _group_meta_key = 'attributes.json'
    _attrs_key = 'attributes.json'

    def __init__(self, *args, **kwargs):
        if 'dimension_separator' in kwargs:
            kwargs.pop('dimension_separator')
            warnings.warn('Keyword argument `dimension_separator` will be ignored')
        dimension_separator = "."
        super().__init__(*args, dimension_separator=dimension_separator, **kwargs)

    @staticmethod
    def _swap_separator(key: str):
        segments = list(key.split('/'))
        if segments:
            last_segment = segments[-1]
            if _prog_ckey.match(last_segment):
                coords = list(last_segment.split('.'))
                last_segment = '/'.join(coords[::-1])
                segments = segments[:-1] + [last_segment]
                key = '/'.join(segments)
        return key

    def _normalize_key(self, key: str):
        if is_chunk_key(key):
            key = invert_chunk_coords(key)

        key = normalize_storage_path(key).lstrip("/")
        if key:
            *bits, end = key.split("/")

            if end not in (self._array_meta_key, self._group_meta_key, self._attrs_key):
                end = end.replace(".", "/")
                key = "/".join(bits + [end])
        return key.lower() if self.normalize_keys else key

    def __getitem__(self, key: str) -> bytes:
        if key.endswith(zarr_group_meta_key):

            key_new = key.replace(zarr_group_meta_key, self._group_meta_key)
            value = group_metadata_to_zarr(self._load_n5_attrs(key_new))

            return json_dumps(value)

        elif key.endswith(zarr_array_meta_key):

            key_new = key.replace(zarr_array_meta_key, self._array_meta_key)
            top_level = key == zarr_array_meta_key
            value = array_metadata_to_zarr(self._load_n5_attrs(key_new), top_level=top_level)
            return json_dumps(value)

        elif key.endswith(zarr_attrs_key):

            key_new = key.replace(zarr_attrs_key, self._attrs_key)
            value = attrs_to_zarr(self._load_n5_attrs(key_new))

            if len(value) == 0:
                raise KeyError(key_new)
            else:
                return json_dumps(value)

        elif is_chunk_key(key):
            key_new = self._swap_separator(key)

        else:
            key_new = key

        return super().__getitem__(key_new)

    def __setitem__(self, key: str, value: Any):
        if key.endswith(zarr_group_meta_key):

            key_new = key.replace(zarr_group_meta_key, self._group_meta_key)

            n5_attrs = self._load_n5_attrs(key_new)
            n5_attrs.update(**group_metadata_to_n5(json_loads(value)))

            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_array_meta_key):

            key_new = key.replace(zarr_array_meta_key, self._array_meta_key)
            top_level = key == zarr_array_meta_key
            n5_attrs = self._load_n5_attrs(key_new)
            n5_attrs.update(**array_metadata_to_n5(json_loads(value), top_level=top_level))

            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_attrs_key):

            key_new = key.replace(zarr_attrs_key, self._attrs_key)

            n5_attrs = self._load_n5_attrs(key_new)
            zarr_attrs = json_loads(value)

            for k in n5_keywords:
                if k in zarr_attrs.keys():
                    warnings.warn(f"Attribute {k} is a reserved N5 keyword", UserWarning)

            # replace previous user attributes
            for k in list(n5_attrs.keys()):
                if k not in n5_keywords:
                    del n5_attrs[k]

            # add new user attributes
            n5_attrs.update(**zarr_attrs)

            value = json_dumps(n5_attrs)

        elif is_chunk_key(key):
            key_new = self._swap_separator(key)

        else:
            key_new = key

        super().__setitem__(key_new, value)

    def __delitem__(self, key: str):

        if key.endswith(zarr_group_meta_key):
            key_new = key.replace(zarr_group_meta_key, self._group_meta_key)
        elif key.endswith(zarr_array_meta_key):
            key_new = key.replace(zarr_array_meta_key, self._array_meta_key)
        elif key.endswith(zarr_attrs_key):
            key_new = key.replace(zarr_attrs_key, self._attrs_key)
        elif is_chunk_key(key):
            key_new = self._swap_separator(key)
        else:
            key_new = key
        super().__delitem__(key_new)

    def __contains__(self, key: Any):
        if key.endswith(zarr_group_meta_key):

            key_new = key.replace(zarr_group_meta_key, self._group_meta_key)
            if key_new not in self:
                return False
            # group if not a dataset (attributes do not contain 'dimensions')
            return "dimensions" not in self._load_n5_attrs(key_new)

        elif key.endswith(zarr_array_meta_key):

            key_new = key.replace(zarr_array_meta_key, self._array_meta_key)
            # array if attributes contain 'dimensions'
            return "dimensions" in self._load_n5_attrs(key_new)

        elif key.endswith(zarr_attrs_key):

            key_new = key.replace(zarr_attrs_key, self._attrs_key)
            return self._contains_attrs(key_new)

        elif is_chunk_key(key):
            key_new = self._swap_separator(key)

        else:
            key_new = key
        return super().__contains__(key_new)

    def __eq__(self, other: Any):
        return isinstance(other, N5FSStore) and self.path == other.path

    def listdir(self, path: Optional[str] = None):
        if path is not None:
            path = invert_chunk_coords(path)

        # We can't use NestedDirectoryStore's listdir, as it requires
        # array_meta_key to be present in array directories, which this store
        # doesn't provide.
        children = super().listdir(path=path)
        if self._is_array(path):

            # replace n5 attribute file with respective zarr attribute files
            children.remove(self._array_meta_key)
            children.append(zarr_array_meta_key)
            if self._contains_attrs(path):
                children.append(zarr_attrs_key)

            # special handling of directories containing an array to map
            # inverted nested chunk keys back to standard chunk keys
            new_children = []
            root_path = self.dir_path(path)
            for entry in children:
                entry_path = os.path.join(root_path, entry)
                if _prog_number.match(entry) and self.fs.isdir(entry_path):
                    for file_name in self.fs.find(entry_path):
                        file_path = os.path.join(root_path, file_name)
                        rel_path = file_path.split(root_path)[1]
                        new_child = rel_path.lstrip('/').replace('/', ".")
                        new_children.append(invert_chunk_coords(new_child))
                else:
                    new_children.append(entry)
            return sorted(new_children)

        elif self._is_group(path):

            # replace n5 attribute file with respective zarr attribute files
            children.remove(self._group_meta_key)
            children.append(zarr_group_meta_key)
            if self._contains_attrs(path):
                children.append(zarr_attrs_key)
            return sorted(children)
        else:
            return children

    def _load_n5_attrs(self, path: str):
        try:
            s = super().__getitem__(path)
            return json_loads(s)
        except KeyError:
            return {}

    def _is_group(self, path: Optional[str]):

        if path is None:
            attrs_key = self._attrs_key
        else:
            attrs_key = os.path.join(path, self._attrs_key)

        n5_attrs = self._load_n5_attrs(attrs_key)
        return len(n5_attrs) > 0 and "dimensions" not in n5_attrs

    def _is_array(self, path: Optional[str]):

        if path is None:
            attrs_key = self._attrs_key
        else:
            attrs_key = os.path.join(path, self._attrs_key)

        return "dimensions" in self._load_n5_attrs(attrs_key)

    def _contains_attrs(self, path: Optional[str]):

        if path is None:
            attrs_key = self._attrs_key
        else:
            if not path.endswith(self._attrs_key):
                attrs_key = os.path.join(path, self._attrs_key)
            else:
                attrs_key = path

        attrs = attrs_to_zarr(self._load_n5_attrs(attrs_key))
        return len(attrs) > 0


def is_chunk_key(key: str):
    rv = False
    segments = list(key.split('/'))
    if segments:
        last_segment = segments[-1]
        rv = bool(_prog_ckey.match(last_segment))
    return rv


def invert_chunk_coords(key: str):
    segments = list(key.split('/'))
    if segments:
        last_segment = segments[-1]
        if _prog_ckey.match(last_segment):
            coords = list(last_segment.split('.'))
            last_segment = '/'.join(coords[::-1])
            segments = segments[:-1] + [last_segment]
            key = '/'.join(segments)
    return key


def group_metadata_to_n5(group_metadata: Dict[str, Any]) -> Dict[str, Any]:
    '''Convert group metadata from zarr to N5 format.'''
    del group_metadata['zarr_format']
    # TODO: This should only exist at the top-level
    group_metadata['n5'] = N5_FORMAT
    return group_metadata


def group_metadata_to_zarr(group_metadata: Dict[str, Any]) -> Dict[str, Any]:
    '''Convert group metadata from N5 to zarr format.'''
    # This only exists at the top level
    group_metadata.pop('n5', None)
    group_metadata['zarr_format'] = ZARR_FORMAT
    return group_metadata


def array_metadata_to_n5(array_metadata: Dict[str, Any], top_level=False) -> Dict[str, Any]:
    '''Convert array metadata from zarr to N5 format. If the `top_level` keyword argument is True,
    then the `N5` : N5_FORMAT key : value pair will be inserted into the metadata.'''

    for f, t in zarr_to_n5_keys:
        array_metadata[t] = array_metadata.pop(f)
    del array_metadata['zarr_format']
    if top_level:
        array_metadata['n5'] = N5_FORMAT
    try:
        dtype = np.dtype(array_metadata['dataType'])
    except TypeError:
        raise TypeError(
            f"Data type {array_metadata['dataType']} is not supported by N5")

    array_metadata['dataType'] = dtype.name
    array_metadata['dimensions'] = array_metadata['dimensions'][::-1]
    array_metadata['blockSize'] = array_metadata['blockSize'][::-1]

    if 'fill_value' in array_metadata:
        if array_metadata['fill_value'] != 0 and array_metadata['fill_value'] is not None:
            raise ValueError(
                f'''Received fill_value = {array_metadata['fill_value']},
                but N5 only supports fill_value = 0'''
                )
        del array_metadata['fill_value']

    if 'order' in array_metadata:
        if array_metadata['order'] != 'C':
            raise ValueError(
                f"Received order = {array_metadata['order']}, but N5 only supports order = C"
                )
        del array_metadata['order']

    if 'filters' in array_metadata:
        if array_metadata['filters'] != [] and array_metadata['filters'] is not None:
            raise ValueError(
                "Received filters, but N5 storage does not support zarr filters"
                )
        del array_metadata['filters']

    assert 'compression' in array_metadata
    compressor_config = array_metadata['compression']
    compressor_config = compressor_config_to_n5(compressor_config)
    array_metadata['compression'] = compressor_config

    if 'dimension_separator' in array_metadata:
        del array_metadata['dimension_separator']

    return array_metadata


def array_metadata_to_zarr(array_metadata: Dict[str, Any],
                           top_level: bool = False) -> Dict[str, Any]:
    '''Convert array metadata from N5 to zarr format.
    If the `top_level` keyword argument is True, then the `N5` key will be removed from metadata'''
    for t, f in zarr_to_n5_keys:
        array_metadata[t] = array_metadata.pop(f)
    if top_level:
        array_metadata.pop('n5')
    array_metadata['zarr_format'] = ZARR_FORMAT

    array_metadata['shape'] = array_metadata['shape'][::-1]
    array_metadata['chunks'] = array_metadata['chunks'][::-1]
    array_metadata['fill_value'] = 0  # also if None was requested
    array_metadata['order'] = 'C'
    array_metadata['filters'] = []
    array_metadata['dimension_separator'] = '.'

    compressor_config = array_metadata['compressor']
    compressor_config = compressor_config_to_zarr(compressor_config)
    array_metadata['compressor'] = {
        'id': N5ChunkWrapper.codec_id,
        'compressor_config': compressor_config,
        'dtype': array_metadata['dtype'],
        'chunk_shape': array_metadata['chunks']
    }

    return array_metadata


def attrs_to_zarr(attrs: Dict[str, Any]) -> Dict[str, Any]:
    '''Get all zarr attributes from an N5 attributes dictionary (i.e.,
    all non-keyword attributes).'''

    # remove all N5 keywords
    for n5_key in n5_keywords:
        if n5_key in attrs:
            del attrs[n5_key]

    return attrs


def compressor_config_to_n5(compressor_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    if compressor_config is None:
        return {'type': 'raw'}
    else:
        _compressor_config = compressor_config

    # peel wrapper, if present
    if _compressor_config['id'] == N5ChunkWrapper.codec_id:
        _compressor_config = _compressor_config['compressor_config']

    codec_id = _compressor_config['id']
    n5_config = {'type': codec_id}

    if codec_id == 'bz2':

        n5_config['type'] = 'bzip2'
        n5_config['blockSize'] = _compressor_config['level']

    elif codec_id == 'blosc':

        warnings.warn(
            "Not all N5 implementations support blosc compression (yet). You "
            "might not be able to open the dataset with another N5 library.",
            RuntimeWarning
        )

        n5_config['cname'] = _compressor_config['cname']
        n5_config['clevel'] = _compressor_config['clevel']
        n5_config['shuffle'] = _compressor_config['shuffle']
        n5_config['blocksize'] = _compressor_config['blocksize']

    elif codec_id == 'lzma':

        # Switch to XZ for N5 if we are using the default XZ format.
        # Note: 4 is the default, which is lzma.CHECK_CRC64.
        if _compressor_config['format'] == 1 and _compressor_config['check'] in [-1, 4]:
            n5_config['type'] = 'xz'
        else:
            warnings.warn(
                "Not all N5 implementations support lzma compression (yet). You "
                "might not be able to open the dataset with another N5 library.",
                RuntimeWarning
            )
            n5_config['format'] = _compressor_config['format']
            n5_config['check'] = _compressor_config['check']
            n5_config['filters'] = _compressor_config['filters']

        # The default is lzma.PRESET_DEFAULT, which is 6.
        if _compressor_config['preset']:
            n5_config['preset'] = _compressor_config['preset']
        else:
            n5_config['preset'] = 6

    elif codec_id == 'zlib':

        n5_config['type'] = 'gzip'
        n5_config['level'] = _compressor_config['level']
        n5_config['useZlib'] = True

    elif codec_id == 'gzip':

        n5_config['type'] = 'gzip'
        n5_config['level'] = _compressor_config['level']
        n5_config['useZlib'] = False

    else:

        n5_config.update({k: v for k, v in _compressor_config.items() if k != 'type'})

    return n5_config


def compressor_config_to_zarr(compressor_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:

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
        else:
            zarr_config['id'] = 'gzip'
            zarr_config['level'] = compressor_config['level']

    elif codec_id == 'raw':

        return None

    else:

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

        if compressor:
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

    def decode(self, chunk, out=None) -> bytes:

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
            if chunk_shape != self.chunk_shape:
                chunk = np.frombuffer(chunk, dtype=self.dtype)
                chunk = chunk.reshape(chunk_shape)
                complete_chunk = np.zeros(self.chunk_shape, dtype=self.dtype)
                target_slices = tuple(slice(0, s) for s in chunk_shape)
                complete_chunk[target_slices] = chunk
                chunk = complete_chunk

            return chunk

    @staticmethod
    def _create_header(chunk):

        mode = struct.pack('>H', 0)
        num_dims = struct.pack('>H', len(chunk.shape))
        shape = b''.join(
            struct.pack('>I', d)
            for d in chunk.shape[::-1]
        )

        return mode + num_dims + shape

    @staticmethod
    def _read_header(chunk):

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
