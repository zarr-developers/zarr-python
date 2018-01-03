# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from zarr.compat import PermissionError


class MetadataError(Exception):
    pass


class CopyError(RuntimeError):
    pass


def err_contains_group(path):
    raise ValueError('path %r contains a group' % path)


def err_contains_array(path):
    raise ValueError('path %r contains an array' % path)


def err_array_not_found(path):
    raise ValueError('array not found at path %r' % path)


def err_group_not_found(path):
    raise ValueError('group not found at path %r' % path)


def err_path_not_found(path):
    raise ValueError('nothing found at path %r' % path)


def err_bad_compressor(compressor):
    raise ValueError('bad compressor; expected Codec object, found %r' %
                     compressor)


def err_fspath_exists_notdir(fspath):
    raise ValueError('path exists but is not a directory: %r' % fspath)


def err_read_only():
    raise PermissionError('object is read-only')


def err_boundscheck(dim_len):
    raise IndexError('index out of bounds for dimension with length {}'
                     .format(dim_len))


def err_negative_step():
    raise IndexError('only slices with step >= 1 are supported')


def err_too_many_indices(selection, shape):
    raise IndexError('too many indices for array; expected {}, got {}'
                     .format(len(shape), len(selection)))


def err_vindex_invalid_selection(selection):
    raise IndexError('unsupported selection type for vectorized indexing; only '
                     'coordinate selection (tuple of integer arrays) and mask selection '
                     '(single Boolean array) are supported; got {!r}'.format(selection))
