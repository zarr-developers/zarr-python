# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


from zarr.compat import PY2


if PY2:  # pragma: no cover

    class PermissionError(Exception):
        pass

else:

    PermissionError = PermissionError


class MetadataError(Exception):
    pass


def err_contains_group(path):
    raise KeyError('path %r contains a group' % path)


def err_contains_array(path):
    raise KeyError('path %r contains an array' % path)


def err_array_not_found(path):
    raise KeyError('array not found at path %r' % path)


def err_group_not_found(path):
    raise KeyError('group not found at path %r' % path)


def err_path_not_found(path):
    raise KeyError('path %r not found' % path)


def err_bad_compressor(compressor):
    raise ValueError('bad compressor; expected Codec object, found %r' %
                     compressor)


def err_fspath_exists_notdir(fspath):
    raise ValueError('path exists but is not a directory: %r' % fspath)


def err_read_only():
    raise PermissionError('object is read-only')
