# -*- coding: utf-8 -*-


class MetadataError(Exception):
    pass


class CopyError(RuntimeError):
    pass


class _BaseZarrError(ValueError):
    _msg = ""

    def __init__(self, *args):
        super().__init__(self._msg.format(*args))


class ContainsGroupError(_BaseZarrError):
    _msg = "path {0!r} contains a group"


def err_contains_group(path):
    raise ContainsGroupError(path)  # pragma: no cover


class ContainsArrayError(_BaseZarrError):
    _msg = "path {0!r} contains an array"


def err_contains_array(path):
    raise ContainsArrayError(path)  # pragma: no cover


class ArrayNotFoundError(_BaseZarrError):
    _msg = "array not found at path %r' {0!r}"


def err_array_not_found(path):
    raise ArrayNotFoundError(path)  # pragma: no cover


class GroupNotFoundError(_BaseZarrError):
    _msg = "group not found at path {0!r}"


def err_group_not_found(path):
    raise GroupNotFoundError(path)  # pragma: no cover


class PathNotFoundError(_BaseZarrError):
    _msg = "nothing found at path {0!r}"


def err_path_not_found(path):
    raise PathNotFoundError(path)  # pragma: no cover


def err_bad_compressor(compressor):
    raise ValueError('bad compressor; expected Codec object, found %r' %
                     compressor)


class FSPathExistNotDir(GroupNotFoundError):
    _msg = "path exists but is not a directory: %r"


class ReadOnlyError(PermissionError):
    def __init__(self):
        super().__init__("object is read-only")


def err_read_only():
    raise ReadOnlyError()  # pragma: no cover


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
