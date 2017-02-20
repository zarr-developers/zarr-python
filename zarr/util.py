# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import collections
import numbers
import operator


import numpy as np

from kenjutsu.format import reformat_slices
from kenjutsu.measure import len_slices

from zarr.compat import integer_types, PY2, reduce


def normalize_shape(shape):
    """Convenience function to normalize the `shape` argument."""

    if shape is None:
        raise TypeError('shape is None')

    # handle 1D convenience form
    if isinstance(shape, integer_types):
        shape = (int(shape),)

    # normalize
    shape = tuple(int(s) for s in shape)
    return shape


# code to guess chunk shape, adapted from h5py

CHUNK_BASE = 64*1024  # Multiplier by which chunks are adjusted
CHUNK_MIN = 128*1024  # Soft lower limit (128k)
CHUNK_MAX = 16*1024*1024  # Hard upper limit (16M)


def guess_chunks(shape, typesize):
    """ Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    Undocumented and subject to change without warning.
    """

    ndims = len(shape)
    chunks = np.array(shape, dtype='=f8')

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = np.product(chunks)*typesize
    target_size = CHUNK_BASE * (2**np.log10(dset_size/(1024.*1024)))

    if target_size > CHUNK_MAX:
        target_size = CHUNK_MAX
    elif target_size < CHUNK_MIN:
        target_size = CHUNK_MIN

    idx = 0
    while True:
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        #  2. The chunk is smaller than the maximum chunk size

        chunk_bytes = np.product(chunks)*typesize

        if (chunk_bytes < target_size or
                abs(chunk_bytes-target_size)/target_size < 0.5) and \
                chunk_bytes < CHUNK_MAX:
            break

        if np.product(chunks) == 1:
            break  # Element size larger than CHUNK_MAX

        chunks[idx % ndims] = np.ceil(chunks[idx % ndims] / 2.0)
        idx += 1

    return tuple(int(x) for x in chunks)


def normalize_chunks(chunks, shape, typesize):
    """Convenience function to normalize the `chunks` argument for an array
    with the given `shape`."""

    # N.B., expect shape already normalized

    # handle auto-chunking
    if chunks is None or chunks is True:
        return guess_chunks(shape, typesize)

    # handle 1D convenience form
    if isinstance(chunks, integer_types):
        chunks = (int(chunks),)

    # handle bad dimensionality
    if len(chunks) > len(shape):
        raise ValueError('too many dimensions in chunks')

    # handle underspecified chunks
    if len(chunks) < len(shape):
        # assume chunks across remaining dimensions
        chunks += shape[len(chunks):]

    # handle None in chunks
    chunks = tuple(s if c is None else int(c)
                   for s, c in zip(shape, chunks))

    return chunks


# noinspection PyTypeChecker
def is_total_slice(item, shape):
    """Determine whether `item` specifies a complete slice of array with the
    given `shape`. Used to optimize __setitem__ operations on the Chunk
    class."""

    rf_item = normalize_array_selection(item, shape)

    # Remove any duplicates from sequences.
    rf_item = list(rf_item)
    for i in range(len(rf_item)):
        if isinstance(rf_item[i], collections.Sequence):
            rf_item[i] = list(set(rf_item[i]))
    rf_item = tuple(rf_item)

    return len_slices(rf_item) == shape


def normalize_axis_selection(item, l):
    """Convenience function to normalize a selection within a single axis
    of size `l`."""

    rf_item = reformat_slices((item,), (l,))[0]

    if isinstance(rf_item, slice) and rf_item.step != 1:
        raise NotImplementedError("slice with step not supported")

    if np.prod(len_slices((rf_item,))) == 0:
        raise IndexError(
            "index out of bounds: %s, %s" % (item.start, item.stop)
        )

    return rf_item


# noinspection PyTypeChecker
def normalize_array_selection(item, shape):
    """Convenience function to normalize a selection within an array with
    the given `shape`."""

    rf_item = reformat_slices(item, shape)

    # Only needed for constraint checks.
    rf_item = tuple(
        normalize_axis_selection(i, l) for i, l in zip(rf_item, shape)
    )

    return rf_item


def get_chunk_range(selection, chunks):
    """Convenience function to get a range over all chunk indices,
    for iterating over chunks."""
    chunk_range = [range(s.start//l, int(np.ceil(s.stop/l)))
                   if isinstance(s, slice)
                   else range(s//l, (s//l)+1)
                   for s, l in zip(selection, chunks)]
    return chunk_range


def normalize_resize_args(old_shape, *args):

    # normalize new shape argument
    if len(args) == 1:
        new_shape = args[0]
    else:
        new_shape = args
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    else:
        new_shape = tuple(new_shape)
    if len(new_shape) != len(old_shape):
        raise ValueError('new shape must have same number of dimensions')

    # handle None in new_shape
    new_shape = tuple(s if n is None else int(n)
                      for s, n in zip(old_shape, new_shape))

    return new_shape


def human_readable_size(size):
    if size < 2**10:
        return '%s' % size
    elif size < 2**20:
        return '%.1fK' % (size / float(2**10))
    elif size < 2**30:
        return '%.1fM' % (size / float(2**20))
    elif size < 2**40:
        return '%.1fG' % (size / float(2**30))
    elif size < 2**50:
        return '%.1fT' % (size / float(2**40))
    else:
        return '%.1fP' % (size / float(2**50))


def normalize_order(order):
    order = str(order).upper()
    if order not in ['C', 'F']:
        raise ValueError("order must be either 'C' or 'F', found: %r" % order)
    return order


def normalize_storage_path(path):
    if path:

        # convert backslash to forward slash
        path = path.replace('\\', '/')

        # ensure no leading slash
        while len(path) > 0 and path[0] == '/':
            path = path[1:]

        # ensure no trailing slash
        while len(path) > 0 and path[-1] == '/':
            path = path[:-1]

        # collapse any repeated slashes
        previous_char = None
        collapsed = ''
        for char in path:
            if char == '/' and previous_char == '/':
                pass
            else:
                collapsed += char
            previous_char = char
        path = collapsed

        # don't allow path segments with just '.' or '..'
        segments = path.split('/')
        if any([s in {'.', '..'} for s in segments]):
            raise ValueError("path containing '.' or '..' segment not allowed")

    else:
        path = ''

    return path


def buffer_size(v):
    from array import array as _stdlib_array
    if PY2 and isinstance(v, _stdlib_array):  # pragma: no cover
        # special case array.array because does not support buffer
        # interface in PY2
        return v.buffer_info()[1] * v.itemsize
    else:
        v = memoryview(v)
        return reduce(operator.mul, v.shape) * v.itemsize


def buffer_tobytes(v):
    from array import array as _stdlib_array
    if isinstance(v, np.ndarray):
        return v.tobytes(order='A')
    elif PY2 and isinstance(v, _stdlib_array):  # pragma: no cover
        return v.tostring()
    else:
        return memoryview(v).tobytes()
