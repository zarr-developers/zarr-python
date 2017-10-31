# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import operator
from textwrap import TextWrapper
import numbers
import functools


import numpy as np


from zarr.compat import PY2, reduce


def normalize_shape(shape):
    """Convenience function to normalize the `shape` argument."""

    if shape is None:
        raise TypeError('shape is None')

    # handle 1D convenience form
    if isinstance(shape, numbers.Integral):
        shape = (int(shape),)

    # normalize
    shape = tuple(int(s) for s in shape)
    return shape


# code to guess chunk shape, adapted from h5py

CHUNK_BASE = 64*1024  # Multiplier by which chunks are adjusted
CHUNK_MIN = 128*1024  # Soft lower limit (128k)
CHUNK_MAX = 16*1024*1024  # Hard upper limit (16M)


def guess_chunks(shape, typesize):
    """
    Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    Undocumented and subject to change without warning.
    """

    ndims = len(shape)
    # require chunks to have non-zero length for all dimensions
    chunks = np.maximum(np.array(shape, dtype='=f8'), 1)

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
        # 2. The chunk is smaller than the maximum chunk size

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
    if isinstance(chunks, numbers.Integral):
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

    # N.B., assume shape is normalized

    if item == Ellipsis:
        return True
    if item == slice(None):
        return True
    if isinstance(item, slice):
        item = item,
    if isinstance(item, tuple):
        return all(
            (isinstance(s, slice) and
                ((s == slice(None)) or
                 ((s.stop - s.start == l) and (s.step in [1, None]))))
            for s, l in zip(item, shape)
        )
    else:
        raise TypeError('expected slice or tuple of slices, found %r' % item)


class BoolArraySelection(object):

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # check number of dimensions, only support indexing with 1d array
        if len(dim_sel.shape) > 1:
            raise IndexError('can only index with 1-dimensional Boolean array')

        # check shape
        if dim_sel.shape[0] != dim_len:
            raise IndexError('Boolean array has wrong length; expected %s, found %s' %
                             (dim_len, dim_sel.shape[0]))

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nchunks = int(np.ceil(self.dim_len / self.dim_chunk_len))

        # precompute number of selected items for each chunk
        self.chunk_nitems = np.zeros(self.nchunks, dtype='i8')
        for dim_chunk_idx in range(self.nchunks):
            dim_chunk_offset = dim_chunk_idx * self.dim_chunk_len
            self.chunk_nitems[dim_chunk_idx] = np.count_nonzero(
                self.dim_sel[dim_chunk_offset:dim_chunk_offset + self.dim_chunk_len]
            )
        self.chunk_nitems_cumsum = np.cumsum(self.chunk_nitems)
        self.nitems = self.chunk_nitems_cumsum[-1]

    def get_chunk_sel(self, dim_chunk_idx):
        dim_chunk_offset = dim_chunk_idx * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel[dim_chunk_offset:dim_chunk_offset + self.dim_chunk_len]
        # pad out if final chunk
        if dim_chunk_sel.shape[0] < self.dim_chunk_len:
            tmp = np.zeros(self.dim_chunk_len, dtype=bool)
            tmp[:dim_chunk_sel.shape[0]] = dim_chunk_sel
            dim_chunk_sel = tmp
        return dim_chunk_sel

    def get_out_sel(self, dim_chunk_idx):
        if dim_chunk_idx == 0:
            start = 0
        else:
            start = self.chunk_nitems_cumsum[dim_chunk_idx - 1]
        stop = self.chunk_nitems_cumsum[dim_chunk_idx]
        return slice(start, stop)

    def get_chunk_ranges(self):
        return np.nonzero(self.chunk_nitems)[0]


class IntArraySelection(object):

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # has to be a numpy array so we can do bincount
        dim_sel = np.asanyarray(dim_sel)

        # check number of dimensions, only support indexing with 1d array
        if len(dim_sel.shape) > 1:
            raise IndexError('can only index with 1-dimensional integer array')

        # handle wraparound
        loc_neg = dim_sel < 0
        if np.any(loc_neg):
            dim_sel[loc_neg] = dim_sel[loc_neg] + dim_len

        # handle out of bounds
        if np.any(dim_sel < 0) or np.any(dim_sel >= dim_len):
            raise IndexError('index out of bounds')

        # validate monotonically increasing
        if np.any(np.diff(dim_sel) < 0):
            raise NotImplementedError('only monotonically increasing indices are supported')

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nchunks = int(np.ceil(self.dim_len / self.dim_chunk_len))

        # precompute number of selected items for each chunk
        # note: for dense integer selections, the division operation here is the bottleneck
        self.chunk_nitems = np.bincount(self.dim_sel // self.dim_chunk_len, minlength=self.nchunks)
        self.chunk_nitems_cumsum = np.cumsum(self.chunk_nitems)
        self.nitems = len(dim_sel)

    def get_chunk_sel(self, dim_chunk_idx):
        # need to slice out relevant indices from the total selection, then subtract the chunk
        # offset

        dim_out_sel = self.get_out_sel(dim_chunk_idx)
        dim_chunk_offset = dim_chunk_idx * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel[dim_out_sel] - dim_chunk_offset

        return dim_chunk_sel

    def get_out_sel(self, dim_chunk_idx):
        if dim_chunk_idx == 0:
            start = 0
        else:
            start = self.chunk_nitems_cumsum[dim_chunk_idx - 1]
        stop = self.chunk_nitems_cumsum[dim_chunk_idx]
        return slice(start, stop)

    def get_chunk_ranges(self):
        return np.nonzero(self.chunk_nitems)[0]


# TODO support slice with step via integer selection (convert to np.arange)


def normalize_dim_selection(dim_sel, dim_len, dim_chunk_len):
    """Convenience function to normalize a selection within a single axis
    of size `dim_len` for an array with chunk length `dim_chunk_len`."""

    # normalize list to array
    if isinstance(dim_sel, list):
        dim_sel = np.asarray(dim_sel)

    if isinstance(dim_sel, numbers.Integral):

        # normalize type to int
        dim_sel = int(dim_sel)

        # handle wraparound
        if dim_sel < 0:
            dim_sel = dim_len + dim_sel

        # handle out of bounds
        if dim_sel >= dim_len or dim_sel < 0:
            raise IndexError('index out of bounds: %s' % dim_sel)

        return dim_sel

    elif isinstance(dim_sel, slice):

        # handle slice with step
        if dim_sel.step is not None and dim_sel.step != 1:
            raise NotImplementedError('slice with step not implemented')

        # handle slice with None bound
        start = 0 if dim_sel.start is None else dim_sel.start
        stop = dim_len if dim_sel.stop is None else dim_sel.stop

        # handle wraparound
        if start < 0:
            start = dim_len + start
        if stop < 0:
            stop = dim_len + stop

        # handle zero-length axis
        if start == stop == dim_len == 0:
            return slice(0, 0)

        # handle out of bounds
        if start < 0:
            raise IndexError('start index out of bounds: %s' % dim_sel.start)
        if stop < 0:
            raise IndexError('stop index out of bounds: %s' % dim_sel.stop)
        if start >= dim_len:
            raise IndexError('start index out of bounds: %ss' % dim_sel.start)
        if stop > dim_len:
            stop = dim_len
        if stop < start:
            stop = start

        return slice(start, stop)

    elif hasattr(dim_sel, 'dtype') and hasattr(dim_sel, 'shape'):

        if dim_sel.dtype == bool:
            return BoolArraySelection(dim_sel, dim_len, dim_chunk_len)

        elif dim_sel.dtype.kind in 'ui':
            return IntArraySelection(dim_sel, dim_len, dim_chunk_len)

        else:
            raise IndexError('unsupported index item type: %r' % dim_sel)

    else:
        raise IndexError('unsupported index item type: %r' % dim_sel)


# noinspection PyTypeChecker
def normalize_array_selection(selection, shape, chunks):
    """Convenience function to normalize a selection within an array with
    the given `shape`."""

    # ensure tuple
    if not isinstance(selection, tuple):
        selection = (selection,)

    # handle ellipsis
    n_ellipsis = sum(1 for i in selection if i is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif n_ellipsis == 1:
        n_items_l = selection.index(Ellipsis)  # items to left of ellipsis
        n_items_r = len(selection) - (n_items_l + 1)  # items to right of ellipsis
        n_items = len(selection) - 1  # all non-ellipsis items
        if n_items >= len(shape):
            # ellipsis does nothing, just remove it
            selection = tuple(i for i in selection if i != Ellipsis)
        else:
            # replace ellipsis with as many slices are needed for number of dims
            new_item = selection[:n_items_l] + ((slice(None),) * (len(shape) - n_items))
            if n_items_r:
                new_item += selection[-n_items_r:]
            selection = new_item

    # check dimensionality
    if len(selection) > len(shape):
        raise IndexError('too many indices for array')

    # determine start and stop indices for all axes
    selection = tuple(normalize_dim_selection(i, l, c) for i, l, c in zip(selection, shape, chunks))

    # fill out selection if not completely specified
    if len(selection) < len(shape):
        selection += tuple(slice(0, l) for l in shape[len(selection):])

    return selection


def get_chunks_for_selection(selection, chunks):
    """Convenience function to find chunks overlapping an array selection. N.B.,
    assumes selection has already been normalized."""

    # indices of chunks overlapping the selection
    chunk_ranges = []

    # shape of the selection
    sel_shape = []

    # iterate over dimensions of the array
    for dim_sel, dim_chunk_len in zip(selection, chunks):

        # dim_sel: selection for current dimension
        # dim_chunk_len: length of chunk along current dimension

        dim_sel_len = None

        if isinstance(dim_sel, int):

            # dim selection is an integer, i.e., single item, so only need single chunk index for
            # this dimension
            dim_chunk_range = [dim_sel//dim_chunk_len]

        elif isinstance(dim_sel, slice):

            # dim selection is a slice, need range of chunk indices including start and stop of
            # selection
            dim_chunk_from = dim_sel.start//dim_chunk_len
            dim_chunk_to = int(np.ceil(dim_sel.stop/dim_chunk_len))
            dim_chunk_range = range(dim_chunk_from, dim_chunk_to)
            dim_sel_len = dim_sel.stop - dim_sel.start

        elif isinstance(dim_sel, BoolArraySelection):

            # dim selection is a boolean array, delegate this to the BooleanSelection class
            dim_chunk_range = dim_sel.get_chunk_ranges()
            dim_sel_len = dim_sel.nitems

        elif isinstance(dim_sel, IntArraySelection):

            # dim selection is an integer array, delegate this to the integerSelection class
            dim_chunk_range = dim_sel.get_chunk_ranges()
            dim_sel_len = dim_sel.nitems

        else:
            raise RuntimeError('unexpected selection type')

        chunk_ranges.append(dim_chunk_range)
        if dim_sel_len is not None:
            sel_shape.append(dim_sel_len)

    return chunk_ranges, tuple(sel_shape)


def get_chunk_selections(selection, chunk_coords, chunks, n_advanced_selection):

    # chunk_coords: holds the index along each dimension for the current chunk within the
    # chunk grid. E.g., (0, 0) locates the first (top left) chunk in a 2D chunk grid.

    chunk_selection = []
    out_selection = []

    # iterate over dimensions (axes) of the array
    for dim_sel, dim_chunk_idx, dim_chunk_len in zip(selection, chunk_coords, chunks):

        # dim_sel: selection for current dimension
        # dim_chunk_idx: chunk index along current dimension
        # dim_chunk_len: chunk length along current dimension

        # selection for current chunk along current dimension
        dim_chunk_sel = None

        # selection into output array to store data from current chunk
        dim_out_sel = None

        # calculate offset for current chunk along current dimension - this is used to
        # determine the values to be extracted from the current chunk
        dim_chunk_offset = dim_chunk_idx * dim_chunk_len

        # handle integer selection, i.e., single item
        if isinstance(dim_sel, int):

            dim_chunk_sel = dim_sel - dim_chunk_offset

            # N.B., leave dim_out_sel as None, as this dimension has been dropped in the
            # output array because of single value index

        # handle slice selection, i.e., contiguous range of items
        elif isinstance(dim_sel, slice):

            if dim_sel.start <= dim_chunk_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                dim_out_offset = dim_chunk_offset - dim_sel.start

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = dim_sel.start - dim_chunk_offset
                dim_out_offset = 0

            if dim_sel.stop > dim_chunk_offset + dim_chunk_len:
                # selection ends after current chunk
                dim_chunk_sel_stop = dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = dim_sel.stop - dim_chunk_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop)
            dim_chunk_nitems = dim_chunk_sel_stop - dim_chunk_sel_start
            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

        elif isinstance(dim_sel, (BoolArraySelection, IntArraySelection)):

            # get selection to extract data for the current chunk
            dim_chunk_sel = dim_sel.get_chunk_sel(dim_chunk_idx)

            # figure out where to put these items in the output array
            dim_out_sel = dim_sel.get_out_sel(dim_chunk_idx)

        else:
            raise RuntimeError('unexpected selection type')

        # add to chunk selection
        chunk_selection.append(dim_chunk_sel)

        # add to output selection
        if dim_out_sel is not None:
            out_selection.append(dim_out_sel)

    # normalise for indexing into numpy arrays
    chunk_selection = tuple(chunk_selection)
    out_selection = tuple(out_selection)

    # handle advanced indexing arrays orthogonally
    if n_advanced_selection > 1:
        # numpy doesn't support orthogonal indexing directly as yet, so need to work
        # around via np.ix_. Also np.ix_ does not support a mixture of arrays and slices
        # or integers, so need to convert slices and integers into ranges.
        chunk_selection = [range(dim_chunk_sel.start, dim_chunk_sel.stop)
                           if isinstance(dim_chunk_sel, slice)
                           else [dim_chunk_sel] if isinstance(dim_chunk_sel, int)
                           else dim_chunk_sel
                           for dim_chunk_sel in chunk_selection]
        chunk_selection = np.ix_(*chunk_selection)

    return chunk_selection, out_selection


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

    # handle bytes
    if not PY2 and isinstance(path, bytes):  # pragma: py2 no cover
        path = str(path, 'ascii')

    # ensure str
    if path is not None and not isinstance(path, str):
        path = str(path)

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
    if PY2 and isinstance(v, _stdlib_array):  # pragma: py3 no cover
        # special case array.array because does not support buffer
        # interface in PY2
        return v.buffer_info()[1] * v.itemsize
    else:  # pragma: py2 no cover
        v = memoryview(v)
        return reduce(operator.mul, v.shape) * v.itemsize


def info_text_report(items):
    keys = [k for k, v in items]
    max_key_len = max(len(k) for k in keys)
    report = ''
    for k, v in items:
        wrapper = TextWrapper(width=80,
                              initial_indent=k.ljust(max_key_len) + ' : ',
                              subsequent_indent=' '*max_key_len + ' : ')
        text = wrapper.fill(str(v))
        report += text + '\n'
    return report


def info_html_report(items):
    report = '<table class="zarr-info">'
    report += '<tbody>'
    for k, v in items:
        report += '<tr>' \
                  '<th style="text-align: left">%s</th>' \
                  '<td style="text-align: left">%s</td>' \
                  '</tr>' \
                  % (k, v)
    report += '</tbody>'
    report += '</table>'
    return report


class InfoReporter(object):

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        items = self.obj.info_items()
        return info_text_report(items)

    def _repr_html_(self):
        items = self.obj.info_items()
        return info_html_report(items)
