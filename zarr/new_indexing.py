# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import numbers
import itertools


import numpy as np


def normalize_integer_selection(dim_sel, dim_len):

    # normalize type to int
    dim_sel = int(dim_sel)

    # handle wraparound
    if dim_sel < 0:
        dim_sel = dim_len + dim_sel

    # handle out of bounds
    if dim_sel >= dim_len or dim_sel < 0:
        raise IndexError('index out of bounds')

    return dim_sel


class IntSelection(object):

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # check type
        if not isinstance(dim_sel, numbers.Integral):
            raise ValueError('selection must be an integer')

        # normalize
        dim_sel = normalize_integer_selection(dim_sel, dim_len)

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = 1

    def get_overlapping_chunks(self):

        dim_chunk_ix = self.dim_sel // self.dim_chunk_len
        dim_offset = dim_chunk_ix * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel - dim_offset
        dim_out_sel = None
        yield dim_chunk_ix, dim_chunk_sel, dim_out_sel


def normalize_slice_selection(dim_sel, dim_len):

    # handle slice with None bound
    start = 0 if dim_sel.start is None else dim_sel.start
    stop = dim_len if dim_sel.stop is None else dim_sel.stop
    step = 1 if dim_sel.step is None else dim_sel.step

    # handle wraparound
    if start < 0:
        start = dim_len + start
    if stop < 0:
        stop = dim_len + stop

    # handle out of bounds
    if start < 0:
        raise IndexError('start index out of bounds: %s' % dim_sel.start)
    if stop < 0:
        raise IndexError('stop index out of bounds: %s' % dim_sel.stop)
    if start >= dim_len and dim_len > 0:
        raise IndexError('start index out of bounds: %ss' % dim_sel.start)
    if stop > dim_len:
        stop = dim_len
    if stop < start:
        stop = start

    return slice(start, stop, step)


class SliceSelection(object):

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # check type
        if not isinstance(dim_sel, slice):
            raise ValueError('selection must be a slice')

        # normalize
        dim_sel = normalize_slice_selection(dim_sel, dim_len)

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = dim_sel.stop - dim_sel.start

    def get_overlapping_chunks(self):

        dim_chunk_from = self.dim_sel.start // self.dim_chunk_len
        dim_chunk_to = int(np.ceil(self.dim_sel.stop / self.dim_chunk_len))

        for dim_chunk_ix in range(dim_chunk_from, dim_chunk_to):

            dim_offset = dim_chunk_ix * self.dim_chunk_len

            if self.dim_sel.start <= dim_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                dim_out_offset = dim_offset - self.dim_sel.start

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = self.dim_sel.start - dim_offset
                dim_out_offset = 0

            if self.dim_sel.stop > (dim_offset + self.dim_chunk_len):
                # selection ends after current chunk
                dim_chunk_sel_stop = self.dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = self.dim_sel.stop - dim_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop)
            dim_chunk_nitems = dim_chunk_sel_stop - dim_chunk_sel_start
            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

            yield dim_chunk_ix, dim_chunk_sel, dim_out_sel


class BoolArrayDimSelection(object):

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # check number of dimensions
        if len(dim_sel.shape) > 1:
            raise IndexError('selection must be a 1d array')

        # check shape
        if dim_sel.shape[0] != dim_len:
            raise IndexError('selection has the wrong length')

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nchunks = int(np.ceil(self.dim_len / self.dim_chunk_len))

        # precompute number of selected items for each chunk
        self.chunk_nitems = np.zeros(self.nchunks, dtype='i8')
        for dim_chunk_idx in range(self.nchunks):
            dim_offset = dim_chunk_idx * self.dim_chunk_len
            self.chunk_nitems[dim_chunk_idx] = np.count_nonzero(
                self.dim_sel[dim_offset:dim_offset + self.dim_chunk_len]
            )
        self.chunk_nitems_cumsum = np.cumsum(self.chunk_nitems)
        self.nitems = self.chunk_nitems_cumsum[-1]

    def get_overlapping_chunks(self):

        # iterate over chunks with at least one item
        for dim_chunk_ix in np.nonzero(self.chunk_nitems)[0]:

            # find region in chunk
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_chunk_sel = self.dim_sel[dim_offset:dim_offset + self.dim_chunk_len]

            # pad out if final chunk
            if dim_chunk_sel.shape[0] < self.dim_chunk_len:
                tmp = np.zeros(self.dim_chunk_len, dtype=bool)
                tmp[:dim_chunk_sel.shape[0]] = dim_chunk_sel
                dim_chunk_sel = tmp

            # find region in output
            if dim_chunk_ix == 0:
                start = 0
            else:
                start = self.chunk_nitems_cumsum[dim_chunk_ix - 1]
            stop = self.chunk_nitems_cumsum[dim_chunk_ix]
            dim_out_sel = slice(start, stop)

            yield dim_chunk_ix, dim_chunk_sel, dim_out_sel


def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


class IntArrayDimSelection(object):
    """Integer array selection against a single dimension."""

    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # ensure array
        dim_sel = np.asanyarray(dim_sel)

        # check number of dimensions
        if dim_sel.ndim != 1:
            raise IndexError('selection must be a 1d array')

        # check dtype
        if dim_sel.dtype.kind not in 'ui':
            raise IndexError('selection must be an integer array')

        # handle wraparound
        loc_neg = dim_sel < 0
        if np.any(loc_neg):
            dim_sel[loc_neg] = dim_sel[loc_neg] + dim_len

        # handle out of bounds
        if np.any(dim_sel < 0) or np.any(dim_sel >= dim_len):
            raise IndexError('selection contains index out of bounds')

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nchunks = int(np.ceil(self.dim_len / self.dim_chunk_len))
        self.nitems = len(dim_sel)

    def get_overlapping_chunks(self):

        # locate required chunk for each index
        dim_chunk_sel = self.dim_sel // self.dim_chunk_len

        # find runs of indices in the same chunk
        dim_chunk_ixs, run_starts, run_lengths = find_runs(dim_chunk_sel)

        # iterate over chunks
        for dim_chunk_ix, s, l in zip(dim_chunk_ixs, run_starts, run_lengths):

            # find region in output array
            dim_out_sel = slice(s, s + l)

            # find region in chunk
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_chunk_sel = self.dim_sel[dim_out_sel] - dim_offset

            yield dim_chunk_ix, dim_chunk_sel, dim_out_sel


def replace_ellipsis(selection, shape):

    # count number of ellipsis present
    n_ellipsis = sum(1 for i in selection if i is Ellipsis)

    if n_ellipsis > 1:
        # more than 1 is an error
        raise IndexError("an index can only have a single ellipsis ('...')")

    elif n_ellipsis == 1:
        # locate the ellipsis, count how many items to left and right
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

    # fill out selection if not completely specified
    if len(selection) < len(shape):
        selection += tuple(slice(0, l) for l in shape[len(selection):])

    return selection


class BasicIndexer(object):

    def __init__(self, selection, array):

        # ensure tuple
        if not isinstance(selection, tuple):
            selection = (selection,)

        # handle ellipsis
        selection = replace_ellipsis(selection, array._shape)

        # validation - check dimensionality
        if len(selection) > len(array._shape):
            raise IndexError('too many indices for array')
        if len(selection) < len(array._shape):
            raise IndexError('not enough indices for array')

        # TODO normalize and setup selections

    def get_overlapping_chunks(self):
        dim_tasks = [s.get_overlapping_chunks() for s in self.selection]
        return itertools.product(*dim_tasks)
