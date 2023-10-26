import collections
import itertools
import math
import numbers

import numpy as np


from zarr.errors import (
    ArrayIndexError,
    NegativeStepError,
    err_too_many_indices,
    VindexInvalidSelectionError,
    BoundsCheckError,
)


def is_integer(x):
    """True if x is an integer (both pure Python or NumPy).

    Note that Python's bool is considered an integer too.
    """
    return isinstance(x, numbers.Integral)


def is_integer_list(x):
    """True if x is a list of integers.

    This function assumes ie *does not check* that all elements of the list
    have the same type. Mixed type lists will result in other errors that will
    bubble up anyway.
    """
    return isinstance(x, list) and len(x) > 0 and is_integer(x[0])


def is_integer_array(x, ndim=None):
    t = not np.isscalar(x) and hasattr(x, "shape") and hasattr(x, "dtype") and x.dtype.kind in "ui"
    if ndim is not None:
        t = t and len(x.shape) == ndim
    return t


def is_bool_array(x, ndim=None):
    t = hasattr(x, "shape") and hasattr(x, "dtype") and x.dtype == bool
    if ndim is not None:
        t = t and len(x.shape) == ndim
    return t


def is_scalar(value, dtype):
    if np.isscalar(value):
        return True
    if isinstance(value, tuple) and dtype.names and len(value) == len(dtype.names):
        return True
    return False


def is_pure_fancy_indexing(selection, ndim):
    """Check whether a selection contains only scalars or integer array-likes.

    Parameters
    ----------
    selection : tuple, slice, or scalar
        A valid selection value for indexing into arrays.

    Returns
    -------
    is_pure : bool
        True if the selection is a pure fancy indexing expression (ie not mixed
        with boolean or slices).
    """
    if ndim == 1:
        if is_integer_list(selection) or is_integer_array(selection):
            return True
        # if not, we go through the normal path below, because a 1-tuple
        # of integers is also allowed.
    no_slicing = (
        isinstance(selection, tuple)
        and len(selection) == ndim
        and not (any(isinstance(elem, slice) or elem is Ellipsis for elem in selection))
    )
    return (
        no_slicing
        and all(
            is_integer(elem) or is_integer_list(elem) or is_integer_array(elem)
            for elem in selection
        )
        and any(is_integer_list(elem) or is_integer_array(elem) for elem in selection)
    )


def is_pure_orthogonal_indexing(selection, ndim):
    if not ndim:
        return False

    # Case 1: Selection is a single iterable of integers
    if is_integer_list(selection) or is_integer_array(selection, ndim=1):
        return True

    # Case two: selection contains either zero or one integer iterables.
    # All other selection elements are slices or integers
    return (
        isinstance(selection, tuple)
        and len(selection) == ndim
        and sum(is_integer_list(elem) or is_integer_array(elem) for elem in selection) <= 1
        and all(
            is_integer_list(elem) or is_integer_array(elem) or isinstance(elem, (int, slice))
            for elem in selection
        )
    )


def normalize_integer_selection(dim_sel, dim_len):

    # normalize type to int
    dim_sel = int(dim_sel)

    # handle wraparound
    if dim_sel < 0:
        dim_sel = dim_len + dim_sel

    # handle out of bounds
    if dim_sel >= dim_len or dim_sel < 0:
        raise BoundsCheckError(dim_len)

    return dim_sel


ChunkDimProjection = collections.namedtuple(
    "ChunkDimProjection", ("dim_chunk_ix", "dim_chunk_sel", "dim_out_sel")
)
"""A mapping from chunk to output array for a single dimension.

Parameters
----------
dim_chunk_ix
    Index of chunk.
dim_chunk_sel
    Selection of items from chunk array.
dim_out_sel
    Selection of items in target (output) array.

"""


class IntDimIndexer:
    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # normalize
        dim_sel = normalize_integer_selection(dim_sel, dim_len)

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = 1

    def __iter__(self):
        dim_chunk_ix = self.dim_sel // self.dim_chunk_len
        dim_offset = dim_chunk_ix * self.dim_chunk_len
        dim_chunk_sel = self.dim_sel - dim_offset
        dim_out_sel = None
        yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def ceildiv(a, b):
    return math.ceil(a / b)


class SliceDimIndexer:
    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # normalize
        self.start, self.stop, self.step = dim_sel.indices(dim_len)
        if self.step < 1:
            raise NegativeStepError()

        # store attributes
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nitems = max(0, ceildiv((self.stop - self.start), self.step))
        self.nchunks = ceildiv(self.dim_len, self.dim_chunk_len)

    def __iter__(self):

        # figure out the range of chunks we need to visit
        dim_chunk_ix_from = self.start // self.dim_chunk_len
        dim_chunk_ix_to = ceildiv(self.stop, self.dim_chunk_len)

        # iterate over chunks in range
        for dim_chunk_ix in range(dim_chunk_ix_from, dim_chunk_ix_to):

            # compute offsets for chunk within overall array
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_limit = min(self.dim_len, (dim_chunk_ix + 1) * self.dim_chunk_len)

            # determine chunk length, accounting for trailing chunk
            dim_chunk_len = dim_limit - dim_offset

            if self.start < dim_offset:
                # selection starts before current chunk
                dim_chunk_sel_start = 0
                remainder = (dim_offset - self.start) % self.step
                if remainder:
                    dim_chunk_sel_start += self.step - remainder
                # compute number of previous items, provides offset into output array
                dim_out_offset = ceildiv((dim_offset - self.start), self.step)

            else:
                # selection starts within current chunk
                dim_chunk_sel_start = self.start - dim_offset
                dim_out_offset = 0

            if self.stop > dim_limit:
                # selection ends after current chunk
                dim_chunk_sel_stop = dim_chunk_len

            else:
                # selection ends within current chunk
                dim_chunk_sel_stop = self.stop - dim_offset

            dim_chunk_sel = slice(dim_chunk_sel_start, dim_chunk_sel_stop, self.step)
            dim_chunk_nitems = ceildiv((dim_chunk_sel_stop - dim_chunk_sel_start), self.step)

            # If there are no elements on the selection within this chunk, then skip
            if dim_chunk_nitems == 0:
                continue

            dim_out_sel = slice(dim_out_offset, dim_out_offset + dim_chunk_nitems)

            yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def check_selection_length(selection, shape):
    if len(selection) > len(shape):
        err_too_many_indices(selection, shape)


def replace_ellipsis(selection, shape):

    selection = ensure_tuple(selection)

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
        selection += (slice(None),) * (len(shape) - len(selection))

    # check selection not too long
    check_selection_length(selection, shape)

    return selection


def replace_lists(selection):
    return tuple(
        np.asarray(dim_sel) if isinstance(dim_sel, list) else dim_sel for dim_sel in selection
    )


def ensure_tuple(v):
    if not isinstance(v, tuple):
        v = (v,)
    return v


ChunkProjection = collections.namedtuple(
    "ChunkProjection", ("chunk_coords", "chunk_selection", "out_selection")
)
"""A mapping of items from chunk to output array. Can be used to extract items from the
chunk array for loading into an output array. Can also be used to extract items from a
value array for setting/updating in a chunk array.

Parameters
----------
chunk_coords
    Indices of chunk.
chunk_selection
    Selection of items from chunk array.
out_selection
    Selection of items in target (output) array.

"""


def is_slice(s):
    return isinstance(s, slice)


def is_contiguous_slice(s):
    return is_slice(s) and (s.step is None or s.step == 1)


def is_positive_slice(s):
    return is_slice(s) and (s.step is None or s.step >= 1)


def is_contiguous_selection(selection):
    selection = ensure_tuple(selection)
    return all((is_integer_array(s) or is_contiguous_slice(s) or s == Ellipsis) for s in selection)


def is_basic_selection(selection):
    selection = ensure_tuple(selection)
    return all(is_integer(s) or is_positive_slice(s) for s in selection)


# noinspection PyProtectedMember
class BasicIndexer:
    def __init__(self, selection, array):

        # handle ellipsis
        selection = replace_ellipsis(selection, array._shape)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_len in zip(selection, array._shape, array._chunks):

            if is_integer(dim_sel):
                dim_indexer = IntDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif is_slice(dim_sel):
                dim_indexer = SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)

            else:
                raise IndexError(
                    "unsupported selection item for basic indexing; "
                    "expected integer or slice, got {!r}".format(type(dim_sel))
                )

            dim_indexers.append(dim_indexer)

        self.dim_indexers = dim_indexers
        self.shape = tuple(s.nitems for s in self.dim_indexers if not isinstance(s, IntDimIndexer))
        self.drop_axes = None

    def __iter__(self):
        for dim_projections in itertools.product(*self.dim_indexers):

            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


class BoolArrayDimIndexer:
    def __init__(self, dim_sel, dim_len, dim_chunk_len):

        # check number of dimensions
        if not is_bool_array(dim_sel, 1):
            raise IndexError(
                "Boolean arrays in an orthogonal selection must " "be 1-dimensional only"
            )

        # check shape
        if dim_sel.shape[0] != dim_len:
            raise IndexError(
                "Boolean array has the wrong length for dimension; "
                "expected {}, got {}".format(dim_len, dim_sel.shape[0])
            )

        # store attributes
        self.dim_sel = dim_sel
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nchunks = ceildiv(self.dim_len, self.dim_chunk_len)

        # precompute number of selected items for each chunk
        self.chunk_nitems = np.zeros(self.nchunks, dtype="i8")
        for dim_chunk_ix in range(self.nchunks):
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            self.chunk_nitems[dim_chunk_ix] = np.count_nonzero(
                self.dim_sel[dim_offset : dim_offset + self.dim_chunk_len]
            )
        self.chunk_nitems_cumsum = np.cumsum(self.chunk_nitems)
        self.nitems = self.chunk_nitems_cumsum[-1]
        self.dim_chunk_ixs = np.nonzero(self.chunk_nitems)[0]

    def __iter__(self):

        # iterate over chunks with at least one item
        for dim_chunk_ix in self.dim_chunk_ixs:

            # find region in chunk
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_chunk_sel = self.dim_sel[dim_offset : dim_offset + self.dim_chunk_len]

            # pad out if final chunk
            if dim_chunk_sel.shape[0] < self.dim_chunk_len:
                tmp = np.zeros(self.dim_chunk_len, dtype=bool)
                tmp[: dim_chunk_sel.shape[0]] = dim_chunk_sel
                dim_chunk_sel = tmp

            # find region in output
            if dim_chunk_ix == 0:
                start = 0
            else:
                start = self.chunk_nitems_cumsum[dim_chunk_ix - 1]
            stop = self.chunk_nitems_cumsum[dim_chunk_ix]
            dim_out_sel = slice(start, stop)

            yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


class Order:
    UNKNOWN = 0
    INCREASING = 1
    DECREASING = 2
    UNORDERED = 3

    @staticmethod
    def check(a):
        diff = np.diff(a)
        diff_positive = diff >= 0
        n_diff_positive = np.count_nonzero(diff_positive)
        all_increasing = n_diff_positive == len(diff_positive)
        any_increasing = n_diff_positive > 0
        if all_increasing:
            order = Order.INCREASING
        elif any_increasing:
            order = Order.UNORDERED
        else:
            order = Order.DECREASING
        return order


def wraparound_indices(x, dim_len):
    loc_neg = x < 0
    if np.any(loc_neg):
        x[loc_neg] = x[loc_neg] + dim_len


def boundscheck_indices(x, dim_len):
    if np.any(x < 0) or np.any(x >= dim_len):
        raise BoundsCheckError(dim_len)


class IntArrayDimIndexer:
    """Integer array selection against a single dimension."""

    def __init__(
        self,
        dim_sel,
        dim_len,
        dim_chunk_len,
        wraparound=True,
        boundscheck=True,
        order=Order.UNKNOWN,
    ):

        # ensure 1d array
        dim_sel = np.asanyarray(dim_sel)
        if not is_integer_array(dim_sel, 1):
            raise IndexError(
                "integer arrays in an orthogonal selection must be " "1-dimensional only"
            )

        # handle wraparound
        if wraparound:
            wraparound_indices(dim_sel, dim_len)

        # handle out of bounds
        if boundscheck:
            boundscheck_indices(dim_sel, dim_len)

        # store attributes
        self.dim_len = dim_len
        self.dim_chunk_len = dim_chunk_len
        self.nchunks = ceildiv(self.dim_len, self.dim_chunk_len)
        self.nitems = len(dim_sel)

        # determine which chunk is needed for each selection item
        # note: for dense integer selections, the division operation here is the
        # bottleneck
        dim_sel_chunk = dim_sel // dim_chunk_len

        # determine order of indices
        if order == Order.UNKNOWN:
            order = Order.check(dim_sel)
        self.order = order

        if self.order == Order.INCREASING:
            self.dim_sel = dim_sel
            self.dim_out_sel = None
        elif self.order == Order.DECREASING:
            self.dim_sel = dim_sel[::-1]
            # TODO should be possible to do this without creating an arange
            self.dim_out_sel = np.arange(self.nitems - 1, -1, -1)
        else:
            # sort indices to group by chunk
            self.dim_out_sel = np.argsort(dim_sel_chunk)
            self.dim_sel = np.take(dim_sel, self.dim_out_sel)

        # precompute number of selected items for each chunk
        self.chunk_nitems = np.bincount(dim_sel_chunk, minlength=self.nchunks)

        # find chunks that we need to visit
        self.dim_chunk_ixs = np.nonzero(self.chunk_nitems)[0]

        # compute offsets into the output array
        self.chunk_nitems_cumsum = np.cumsum(self.chunk_nitems)

    def __iter__(self):

        for dim_chunk_ix in self.dim_chunk_ixs:

            # find region in output
            if dim_chunk_ix == 0:
                start = 0
            else:
                start = self.chunk_nitems_cumsum[dim_chunk_ix - 1]
            stop = self.chunk_nitems_cumsum[dim_chunk_ix]
            if self.order == Order.INCREASING:
                dim_out_sel = slice(start, stop)
            else:
                dim_out_sel = self.dim_out_sel[start:stop]

            # find region in chunk
            dim_offset = dim_chunk_ix * self.dim_chunk_len
            dim_chunk_sel = self.dim_sel[start:stop] - dim_offset

            yield ChunkDimProjection(dim_chunk_ix, dim_chunk_sel, dim_out_sel)


def slice_to_range(s: slice, l: int):  # noqa: E741
    return range(*s.indices(l))


def ix_(selection, shape):
    """Convert an orthogonal selection to a numpy advanced (fancy) selection, like numpy.ix_
    but with support for slices and single ints."""

    # normalisation
    selection = replace_ellipsis(selection, shape)

    # replace slice and int as these are not supported by numpy.ix_
    selection = [
        slice_to_range(dim_sel, dim_len)
        if isinstance(dim_sel, slice)
        else [dim_sel]
        if is_integer(dim_sel)
        else dim_sel
        for dim_sel, dim_len in zip(selection, shape)
    ]

    # now get numpy to convert to a coordinate selection
    selection = np.ix_(*selection)

    return selection


def oindex(a, selection):
    """Implementation of orthogonal indexing with slices and ints."""
    selection = replace_ellipsis(selection, a.shape)
    drop_axes = tuple(i for i, s in enumerate(selection) if is_integer(s))
    selection = ix_(selection, a.shape)
    result = a[selection]
    if drop_axes:
        result = result.squeeze(axis=drop_axes)
    return result


def oindex_set(a, selection, value):
    selection = replace_ellipsis(selection, a.shape)
    drop_axes = tuple(i for i, s in enumerate(selection) if is_integer(s))
    selection = ix_(selection, a.shape)
    if not np.isscalar(value) and drop_axes:
        value = np.asanyarray(value)
        value_selection = [slice(None)] * len(a.shape)
        for i in drop_axes:
            value_selection[i] = np.newaxis
        value_selection = tuple(value_selection)
        value = value[value_selection]
    a[selection] = value


# noinspection PyProtectedMember
class OrthogonalIndexer:
    def __init__(self, selection, array):

        # handle ellipsis
        selection = replace_ellipsis(selection, array._shape)

        # normalize list to array
        selection = replace_lists(selection)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_len in zip(selection, array._shape, array._chunks):

            if is_integer(dim_sel):
                dim_indexer = IntDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif isinstance(dim_sel, slice):
                dim_indexer = SliceDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif is_integer_array(dim_sel):
                dim_indexer = IntArrayDimIndexer(dim_sel, dim_len, dim_chunk_len)

            elif is_bool_array(dim_sel):
                dim_indexer = BoolArrayDimIndexer(dim_sel, dim_len, dim_chunk_len)

            else:
                raise IndexError(
                    "unsupported selection item for orthogonal indexing; "
                    "expected integer, slice, integer array or Boolean "
                    "array, got {!r}".format(type(dim_sel))
                )

            dim_indexers.append(dim_indexer)

        self.array = array
        self.dim_indexers = dim_indexers
        self.shape = tuple(s.nitems for s in self.dim_indexers if not isinstance(s, IntDimIndexer))
        self.is_advanced = not is_basic_selection(selection)
        if self.is_advanced:
            self.drop_axes = tuple(
                i
                for i, dim_indexer in enumerate(self.dim_indexers)
                if isinstance(dim_indexer, IntDimIndexer)
            )
        else:
            self.drop_axes = None

    def __iter__(self):
        for dim_projections in itertools.product(*self.dim_indexers):

            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            # handle advanced indexing arrays orthogonally
            if self.is_advanced:

                # N.B., numpy doesn't support orthogonal indexing directly as yet,
                # so need to work around via np.ix_. Also np.ix_ does not support a
                # mixture of arrays and slices or integers, so need to convert slices
                # and integers into ranges.
                chunk_selection = ix_(chunk_selection, self.array._chunks)

                # special case for non-monotonic indices
                if not is_basic_selection(out_selection):
                    out_selection = ix_(out_selection, self.shape)

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


class OIndex:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, selection):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.get_orthogonal_selection(selection, fields=fields)

    def __setitem__(self, selection, value):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.set_orthogonal_selection(selection, value, fields=fields)


# noinspection PyProtectedMember
class BlockIndexer:
    def __init__(self, selection, array):

        # handle ellipsis
        selection = replace_ellipsis(selection, array._shape)

        # normalize list to array
        selection = replace_lists(selection)

        # setup per-dimension indexers
        dim_indexers = []
        for dim_sel, dim_len, dim_chunk_size in zip(selection, array._shape, array._chunks):
            dim_numchunks = int(np.ceil(dim_len / dim_chunk_size))

            if is_integer(dim_sel):
                if dim_sel < 0:
                    dim_sel = dim_numchunks + dim_sel

                start = dim_sel * dim_chunk_size
                stop = start + dim_chunk_size
                slice_ = slice(start, stop)

            elif is_slice(dim_sel):
                start = dim_sel.start if dim_sel.start is not None else 0
                stop = dim_sel.stop if dim_sel.stop is not None else dim_numchunks

                if dim_sel.step not in {1, None}:
                    raise IndexError(
                        "unsupported selection item for block indexing; "
                        "expected integer or slice with step=1, got {!r}".format(type(dim_sel))
                    )

                # Can't reuse wraparound_indices because it expects a numpy array
                # We have integers here.
                if start < 0:
                    start = dim_numchunks + start
                if stop < 0:
                    stop = dim_numchunks + stop

                start = start * dim_chunk_size
                stop = stop * dim_chunk_size
                slice_ = slice(start, stop)

            else:
                raise IndexError(
                    "unsupported selection item for block indexing; "
                    "expected integer or slice, got {!r}".format(type(dim_sel))
                )

            dim_indexer = SliceDimIndexer(slice_, dim_len, dim_chunk_size)
            dim_indexers.append(dim_indexer)

            if start >= dim_len or start < 0:
                raise BoundsCheckError(dim_len)

        self.dim_indexers = dim_indexers
        self.shape = tuple(s.nitems for s in self.dim_indexers)
        self.drop_axes = None

    def __iter__(self):
        for dim_projections in itertools.product(*self.dim_indexers):
            chunk_coords = tuple(p.dim_chunk_ix for p in dim_projections)
            chunk_selection = tuple(p.dim_chunk_sel for p in dim_projections)
            out_selection = tuple(
                p.dim_out_sel for p in dim_projections if p.dim_out_sel is not None
            )

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


class BlockIndex:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, selection):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.get_block_selection(selection, fields=fields)

    def __setitem__(self, selection, value):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return self.array.set_block_selection(selection, value, fields=fields)


# noinspection PyProtectedMember
def is_coordinate_selection(selection, array):
    return (len(selection) == len(array._shape)) and all(
        is_integer(dim_sel) or is_integer_array(dim_sel) for dim_sel in selection
    )


# noinspection PyProtectedMember
def is_mask_selection(selection, array):
    return (
        len(selection) == 1 and is_bool_array(selection[0]) and selection[0].shape == array._shape
    )


# noinspection PyProtectedMember
class CoordinateIndexer:
    def __init__(self, selection, array):

        # some initial normalization
        selection = ensure_tuple(selection)
        selection = tuple([i] if is_integer(i) else i for i in selection)
        selection = replace_lists(selection)

        # validation
        if not is_coordinate_selection(selection, array):
            raise IndexError(
                "invalid coordinate selection; expected one integer "
                "(coordinate) array per dimension of the target array, "
                "got {!r}".format(selection)
            )

        # handle wraparound, boundscheck
        for dim_sel, dim_len in zip(selection, array.shape):

            # handle wraparound
            wraparound_indices(dim_sel, dim_len)

            # handle out of bounds
            boundscheck_indices(dim_sel, dim_len)

        # compute chunk index for each point in the selection
        chunks_multi_index = tuple(
            dim_sel // dim_chunk_len for (dim_sel, dim_chunk_len) in zip(selection, array._chunks)
        )

        # broadcast selection - this will raise error if array dimensions don't match
        selection = np.broadcast_arrays(*selection)
        chunks_multi_index = np.broadcast_arrays(*chunks_multi_index)

        # remember shape of selection, because we will flatten indices for processing
        self.sel_shape = selection[0].shape if selection[0].shape else (1,)

        # flatten selection
        selection = [dim_sel.reshape(-1) for dim_sel in selection]
        chunks_multi_index = [dim_chunks.reshape(-1) for dim_chunks in chunks_multi_index]

        # ravel chunk indices
        chunks_raveled_indices = np.ravel_multi_index(chunks_multi_index, dims=array._cdata_shape)

        # group points by chunk
        if np.any(np.diff(chunks_raveled_indices) < 0):
            # optimisation, only sort if needed
            sel_sort = np.argsort(chunks_raveled_indices)
            selection = tuple(dim_sel[sel_sort] for dim_sel in selection)
        else:
            sel_sort = None

        # store attributes
        self.selection = selection
        self.sel_sort = sel_sort
        self.shape = selection[0].shape if selection[0].shape else (1,)
        self.drop_axes = None
        self.array = array

        # precompute number of selected items for each chunk
        self.chunk_nitems = np.bincount(chunks_raveled_indices, minlength=array.nchunks)
        self.chunk_nitems_cumsum = np.cumsum(self.chunk_nitems)
        # locate the chunks we need to process
        self.chunk_rixs = np.nonzero(self.chunk_nitems)[0]

        # unravel chunk indices
        self.chunk_mixs = np.unravel_index(self.chunk_rixs, array._cdata_shape)

    def __iter__(self):

        # iterate over chunks
        for i, chunk_rix in enumerate(self.chunk_rixs):

            chunk_coords = tuple(m[i] for m in self.chunk_mixs)
            if chunk_rix == 0:
                start = 0
            else:
                start = self.chunk_nitems_cumsum[chunk_rix - 1]
            stop = self.chunk_nitems_cumsum[chunk_rix]
            if self.sel_sort is None:
                out_selection = slice(start, stop)
            else:
                out_selection = self.sel_sort[start:stop]

            chunk_offsets = tuple(
                dim_chunk_ix * dim_chunk_len
                for dim_chunk_ix, dim_chunk_len in zip(chunk_coords, self.array._chunks)
            )
            chunk_selection = tuple(
                dim_sel[start:stop] - dim_chunk_offset
                for (dim_sel, dim_chunk_offset) in zip(self.selection, chunk_offsets)
            )

            yield ChunkProjection(chunk_coords, chunk_selection, out_selection)


# noinspection PyProtectedMember
class MaskIndexer(CoordinateIndexer):
    def __init__(self, selection, array):

        # some initial normalization
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)

        # validation
        if not is_mask_selection(selection, array):
            raise IndexError(
                "invalid mask selection; expected one Boolean (mask)"
                "array with the same shape as the target array, got {!r}".format(selection)
            )

        # convert to indices
        selection = np.nonzero(selection[0])

        # delegate the rest to superclass
        super().__init__(selection, array)


class VIndex:
    def __init__(self, array):
        self.array = array

    def __getitem__(self, selection):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        if is_coordinate_selection(selection, self.array):
            return self.array.get_coordinate_selection(selection, fields=fields)
        elif is_mask_selection(selection, self.array):
            return self.array.get_mask_selection(selection, fields=fields)
        else:
            raise VindexInvalidSelectionError(selection)

    def __setitem__(self, selection, value):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        if is_coordinate_selection(selection, self.array):
            self.array.set_coordinate_selection(selection, value, fields=fields)
        elif is_mask_selection(selection, self.array):
            self.array.set_mask_selection(selection, value, fields=fields)
        else:
            raise VindexInvalidSelectionError(selection)


def check_fields(fields, dtype):
    # early out
    if fields is None:
        return dtype
    # check type
    if not isinstance(fields, (str, list, tuple)):
        raise IndexError(
            "'fields' argument must be a string or list of strings; found "
            "{!r}".format(type(fields))
        )
    if fields:
        if dtype.names is None:
            raise IndexError("invalid 'fields' argument, array does not have any fields")
        try:
            if isinstance(fields, str):
                # single field selection
                out_dtype = dtype[fields]
            else:
                # multiple field selection
                out_dtype = np.dtype([(f, dtype[f]) for f in fields])
        except KeyError as e:
            raise IndexError("invalid 'fields' argument, field not found: {!r}".format(e))
        else:
            return out_dtype
    else:
        return dtype


def check_no_multi_fields(fields):
    if isinstance(fields, list):
        if len(fields) == 1:
            return fields[0]
        elif len(fields) > 1:
            raise IndexError("multiple fields are not supported for this operation")
    return fields


def pop_fields(selection):
    if isinstance(selection, str):
        # single field selection
        fields = selection
        selection = ()
    elif not isinstance(selection, tuple):
        # single selection item, no fields
        fields = None
        # leave selection as-is
    else:
        # multiple items, split fields from selection items
        fields = [f for f in selection if isinstance(f, str)]
        fields = fields[0] if len(fields) == 1 else fields
        selection = tuple(s for s in selection if not isinstance(s, str))
        selection = selection[0] if len(selection) == 1 else selection
    return fields, selection


def make_slice_selection(selection):
    ls = []
    for dim_selection in selection:
        if is_integer(dim_selection):
            ls.append(slice(int(dim_selection), int(dim_selection) + 1, 1))
        elif isinstance(dim_selection, np.ndarray):
            if len(dim_selection) == 1:
                ls.append(slice(int(dim_selection[0]), int(dim_selection[0]) + 1, 1))
            else:
                raise ArrayIndexError()
        else:
            ls.append(dim_selection)
    return ls


class PartialChunkIterator:
    """Iterator to retrieve the specific coordinates of requested data
    from within a compressed chunk.

    Parameters
    ----------
    selection : tuple
        tuple of slice objects to take from the chunk
    arr_shape : shape of chunk to select data from

    Attributes
    -----------
    arr_shape
    selection

    Returns
    -------
    Tuple with 3 elements:

    start: int
        elements offset in the chunk to read from
    nitems: int
        number of elements to read in the chunk from start
    partial_out_selection: list of slices
        indices of a temporary empty array of size `Array._chunks` to assign
        the decompressed data to after the partial read.

    Notes
    -----
    An array is flattened when compressed with blosc, so this iterator takes
    the wanted selection of an array and determines the wanted coordinates
    of the flattened, compressed data to be read and then decompressed. The
    decompressed data is then placed in a temporary empty array of size
    `Array._chunks` at the indices yielded as partial_out_selection.
    Once all the slices yielded by this iterator have been read, decompressed
    and written to the temporary array, the wanted slice of the chunk can be
    indexed from the temporary array and written to the out_selection slice
    of the out array.

    """

    def __init__(self, selection, arr_shape):
        selection = make_slice_selection(selection)
        self.arr_shape = arr_shape

        # number of selection dimensions can't be greater than the number of chunk dimensions
        if len(selection) > len(self.arr_shape):
            raise ValueError(
                "Selection has more dimensions then the array:\n"
                f"selection dimensions = {len(selection)}\n"
                f"array dimensions = {len(self.arr_shape)}"
            )

        # any selection can not be out of the range of the chunk
        selection_shape = np.empty(self.arr_shape)[tuple(selection)].shape
        if any(
            selection_dim < 0 or selection_dim > arr_dim
            for selection_dim, arr_dim in zip(selection_shape, self.arr_shape)
        ):
            raise IndexError(
                "a selection index is out of range for the dimension"
            )  # pragma: no cover

        for i, dim_size in enumerate(self.arr_shape[::-1]):
            index = len(self.arr_shape) - (i + 1)
            if index <= len(selection) - 1:
                slice_size = selection_shape[index]
                if slice_size == dim_size and index > 0:
                    selection.pop()
                else:
                    break

        chunk_loc_slices = []
        last_dim_slice = None if selection[-1].step > 1 else selection.pop()
        for arr_shape_i, sl in zip(arr_shape, selection):
            dim_chunk_loc_slices = []
            assert isinstance(sl, slice)
            for x in slice_to_range(sl, arr_shape_i):
                dim_chunk_loc_slices.append(slice(x, x + 1, 1))
            chunk_loc_slices.append(dim_chunk_loc_slices)
        if last_dim_slice:
            chunk_loc_slices.append([last_dim_slice])
        self.chunk_loc_slices = list(itertools.product(*chunk_loc_slices))

    def __iter__(self):
        chunk1 = self.chunk_loc_slices[0]
        nitems = (chunk1[-1].stop - chunk1[-1].start) * np.prod(
            self.arr_shape[len(chunk1) :], dtype=int
        )
        for partial_out_selection in self.chunk_loc_slices:
            start = 0
            for i, sl in enumerate(partial_out_selection):
                start += sl.start * np.prod(self.arr_shape[i + 1 :], dtype=int)
            yield start, nitems, partial_out_selection
