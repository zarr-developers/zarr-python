import itertools
from typing import Any, Dict, Generator, Tuple
from zarr.v3.array.v3 import ArrayMetadata
import numpy as np
import numpy.typing as npt
from functools import reduce
from operator import add

from zarr.v3.store import Store
from zarr.v3.types import Attributes


def slicex(start: int, stop: int, step: int):
    """
    Create an "explicit" slice, i.e, a slice where start, stop and step are integers.
    """
    if not isinstance(start, int):
        raise ValueError(f"Start must be an int. Got {start} which has type {type(start)}")
    if not isinstance(stop, int):
        raise ValueError(f"Stop must be an int. Got {stop} which has type {type(stop)}")
    if not isinstance(step, int):
        raise ValueError(f"step must be an int. Got {step} which has type {type(step)}")

    if start < 0:
        raise ValueError(f"Start cannot be negative. Got {start}")
    if stop < 0:
        raise ValueError(f"Stop cannot be negative. Got {stop}")
    if step < 0:
        raise ValueError(f"Step cannot be negative. Got {step}")
    return slice(start, stop, step)


Index = slice


class StoreX:
    async def read(*args) -> Tuple[bytes, ...]:
        ...

    async def write(*args):
        ...


class Array:
    zmeta: ArrayMetadata
    store: Store
    shape: Tuple[int, ...]
    index: Tuple[slice, ...]
    attributes: Attributes
    dtype: np.dtype

    def __init__(
        self, zmeta: ArrayMetadata, store: Store, attributes: Attributes, index: Tuple[slice, ...]
    ):

        self.zmeta = zmeta
        self.store = store
        self.attributes = attributes
        self.index = index
        self.dtype = np.dtype(zmeta.data_type)

    @property
    def shape(self):
        indices = tuple(np.arange(s)[arg] for s, arg in zip(self.zmeta.shape, self.selection))
        shape = tuple(map(len, indices))
        return shape

    def __getitem__(self, *args):
        selection = args[0]
        if len(selection) != len(self.shape):
            raise ValueError(
                f"You provided {len(selection)} slices for {len(self.shape)} dimensions, and we dont normalize / broadcast slices yet"
            )

        return Array(
            zmeta=self.zmeta, store=self.store, attributes=self.attributes, index=selection
        )

    def lowered_indices(self) -> Generator[Tuple[str, Index], None, None]:
        """
        Get a generator for the lowered indices for this array
        """
        return lower_index(self.index, self.zmeta.chunk_grid.configuration.chunk_shape)

    def result(self) -> npt.NDArray:

        out = np.zeros_like(self)

        # lower the indices of this array to (chunk index, slice) pairs
        chunk_indices, chunk_slices = zip(*self.lowered_indices())

        # resolve the chunk_indices to object names
        # with sharding we do something similar to the previous operation, we should exploit the similarity

        chunk_keys = map(self.zmeta.chunk_key_encoding.encode_chunk_key, chunk_indices)

        # request bytes from the store
        self.store.get_async()
        # decode bytes

        # insert each sub-array into the correct location in the output
        ...


def raise_index(
    index: Tuple[Tuple[int, ...], Tuple[slice, ...]], chunk_size: tuple[int, ...]
) -> Tuple[slice, ...]:

    chunk_index, chunk_slice = index
    result = ()

    for chunk_sz, chunk_idx, slce in zip(chunk_size, chunk_index, chunk_slice):
        start = (chunk_sz * chunk_idx) + slce.start
        result += slice(start=start, stop=slce.stop + start, step=1)
    return result


def lower_index(
    slices: Tuple[slice, ...], chunk_size: Tuple[int, ...]
) -> Generator[Tuple[Tuple[int, ...], slice], None, None]:
    """
    Take a tuple of slices that represents a dense N-dimensional discrete interval and a
    tuple of integers that represents a chunk size, and return the indices of each chunk
    intersecting with the interval, as well as the slice within that chunk corresponding to
    the intersection.

    Caveats:
        - no negative slicing for now.
        - only dense slicing, i.e. slice(a, b, 1)
        - this only works for constant chunk size (relaxing this constraint would make this work for sharding, i think)
        - the current implementation has very bad performance scaling.
           It should be reimplemented to use numpy arrays instead of slice objects.

    Example
    -------

    If a 1D array with 2 chunks each with size 4 is sliced by slice(3, 7, 1),
    then this is equivalent to concatenating the result of:
        selecting the chunk with index (0,), and slicing it with slice(3, 4, 1)
        selecting the chunk with index (1,), and slicing it with slice(0, 3, 1)

    This is illustrated by the following diagram.

    data          |1 1 1 1|2 2 2 2|
    element index |0 1 2 3|4 5 6 7|
    slice(3, 7, 1)      [- - - - )
    chunk index   |   0   |   1   |
    chunk_slice_0       [-)
    chunk_slice_1         [- - - )

    accordingly, `lower_index(slice(3, 7, 1), (4,))` yields
    (
        ((0,), slice(3, 4, 1)),
        ((1,), slice(0, 3, 1))
    )

    """
    assert len(slices) == len(chunk_size)

    dim_slices: Tuple[Tuple[Tuple[Tuple[int, ...], Tuple[slice, ...]], ...], ...] = ()

    # for each axis, construct an ordered collection of ((chunk_index,), (chunk_slice),) tuples
    for slc, chunk in zip(slices, chunk_size):
        start_idx = slc.start
        stop_idx = slc.stop
        first_chunk = start_idx // chunk
        last_chunk = (stop_idx - 1) // chunk

        dim_slice = ()

        for lidx, cidx in enumerate(range(first_chunk, 1 + last_chunk)):
            if lidx == 0:
                if slc.start == 0:
                    start = 0
                else:
                    start = np.remainder(slc.start, chunk)
                dim_slice += ((cidx, slice(start, min(chunk, slc.stop), 1)),)
            elif lidx == last_chunk:
                dim_slice += ((cidx, slice(0, np.remainder(slc.stop, chunk), 1)),)
            else:
                dim_slice += ((cidx, slice(0, chunk, 1)),)
        # it's a headache keeping the implicit tuples straight here
        dim_slices += (dim_slice,)

    # we have a tuple of ((chunk_index,), slice) tuples
    for item in itertools.product(*dim_slices):
        chunk_index_parts, chunk_slice_parts = zip(*item)
        yield (tuple(chunk_index_parts), tuple(chunk_slice_parts))
