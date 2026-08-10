"""The three output map kinds, each demonstrated against its NumPy counterpart."""

from typing import Any

import numpy as np

from zarr_indexing import (
    ArrayMap,
    ConstantMap,
    DimensionMap,
    IndexDomain,
    IndexTransform,
    ReadContext,
    numpy_reader,
)

source = np.array([10, 11, 12, 13, 14, 15])


def resolve(transform: IndexTransform, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Materialize `transform` against `values` through the public reader."""
    out = np.empty(transform.domain.shape, dtype=values.dtype)
    numpy_reader.read_into(values, ReadContext(transform), out)
    return out


# --8<-- [start:dimension-map]
# DimensionMap is an affine rule: request i reads source offset + stride * i.
# Its NumPy counterpart is a basic slice.
sliced = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(DimensionMap(input_dimension=0, offset=2, stride=1),),
)
assert resolve(sliced, source).tolist() == source[2:5].tolist()

# A negative stride walks the source backward, like a negative-step slice.
reversed_view = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(DimensionMap(input_dimension=0, offset=4, stride=-2),),
)
assert resolve(reversed_view, source).tolist() == source[4::-2].tolist()
# --8<-- [end:dimension-map]

# --8<-- [start:array-map]
# ArrayMap is an explicit list of source coordinates; order and duplicates
# are semantic. Its NumPy counterpart is fancy indexing.
gather = IndexTransform(
    domain=IndexDomain.from_shape((3,)),
    output=(ArrayMap(index_array=np.array([4, 1, 1])),),
)
assert resolve(gather, source).tolist() == source[[4, 1, 1]].tolist()
# --8<-- [end:array-map]

# --8<-- [start:constant-map]
# ConstantMap reads one source coordinate for every request cell. No NumPy
# selection spells this operation: source[0] drops the axis, and a repeated
# fancy index source[[0, 0, 0, 0]] matches the values but degrades the
# description to a coordinate list. The value-faithful counterpart is a
# broadcast.
repeat = IndexTransform(
    domain=IndexDomain.from_shape((4,)),
    output=(ConstantMap(offset=0),),
)
assert resolve(repeat, source).tolist() == np.broadcast_to(source[0:1], (4,)).tolist()
# --8<-- [end:constant-map]
