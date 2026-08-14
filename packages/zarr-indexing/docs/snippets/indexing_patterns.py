"""Every indexing idiom hand-built as an IndexTransform, proven against the compiler."""

from typing import Any, Literal, TypedDict

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


# --8<-- [start:indexing-patterns]
class PatternCase(TypedDict):
    """One indexing idiom: its hand-built transform model and its NumPy result."""

    name: str
    mode: Literal["basic", "oindex", "vindex"]
    selection: Any
    transform: IndexTransform
    expected: Any
    shape: tuple[int, ...]
    category: Literal["box", "query"]


image = np.arange(48).reshape(6, 8)
rows = np.array([4, 1, 1], dtype=np.intp)
columns = np.array([2, 5], dtype=np.intp)
mask = image % 5 == 0
mask_rows, mask_columns = np.nonzero(mask)
vector_rows = np.array([4, 1, 1], dtype=np.intp)
vector_columns = np.array([2, 5, 2], dtype=np.intp)
broadcast_rows = np.array([[0], [3]], dtype=np.intp)
broadcast_columns = np.array([[1, 4, 6]], dtype=np.intp)

PATTERN_CASES: tuple[PatternCase, ...] = (
    {
        # image[1:5, ::2] — an offset picks where cell 0 reads; a stride skips.
        "name": "basic-slice",
        "mode": "basic",
        "selection": (slice(1, 5), slice(None, None, 2)),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((4, 4)),
            output=(
                DimensionMap(input_dimension=0, offset=1),
                DimensionMap(input_dimension=1, stride=2),
            ),
        ),
        "expected": image[1:5, ::2],
        "shape": (4, 4),
        "category": "box",
    },
    {
        # image[2, :] — the dropped axis survives as a ConstantMap: the result
        # is rank 1, but there is still one output map per source dimension.
        "name": "integer-axis-removal",
        "mode": "basic",
        "selection": (2, slice(None)),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((8,)),
            output=(ConstantMap(2), DimensionMap(input_dimension=0)),
        ),
        "expected": image[2, :],
        "shape": (8,),
        "category": "box",
    },
    {
        # image[::-2, :] — reversal is only a negative stride; the offset is
        # where result cell 0 reads (the last selected row, 5).
        "name": "negative-stride",
        "mode": "basic",
        "selection": (slice(None, None, -2), slice(None)),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((3, 8)),
            output=(
                DimensionMap(input_dimension=0, offset=5, stride=-2),
                DimensionMap(input_dimension=1),
            ),
        ),
        "expected": image[::-2, :],
        "shape": (3, 8),
        "category": "box",
    },
    {
        # image[2:2, :] — emptiness lives in the domain; the maps are ordinary.
        "name": "empty-selection",
        "mode": "basic",
        "selection": (slice(2, 2), slice(None)),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((0, 8)),
            output=(
                DimensionMap(input_dimension=0, offset=2),
                DimensionMap(input_dimension=1),
            ),
        ),
        "expected": image[2:2, :],
        "shape": (0, 8),
        "category": "box",
    },
    {
        # image[mask] — a mask is its nonzero coordinates: two correlated
        # ArrayMaps over one flat result axis, row i paired with column i.
        "name": "boolean-mask",
        "mode": "vindex",
        "selection": mask,
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ArrayMap(mask_rows), ArrayMap(mask_columns)),
        ),
        "expected": image[mask],
        "shape": (10,),
        "category": "query",
    },
    {
        # image[np.ix_(rows, columns)] — the outer product is spelled by shape:
        # each array varies over its own distinct axis, singleton on the other.
        "name": "orthogonal",
        "mode": "oindex",
        "selection": (rows, columns),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((3, 2)),
            output=(ArrayMap(rows.reshape(3, 1)), ArrayMap(columns.reshape(1, 2))),
        ),
        "expected": image[np.ix_(rows, columns)],
        "shape": (3, 2),
        "category": "query",
    },
    {
        # image[vector_rows, vector_columns] — pointwise: both arrays share
        # the same axis, so entry i of each pairs into one coordinate.
        "name": "vectorized",
        "mode": "vindex",
        "selection": (vector_rows, vector_columns),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(vector_rows), ArrayMap(vector_columns)),
        ),
        "expected": image[vector_rows, vector_columns],
        "shape": (3,),
        "category": "query",
    },
    {
        # image[broadcast_rows, broadcast_columns] — NumPy broadcasting,
        # materialized: each map carries the full (2, 3) broadcast block.
        "name": "broadcasting",
        "mode": "vindex",
        "selection": (broadcast_rows, broadcast_columns),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((2, 3)),
            output=(
                ArrayMap(np.broadcast_to(broadcast_rows, (2, 3))),
                ArrayMap(np.broadcast_to(broadcast_columns, (2, 3))),
            ),
        ),
        "expected": image[broadcast_rows, broadcast_columns],
        "shape": (2, 3),
        "category": "query",
    },
    {
        # image[rows, 2:6] — one lookup-table axis (repeats and order kept)
        # beside one ordinary affine axis: one ArrayMap makes the whole
        # selection a query.
        "name": "repeated-out-of-order",
        "mode": "oindex",
        "selection": (rows, slice(2, 6)),
        "transform": IndexTransform(
            domain=IndexDomain.from_shape((3, 4)),
            output=(
                ArrayMap(rows.reshape(3, 1)),
                DimensionMap(input_dimension=1, offset=2),
            ),
        ),
        "expected": image[rows, 2:6],
        "shape": (3, 4),
        "category": "query",
    },
)

base = IndexTransform.from_shape(image.shape)


def resolve(transform: IndexTransform) -> np.ndarray[Any, Any]:
    """Materialize a transform against `image` through the public reader."""
    out = np.empty(transform.domain.shape, dtype=image.dtype)
    numpy_reader.read_into(image, ReadContext(transform), out)
    return out


for case in PATTERN_CASES:
    transform = case["transform"]

    # The model is the idiom: shape, values, and category all follow from it.
    assert transform.domain.shape == case["shape"]
    np.testing.assert_array_equal(resolve(transform), case["expected"])
    is_query = any(isinstance(m, ArrayMap) for m in transform.output)
    assert ("query" if is_query else "box") == case["category"]

    # The selection compiler derives the same transform. Compiled basic
    # selections keep literal domains (t[1:5, ...] starts at 1, not 0);
    # re-zeroing exposes the equality with the NumPy-shaped model.
    compiled = base[case["selection"]] if case["mode"] == "basic" else (
        getattr(base, case["mode"])[case["selection"]]
    )
    assert compiled.translate_domain_to((0,) * compiled.input_rank) == transform
# --8<-- [end:indexing-patterns]
