"""Executable reference matrix for the public lazy indexing modes."""

from typing import Any, Literal, TypedDict

import numpy as np

from zarr_indexing import LazyArray


# --8<-- [start:indexing-patterns]
class PatternCase(TypedDict):
    """One documented selection and its independently evaluated NumPy result."""

    name: str
    mode: Literal["basic", "orthogonal", "vectorized"]
    selection: Any
    expected: Any
    shape: tuple[int, ...]
    category: Literal["box", "query"]


image = np.arange(48).reshape(6, 8)
rows = np.array([4, 1, 1], dtype=np.intp)
columns = np.array([2, 5], dtype=np.intp)
mask = image % 5 == 0
vector_rows = np.array([4, 1, 1], dtype=np.intp)
vector_columns = np.array([2, 5, 2], dtype=np.intp)
broadcast_rows = np.array([[0], [3]], dtype=np.intp)
broadcast_columns = np.array([[1, 4, 6]], dtype=np.intp)

PATTERN_CASES: tuple[PatternCase, ...] = (
    {
        "name": "basic-slice",
        "mode": "basic",
        "selection": (slice(1, 5), slice(None, None, 2)),
        "expected": image[1:5, ::2],
        "shape": (4, 4),
        "category": "box",
    },
    {
        "name": "integer-axis-removal",
        "mode": "basic",
        "selection": (2, slice(None)),
        "expected": image[2, :],
        "shape": (8,),
        "category": "box",
    },
    {
        "name": "negative-stride",
        "mode": "basic",
        "selection": (slice(None, None, -2), slice(None)),
        "expected": image[::-2, :],
        "shape": (3, 8),
        "category": "box",
    },
    {
        "name": "empty-selection",
        "mode": "basic",
        "selection": (slice(2, 2), slice(None)),
        "expected": image[2:2, :],
        "shape": (0, 8),
        "category": "box",
    },
    {
        "name": "boolean-mask",
        "mode": "vectorized",
        "selection": mask,
        "expected": image[mask],
        "shape": (10,),
        "category": "query",
    },
    {
        "name": "orthogonal",
        "mode": "orthogonal",
        "selection": (rows, columns),
        "expected": image[np.ix_(rows, columns)],
        "shape": (3, 2),
        "category": "query",
    },
    {
        "name": "vectorized",
        "mode": "vectorized",
        "selection": (vector_rows, vector_columns),
        "expected": image[vector_rows, vector_columns],
        "shape": (3,),
        "category": "query",
    },
    {
        "name": "broadcasting",
        "mode": "vectorized",
        "selection": (broadcast_rows, broadcast_columns),
        "expected": image[broadcast_rows, broadcast_columns],
        "shape": (2, 3),
        "category": "query",
    },
    {
        "name": "repeated-out-of-order",
        "mode": "orthogonal",
        "selection": (rows, slice(2, 6)),
        "expected": image[rows, 2:6],
        "shape": (3, 4),
        "category": "query",
    },
)

lazy = LazyArray(image)
for case in PATTERN_CASES:
    accessor = {
        "basic": lazy.lazy,
        "orthogonal": lazy.lazy.oindex,
        "vectorized": lazy.lazy.vindex,
    }[case["mode"]]
    view = accessor[case["selection"]]
    result = view.result()

    np.testing.assert_array_equal(result, case["expected"])
    assert result.shape == case["shape"]
    assert ("box" if view.is_box else "query") == case["category"]
# --8<-- [end:indexing-patterns]
