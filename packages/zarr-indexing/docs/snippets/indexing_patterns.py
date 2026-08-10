"""Executable reference matrix: every indexing pattern modeled as an IndexTransform."""

from typing import Any, Literal, TypedDict

import numpy as np

from zarr_indexing import ArrayMap, IndexTransform, ReadContext, numpy_reader


# --8<-- [start:indexing-patterns]
class PatternCase(TypedDict):
    """One documented selection and its independently evaluated NumPy result."""

    name: str
    mode: Literal["basic", "oindex", "vindex"]
    selection: Any
    expected: Any
    shape: tuple[int, ...]
    category: Literal["box", "query"]


image = np.arange(48).reshape(6, 8)
t = IndexTransform.from_shape(image.shape)
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
        "mode": "vindex",
        "selection": mask,
        "expected": image[mask],
        "shape": (10,),
        "category": "query",
    },
    {
        "name": "orthogonal",
        "mode": "oindex",
        "selection": (rows, columns),
        "expected": image[np.ix_(rows, columns)],
        "shape": (3, 2),
        "category": "query",
    },
    {
        "name": "vectorized",
        "mode": "vindex",
        "selection": (vector_rows, vector_columns),
        "expected": image[vector_rows, vector_columns],
        "shape": (3,),
        "category": "query",
    },
    {
        "name": "broadcasting",
        "mode": "vindex",
        "selection": (broadcast_rows, broadcast_columns),
        "expected": image[broadcast_rows, broadcast_columns],
        "shape": (2, 3),
        "category": "query",
    },
    {
        "name": "repeated-out-of-order",
        "mode": "oindex",
        "selection": (rows, slice(2, 6)),
        "expected": image[rows, 2:6],
        "shape": (3, 4),
        "category": "query",
    },
)


def compile_selection(mode: str, selection: Any) -> IndexTransform:
    """A selection in any dialect compiles to a transform through its accessor."""
    if mode == "basic":
        return t[selection]
    return getattr(t, mode)[selection]


def resolve(transform: IndexTransform) -> np.ndarray[Any, Any]:
    """Materialize a transform against `image` through the public reader."""
    out = np.empty(transform.domain.shape, dtype=image.dtype)
    numpy_reader.read_into(image, ReadContext(transform), out)
    return out


def category(transform: IndexTransform) -> str:
    """A box carries only constant and affine maps; one lookup table makes a query."""
    return "query" if any(isinstance(m, ArrayMap) for m in transform.output) else "box"


for case in PATTERN_CASES:
    transform = compile_selection(case["mode"], case["selection"])

    assert transform.domain.shape == case["shape"]
    assert category(transform) == case["category"]
    np.testing.assert_array_equal(resolve(transform), case["expected"])
# --8<-- [end:indexing-patterns]
