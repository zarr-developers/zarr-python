"""Boundary validation must preserve coordinates and canonical messages."""

from typing import Any

import numpy as np
import pytest

from zarr_indexing._wire import lower_index_array
from zarr_indexing.messages import NdselError, normalize_ndsel


@pytest.mark.parametrize("raw", [[True, 2], [[0, False]], [True, False]])
def test_index_array_rejects_boolean_elements(raw: Any) -> None:
    with pytest.raises(NdselError, match="integers"):
        lower_index_array(raw, "index_array")


@pytest.mark.parametrize("value", [int(np.iinfo(np.intp).max) + 1, int(np.iinfo(np.intp).min) - 1])
def test_index_array_rejects_out_of_range_coordinates(value: int) -> None:
    with pytest.raises(NdselError):
        lower_index_array([value], "index_array")


@pytest.mark.parametrize(
    "raw", [[], [[], []], [0, -1, 2], [int(np.iinfo(np.intp).min), int(np.iinfo(np.intp).max)]]
)
def test_index_array_preserves_integer_coordinates(raw: Any) -> None:
    result = lower_index_array(raw, "index_array")
    assert result.dtype == np.dtype(np.intp)
    assert result.tolist() == raw


@pytest.mark.parametrize(("start", "stop"), [(-(2**63), -(2**63)), (-(2**63) + 1, -(2**63))])
def test_slice_rejects_unrepresentable_canonical_bounds(start: int, stop: int) -> None:
    with pytest.raises(NdselError):
        normalize_ndsel({"kind": "slice", "start": [start], "stop": [stop], "step": [-1]})


@pytest.mark.parametrize(
    "message",
    [
        {"kind": "box", "shape": [1] * 33},
        {"kind": "slice", "start": [0] * 33, "stop": [1] * 33},
        {"kind": "transform", "input_shape": [1] * 33},
    ],
)
def test_normalize_rejects_excessive_inferred_rank(message: dict[str, Any]) -> None:
    with pytest.raises(NdselError, match="rank"):
        normalize_ndsel(message)


@pytest.mark.parametrize(
    "message",
    [
        {"kind": "box", "shape": [1] * 32},
        {"kind": "transform", "input_rank": 32},
        {"kind": "slice", "start": [-(2**63) + 1], "stop": [-(2**63) + 1], "step": [-1]},
        {"kind": "slice", "start": [2**63 - 1], "stop": [-(2**63) + 1], "step": [-(2**63)]},
    ],
)
def test_normalize_boundary_results_are_idempotent(message: dict[str, Any]) -> None:
    canonical = normalize_ndsel(message)
    assert normalize_ndsel({"kind": "transform", **canonical}) == canonical
