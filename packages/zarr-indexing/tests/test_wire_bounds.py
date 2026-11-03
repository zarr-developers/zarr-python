"""Index-array bounds are validated against raw coordinates during loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from zarr_indexing import ConstantMap, IndexTransform, normalize_ndsel
from zarr_indexing.messages import NdselError
from zarr_indexing.output_map import output_index_map_from_json

if TYPE_CHECKING:
    from zarr_indexing.json import IndexTransformJSON, OutputIndexMapJSON


def load(values: Any, bounds: Any, entry: str, offset: int, stride: int) -> Any:
    output: OutputIndexMapJSON = {"index_array": values, "offset": offset, "stride": stride}
    if bounds is not None:
        output["index_array_bounds"] = bounds
    if entry == "output_map":
        return output_index_map_from_json(output)
    body: IndexTransformJSON = {"input_shape": list(np.asarray(values).shape), "output": [output]}
    return IndexTransform.from_json(body)


@pytest.mark.parametrize("entry", ["transform", "output_map"])
@pytest.mark.parametrize(("offset", "stride"), [(0, 1), (10, -2), (7, 0)])
@pytest.mark.parametrize(
    ("values", "bounds"),
    [
        ([-3, 0, 4], [-3, 4]),
        ([-3, 0, 4], ["-inf", 4]),
        ([-3, 0, 4], [-3, "+inf"]),
        ([-3, 0, 4], ["-inf", "+inf"]),
        ([-3, 0, 4], None),
        ([2], [2, 2]),
        ([], [2, 2]),
        ([], ["+inf", "+inf"]),
        ([], ["-inf", "-inf"]),
        ([[0, 2], [2, 0]], [0, 2]),
    ],
)
def test_valid_bounds_preserve_values_and_roundtrip(
    values: Any, bounds: Any, entry: str, offset: int, stride: int
) -> None:
    model = np.asarray(values, dtype=np.intp)
    loaded = load(values, bounds, entry, offset, stride)
    expected = offset + stride * model
    if entry == "transform":
        points = np.array(list(np.ndindex(model.shape)), dtype=np.intp).reshape(-1, model.ndim)
        restored = IndexTransform.from_json(loaded.to_json())
        np.testing.assert_array_equal(loaded.apply_many(points).reshape(model.shape), expected)
        np.testing.assert_array_equal(restored.apply_many(points).reshape(model.shape), expected)
        # Further selection and affine adjustment operate on validated coordinates.
        if model.ndim == 1 and model.size > 1:
            selected = loaded.oindex[[model.size - 1, 0]].translate((3,))
            np.testing.assert_array_equal(
                selected.apply_many([[0], [1]]).ravel(), expected[[-1, 0]] + 3
            )
    else:
        restored = output_index_map_from_json(loaded.to_json())
        if isinstance(restored, ConstantMap):
            np.testing.assert_array_equal(np.full(model.shape, restored.offset), expected)
        else:
            assert restored == loaded
        np.testing.assert_array_equal(loaded.offset + loaded.stride * loaded.index_array, expected)
    if bounds is not None:
        canonical = normalize_ndsel(
            {
                "kind": "transform",
                "input_shape": list(model.shape),
                "output": [{"index_array": values, "index_array_bounds": bounds}],
            }
        )
        assert canonical["output"][0]["index_array_bounds"] == bounds


@pytest.mark.parametrize("entry", ["transform", "output_map"])
@pytest.mark.parametrize(("offset", "stride"), [(0, 1), (10, -2), (7, 0)])
@pytest.mark.parametrize(
    ("values", "bounds"),
    [
        ([-1, 0], [0, 2]),
        ([0, 3], [0, 2]),
        ([3], [2, 2]),
        ([-1], [0, "+inf"]),
        ([3], ["-inf", 2]),
        ([0], ["+inf", "+inf"]),
        ([0], ["-inf", "-inf"]),
        ([[0, 1], [1, 3]], [0, 2]),
    ],
)
def test_out_of_bounds_raw_values_are_rejected(
    values: Any, bounds: Any, entry: str, offset: int, stride: int
) -> None:
    with pytest.raises(NdselError, match="index_array_bounds") as exc:
        load(values, bounds, entry, offset, stride)
    assert exc.value.reason == "invalid_json"


@pytest.mark.parametrize("entry", ["transform", "output_map"])
@pytest.mark.parametrize("values", [[], [1]])
@pytest.mark.parametrize("bounds", [[], [0], [0, 1, 2], [False, 2], [0.5, 2], ["bad", 2]])
def test_malformed_bounds_are_rejected(values: Any, bounds: Any, entry: str) -> None:
    with pytest.raises(NdselError) as exc:
        load(values, bounds, entry, 0, 1)
    assert exc.value.reason == "invalid_json"


@pytest.mark.parametrize("entry", ["transform", "output_map"])
@pytest.mark.parametrize("values", [[], [1]])
def test_reversed_bounds_are_rejected(values: Any, entry: str) -> None:
    with pytest.raises(NdselError) as exc:
        load(values, [2, 0], entry, 0, 1)
    assert exc.value.reason == "bounds_out_of_order"


@pytest.mark.parametrize("entry", ["transform", "output_map"])
@pytest.mark.parametrize("value", [int(np.iinfo(np.intp).min), int(np.iinfo(np.intp).max)])
def test_integer_extreme_bounds_are_exact(entry: str, value: int) -> None:
    result = load([value], [value, value], entry, 0, 1)
    if entry == "transform":
        assert result.apply((0,)) == (value,)
    else:
        assert int(result.index_array[0]) == value
    bounds = [value + 1, "+inf"] if value < 0 else ["-inf", value - 1]
    with pytest.raises(NdselError, match="index_array_bounds"):
        load([value], bounds, entry, 0, 1)
