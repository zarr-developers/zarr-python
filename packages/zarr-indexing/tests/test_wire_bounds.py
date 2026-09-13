"""Engine lowering must never silently discard index-array constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from zarr_indexing import IndexTransform, normalize_ndsel
from zarr_indexing.messages import NdselError
from zarr_indexing.output_map import output_index_map_from_json

if TYPE_CHECKING:
    from zarr_indexing.json import IndexTransformJSON, OutputIndexMapJSON


@pytest.mark.parametrize("bounds", [[0, 3], ["-inf", 3], [0, "+inf"], [0, 0]])
@pytest.mark.parametrize("values", [[], [1], [-1, 1, 3]])
@pytest.mark.parametrize(("offset", "stride"), [(0, 1), (10, -2), (7, 0)])
@pytest.mark.parametrize("entry", ["transform", "output_map"])
def test_lowering_rejects_unsupported_index_array_bounds(
    bounds: list[int | str], values: list[int], offset: int, stride: int, entry: str
) -> None:
    output: OutputIndexMapJSON = {
        "index_array": values,
        "index_array_bounds": bounds,
        "offset": offset,
        "stride": stride,
    }
    body: IndexTransformJSON = {
        "input_inclusive_min": [0],
        "input_shape": [len(values)],
        "output": [output],
    }
    # JSON-only processing preserves constraints for capable consumers.
    canonical = normalize_ndsel({"kind": "transform", **body})
    assert canonical["output"][0]["index_array_bounds"] == bounds
    if entry == "transform":
        with pytest.raises(NdselError, match="index_array_bounds.*unsupported") as exc:
            IndexTransform.from_json(canonical)
        assert exc.value.reason == "invalid_json"
    else:
        with pytest.raises(NdselError, match="index_array_bounds.*unsupported") as exc:
            output_index_map_from_json(output)
        assert exc.value.reason == "invalid_json"


@pytest.mark.parametrize("explicit_bounds", [False, True])
@pytest.mark.parametrize("values", [[], [1], [-1, 1, 3]])
@pytest.mark.parametrize(("offset", "stride"), [(0, 1), (10, -2), (7, 0)])
def test_unconstrained_index_array_roundtrip_evaluation(
    explicit_bounds: bool, values: list[int], offset: int, stride: int
) -> None:
    output: OutputIndexMapJSON = {"index_array": values, "offset": offset, "stride": stride}
    if explicit_bounds:
        output["index_array_bounds"] = ["-inf", "+inf"]
    body: IndexTransformJSON = {
        "input_inclusive_min": [0],
        "input_shape": [len(values)],
        "output": [output],
    }
    transform = IndexTransform.from_json(body)
    restored = IndexTransform.from_json(transform.to_json())
    points = np.arange(len(values)).reshape(-1, 1)
    expected = np.array([offset + stride * value for value in values]).reshape(-1, 1)
    np.testing.assert_array_equal(transform.apply_many(points), expected)
    np.testing.assert_array_equal(restored.apply_many(points), expected)
    # The standalone decoder follows the same omission/default policy.
    assert output_index_map_from_json(output) == transform.output[0]
