from __future__ import annotations

import numpy as np

from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.json import (
    IndexTransformJSON,
    index_domain_from_json,
    index_domain_to_json,
    index_transform_from_json,
    index_transform_to_json,
    output_index_map_from_json,
    output_index_map_to_json,
)
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core.transforms.transform import IndexTransform


class TestIndexDomainJSON:
    def test_roundtrip(self) -> None:
        domain = IndexDomain(inclusive_min=(2, 5), exclusive_max=(10, 20))
        json = index_domain_to_json(domain)
        assert json == {"input_inclusive_min": [2, 5], "input_exclusive_max": [10, 20]}
        restored = index_domain_from_json(json)
        assert restored == domain

    def test_with_labels(self) -> None:
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20), labels=("x", "y"))
        json = index_domain_to_json(domain)
        assert json["input_labels"] == ["x", "y"]
        restored = index_domain_from_json(json)
        assert restored.labels == ("x", "y")

    def test_without_labels(self) -> None:
        domain = IndexDomain.from_shape((5,))
        json = index_domain_to_json(domain)
        assert "input_labels" not in json
        restored = index_domain_from_json(json)
        assert restored.labels is None

    def test_zero_origin(self) -> None:
        domain = IndexDomain.from_shape((10, 20, 30))
        json = index_domain_to_json(domain)
        assert json == {
            "input_inclusive_min": [0, 0, 0],
            "input_exclusive_max": [10, 20, 30],
        }
        assert index_domain_from_json(json) == domain


class TestOutputIndexMapJSON:
    def test_constant(self) -> None:
        m = ConstantMap(offset=42)
        json = output_index_map_to_json(m)
        assert json == {"offset": 42}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ConstantMap)
        assert restored.offset == 42

    def test_constant_zero(self) -> None:
        m = ConstantMap(offset=0)
        json = output_index_map_to_json(m)
        assert json == {"offset": 0}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ConstantMap)
        assert restored.offset == 0

    def test_dimension(self) -> None:
        m = DimensionMap(input_dimension=1, offset=10, stride=3)
        json = output_index_map_to_json(m)
        assert json == {"offset": 10, "stride": 3, "input_dimension": 1}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, DimensionMap)
        assert restored.input_dimension == 1
        assert restored.offset == 10
        assert restored.stride == 3

    def test_dimension_stride_1_omitted(self) -> None:
        """stride=1 is the default and should be omitted from JSON."""
        m = DimensionMap(input_dimension=0)
        json = output_index_map_to_json(m)
        assert "stride" not in json
        assert json == {"offset": 0, "input_dimension": 0}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, DimensionMap)
        assert restored.stride == 1

    def test_array(self) -> None:
        arr = np.array([1, 5, 9], dtype=np.intp)
        m = ArrayMap(index_array=arr, offset=2, stride=3)
        json = output_index_map_to_json(m)
        assert json == {"offset": 2, "stride": 3, "index_array": [1, 5, 9]}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ArrayMap)
        np.testing.assert_array_equal(restored.index_array, arr)
        assert restored.offset == 2
        assert restored.stride == 3

    def test_array_stride_1_omitted(self) -> None:
        arr = np.array([0, 1, 2], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        json = output_index_map_to_json(m)
        assert "stride" not in json
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ArrayMap)
        assert restored.stride == 1

    def test_array_2d(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        json = output_index_map_to_json(m)
        assert json["index_array"] == [[1, 2], [3, 4]]
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ArrayMap)
        np.testing.assert_array_equal(restored.index_array, arr)


class TestIndexTransformJSON:
    def test_identity(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        json = index_transform_to_json(t)
        assert json == {
            "input_inclusive_min": [0, 0],
            "input_exclusive_max": [10, 20],
            "output": [
                {"offset": 0, "input_dimension": 0},
                {"offset": 0, "input_dimension": 1},
            ],
        }
        restored = index_transform_from_json(json)
        assert restored.domain == t.domain
        assert len(restored.output) == 2
        for orig, rest in zip(t.output, restored.output, strict=True):
            assert type(orig) is type(rest)

    def test_sliced(self) -> None:
        t = IndexTransform.from_shape((100,))[10:50:2]
        json = index_transform_to_json(t)
        restored = index_transform_from_json(json)
        assert restored.domain.shape == t.domain.shape
        assert isinstance(restored.output[0], DimensionMap)
        orig = t.output[0]
        assert isinstance(orig, DimensionMap)
        assert restored.output[0].offset == orig.offset
        assert restored.output[0].stride == orig.stride

    def test_with_constant(self) -> None:
        t = IndexTransform.from_shape((10, 20))[3]
        json = index_transform_to_json(t)
        restored = index_transform_from_json(json)
        assert isinstance(restored.output[0], ConstantMap)
        assert restored.output[0].offset == 3
        assert isinstance(restored.output[1], DimensionMap)

    def test_with_array(self) -> None:
        idx = np.array([1, 5, 9], dtype=np.intp)
        t = IndexTransform.from_shape((10, 20)).oindex[idx, :]
        json = index_transform_to_json(t)
        restored = index_transform_from_json(json)
        assert isinstance(restored.output[0], ArrayMap)
        np.testing.assert_array_equal(restored.output[0].index_array, idx)
        assert isinstance(restored.output[1], DimensionMap)

    def test_with_labels(self) -> None:
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20), labels=("x", "y"))
        t = IndexTransform.identity(domain)
        json = index_transform_to_json(t)
        assert json["input_labels"] == ["x", "y"]
        restored = index_transform_from_json(json)
        assert restored.domain.labels == ("x", "y")

    def test_tensorstore_compatible_format(self) -> None:
        """Verify the JSON matches TensorStore's format exactly."""
        json: IndexTransformJSON = {
            "input_inclusive_min": [0, 0, 0],
            "input_exclusive_max": [100, 200, 3],
            "input_labels": ["x", "y", "channel"],
            "output": [
                {"offset": 5},
                {"offset": 10, "stride": 2, "input_dimension": 1},
                {"offset": 0, "stride": 1, "index_array": [1, 2, 0]},
            ],
        }
        t = index_transform_from_json(json)
        assert t.domain.shape == (100, 200, 3)
        assert t.domain.labels == ("x", "y", "channel")
        assert isinstance(t.output[0], ConstantMap)
        assert t.output[0].offset == 5
        assert isinstance(t.output[1], DimensionMap)
        assert t.output[1].offset == 10
        assert t.output[1].stride == 2
        assert t.output[1].input_dimension == 1
        assert isinstance(t.output[2], ArrayMap)
        np.testing.assert_array_equal(t.output[2].index_array, [1, 2, 0])

        # Roundtrip
        json_rt = index_transform_to_json(t)
        t_rt = index_transform_from_json(json_rt)
        assert t_rt.domain == t.domain
