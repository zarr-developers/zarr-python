from __future__ import annotations

import numpy as np
import pytest

from zarr_transforms.domain import IndexDomain
from zarr_transforms.json import (
    IndexTransformJSON,
    index_domain_from_json,
    index_domain_to_json,
    index_transform_from_json,
    index_transform_to_json,
    output_index_map_from_json,
    output_index_map_to_json,
)
from zarr_transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_transforms.transform import IndexTransform


def _maps_equal(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, ConstantMap):
        assert isinstance(b, ConstantMap)
        return a.offset == b.offset
    if isinstance(a, DimensionMap):
        assert isinstance(b, DimensionMap)
        return (a.input_dimension, a.offset, a.stride) == (b.input_dimension, b.offset, b.stride)
    assert isinstance(a, ArrayMap)
    assert isinstance(b, ArrayMap)
    return (
        a.offset == b.offset
        and a.stride == b.stride
        and a.input_dimension == b.input_dimension
        and np.array_equal(a.index_array, b.index_array)
    )


def _transforms_equal(a: IndexTransform, b: IndexTransform) -> bool:
    """Structural equality that compares `ArrayMap` index arrays element-wise
    (`IndexTransform`'s dataclass `__eq__` cannot, as numpy `==` is ambiguous)."""
    return (
        a.domain == b.domain
        and len(a.output) == len(b.output)
        and all(_maps_equal(x, y) for x, y in zip(a.output, b.output, strict=True))
    )


class TestIndexDomainJSON:
    def test_roundtrip(self) -> None:
        domain = IndexDomain(inclusive_min=(2, 5), exclusive_max=(10, 20))
        json = index_domain_to_json(domain)
        assert json == {
            "input_inclusive_min": [2, 5],
            "input_exclusive_max": [10, 20],
            "input_labels": ["", ""],
        }
        restored = index_domain_from_json(json)
        assert restored == domain

    def test_with_labels(self) -> None:
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20), labels=("x", "y"))
        json = index_domain_to_json(domain)
        assert json["input_labels"] == ["x", "y"]
        restored = index_domain_from_json(json)
        assert restored.labels == ("x", "y")

    def test_without_labels_emits_empty_and_round_trips_to_none(self) -> None:
        domain = IndexDomain.from_shape((5,))
        json = index_domain_to_json(domain)
        # Canonical form always writes labels; an unlabeled domain gets [""]*rank.
        assert json["input_labels"] == [""]
        restored = index_domain_from_json(json)
        assert restored.labels is None

    def test_zero_origin(self) -> None:
        domain = IndexDomain.from_shape((10, 20, 30))
        json = index_domain_to_json(domain)
        assert json == {
            "input_inclusive_min": [0, 0, 0],
            "input_exclusive_max": [10, 20, 30],
            "input_labels": ["", "", ""],
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

    def test_dimension_stride_1_written(self) -> None:
        """Canonical form writes stride even at its default of 1."""
        m = DimensionMap(input_dimension=0)
        json = output_index_map_to_json(m)
        assert json == {"offset": 0, "stride": 1, "input_dimension": 0}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, DimensionMap)
        assert restored.stride == 1

    def test_array(self) -> None:
        arr = np.array([1, 5, 9], dtype=np.intp)
        m = ArrayMap(index_array=arr, offset=2, stride=3)
        json = output_index_map_to_json(m)
        # Canonical: stride/offset present, index_array_bounds present, and
        # no input_dimension (ndsel/TensorStore reject it beside index_array).
        assert json == {
            "offset": 2,
            "stride": 3,
            "index_array": [1, 5, 9],
            "index_array_bounds": ["-inf", "+inf"],
        }
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ArrayMap)
        np.testing.assert_array_equal(restored.index_array, arr)
        assert restored.offset == 2
        assert restored.stride == 3

    def test_array_stride_1_written(self) -> None:
        arr = np.array([0, 1, 2], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        json = output_index_map_to_json(m)
        assert json["stride"] == 1
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

    def test_degenerate_singleton_array_collapses_to_constant(self) -> None:
        """An all-singleton index_array selects one coordinate -> constant map."""
        m = ArrayMap(index_array=np.array([[4]], dtype=np.intp), offset=1, stride=2)
        json = output_index_map_to_json(m)
        assert json == {"offset": 1 + 2 * 4}
        restored = output_index_map_from_json(json)
        assert isinstance(restored, ConstantMap)
        assert restored.offset == 9


class TestIndexTransformJSON:
    def test_identity(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        json = index_transform_to_json(t)
        assert json == {
            "input_rank": 2,
            "input_inclusive_min": [0, 0],
            "input_exclusive_max": [10, 20],
            "input_labels": ["", ""],
            "output": [
                {"offset": 0, "stride": 1, "input_dimension": 0},
                {"offset": 0, "stride": 1, "input_dimension": 1},
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
        # The oindex array must not carry input_dimension on the wire.
        assert "input_dimension" not in json["output"][0]
        restored = index_transform_from_json(json)
        assert isinstance(restored.output[0], ArrayMap)
        # Orthogonal arrays are normalized to full input rank with a singleton
        # axis on the dimension they do not vary over.
        assert restored.output[0].index_array.shape == (3, 1)
        np.testing.assert_array_equal(restored.output[0].index_array, idx.reshape(3, 1))
        # input_dimension is reconstructed from the sole non-singleton axis.
        assert restored.output[0].input_dimension == 0
        assert isinstance(restored.output[1], DimensionMap)

    def test_roundtrip_preserves_singleton_axes(self) -> None:
        """Full-rank orthogonal arrays keep their singleton axes across JSON."""
        t = IndexTransform.from_shape((10, 20)).oindex[np.array([1, 3]), np.array([2, 4, 6])]
        restored = index_transform_from_json(index_transform_to_json(t))
        orig0, orig1 = t.output[0], t.output[1]
        rest0, rest1 = restored.output[0], restored.output[1]
        assert isinstance(orig0, ArrayMap)
        assert isinstance(orig1, ArrayMap)
        assert isinstance(rest0, ArrayMap)
        assert isinstance(rest1, ArrayMap)
        assert rest0.index_array.shape == (2, 1)
        assert rest1.index_array.shape == (1, 3)
        np.testing.assert_array_equal(rest0.index_array, orig0.index_array)
        np.testing.assert_array_equal(rest1.index_array, orig1.index_array)
        # Distinct, exclusively-owned axes -> reconstructed as orthogonal.
        assert rest0.input_dimension == 0
        assert rest1.input_dimension == 1

    def test_with_labels(self) -> None:
        domain = IndexDomain(inclusive_min=(0, 0), exclusive_max=(10, 20), labels=("x", "y"))
        t = IndexTransform.identity(domain)
        json = index_transform_to_json(t)
        assert json["input_labels"] == ["x", "y"]
        restored = index_transform_from_json(json)
        assert restored.domain.labels == ("x", "y")

    def test_tensorstore_compatible_format(self) -> None:
        """A canonical body loads and round-trips through the engine layer."""
        json: IndexTransformJSON = {
            "input_rank": 3,
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


class TestCanonicalRoundTrips:
    """Round-trip `transform == from(to(transform))`, up to the documented
    degenerate-collapse (all-singleton ArrayMap -> ConstantMap)."""

    def test_oindex_multi_axis(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30)).oindex[np.array([1, 3]), :, np.array([2, 4, 6])]
        rt = index_transform_from_json(index_transform_to_json(t))
        assert _transforms_equal(rt, t)

    def test_oindex_with_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20))[2:8].oindex[np.array([3, 5, 7]), :]
        rt = index_transform_from_json(index_transform_to_json(t))
        assert _transforms_equal(rt, t)

    def test_vindex(self) -> None:
        t = IndexTransform.from_shape((10, 20)).vindex[np.array([1, 3, 5]), np.array([2, 4, 6])]
        rt = index_transform_from_json(index_transform_to_json(t))
        assert _transforms_equal(rt, t)

    def test_vindex_with_residual_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30)).vindex[np.array([1, 3]), np.array([2, 4]), :]
        rt = index_transform_from_json(index_transform_to_json(t))
        assert _transforms_equal(rt, t)

    def test_length1_degenerate_oindex_collapses(self) -> None:
        """A length-1 oindex array becomes an all-singleton ArrayMap; the JSON
        round-trip collapses it to a ConstantMap (behaviorally identical)."""
        t = IndexTransform.from_shape((10, 20)).oindex[np.array([7]), :]
        m = t.output[0]
        assert isinstance(m, ArrayMap)
        assert m.index_array.size == 1

        rt = index_transform_from_json(index_transform_to_json(t))
        # The degenerate array collapsed to a constant selecting the same cell.
        rm = rt.output[0]
        assert isinstance(rm, ConstantMap)
        assert rm.offset == 7
        # The size-1 input dimension survives, unconsumed, in the domain.
        assert rt.domain == t.domain

    def test_slices_and_constants(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30))[2:8:2, 5, :]
        rt = index_transform_from_json(index_transform_to_json(t))
        assert _transforms_equal(rt, t)


def test_infinite_bound_rejected_on_lowering() -> None:
    body: IndexTransformJSON = {
        "input_rank": 1,
        "input_inclusive_min": [0],
        "input_exclusive_max": [["+inf"]],
        "input_labels": [""],
        "output": [{"offset": 0, "stride": 1, "input_dimension": 0}],
    }
    with pytest.raises(ValueError, match="infinite"):
        index_transform_from_json(body)
