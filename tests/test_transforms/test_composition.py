from __future__ import annotations

import numpy as np
import pytest

from zarr.core.transforms.composition import compose
from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core.transforms.transform import IndexTransform


class TestComposeConstantInner:
    """Inner = constant. Result is always constant."""

    def test_constant_inner_any_outer(self) -> None:
        outer = IndexTransform.from_shape((5,))
        inner = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(ConstantMap(offset=42),),
        )
        result = compose(outer, inner)
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == 42


class TestComposeDimensionInner:
    """Inner = DimensionMap."""

    def test_dimension_inner_constant_outer(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ConstantMap(offset=5),),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(DimensionMap(input_dimension=0, offset=10, stride=3),),
        )
        result = compose(outer, inner)
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == 25

    def test_dimension_inner_dimension_outer(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(DimensionMap(input_dimension=0, offset=5, stride=2),),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(DimensionMap(input_dimension=0, offset=10, stride=3),),
        )
        result = compose(outer, inner)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 25
        assert result.output[0].stride == 6
        assert result.output[0].input_dimension == 0

    def test_dimension_inner_array_outer(self) -> None:
        arr = np.array([0, 2, 4], dtype=np.intp)
        outer = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=arr, offset=5, stride=2),),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(DimensionMap(input_dimension=0, offset=10, stride=3),),
        )
        result = compose(outer, inner)
        assert isinstance(result.output[0], ArrayMap)
        assert result.output[0].offset == 25
        assert result.output[0].stride == 6
        np.testing.assert_array_equal(result.output[0].index_array, arr)


class TestComposeArrayInner:
    """Inner = ArrayMap."""

    def test_array_inner_constant_outer(self) -> None:
        inner_arr = np.array([10, 20, 30], dtype=np.intp)
        outer = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(ConstantMap(offset=1),),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=inner_arr, offset=0, stride=1),),
        )
        result = compose(outer, inner)
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == 20

    def test_array_inner_array_outer(self) -> None:
        outer_arr = np.array([0, 2, 1], dtype=np.intp)
        inner_arr = np.array([10, 20, 30], dtype=np.intp)
        outer = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=outer_arr, offset=0, stride=1),),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=inner_arr, offset=0, stride=1),),
        )
        result = compose(outer, inner)
        assert isinstance(result.output[0], ArrayMap)
        expected = np.array([10, 30, 20], dtype=np.intp)
        np.testing.assert_array_equal(result.output[0].index_array, expected)


class TestComposeMultiDim:
    def test_2d_identity_compose(self) -> None:
        a = IndexTransform.from_shape((10, 20))
        b = IndexTransform.from_shape((10, 20))
        result = compose(a, b)
        assert result.domain.shape == (10, 20)
        for i in range(2):
            m = result.output[i]
            assert isinstance(m, DimensionMap)
            assert m.input_dimension == i
            assert m.offset == 0
            assert m.stride == 1

    def test_mixed_map_types(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(
                ConstantMap(offset=5),
                DimensionMap(input_dimension=0, offset=0, stride=1),
            ),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((10, 10)),
            output=(
                DimensionMap(input_dimension=0, offset=2, stride=3),
                DimensionMap(input_dimension=1, offset=0, stride=1),
            ),
        )
        result = compose(outer, inner)
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == 17
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].input_dimension == 0
        assert result.output[1].offset == 0
        assert result.output[1].stride == 1

    def test_rank_mismatch_raises(self) -> None:
        outer = IndexTransform.from_shape((10,))
        inner = IndexTransform.from_shape((10, 20))
        with pytest.raises(ValueError, match="rank"):
            compose(outer, inner)


class TestComposeChain:
    def test_three_transforms(self) -> None:
        a = IndexTransform.from_shape((100,))
        b = IndexTransform(
            domain=IndexDomain.from_shape((100,)),
            output=(DimensionMap(input_dimension=0, offset=10, stride=1),),
        )
        c = IndexTransform(
            domain=IndexDomain.from_shape((100,)),
            output=(DimensionMap(input_dimension=0, offset=5, stride=2),),
        )
        bc = compose(b, c)
        abc = compose(a, bc)
        assert isinstance(abc.output[0], DimensionMap)
        assert abc.output[0].offset == 25
        assert abc.output[0].stride == 2
