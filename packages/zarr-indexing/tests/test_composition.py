from __future__ import annotations

import numpy as np
import pytest

from zarr_indexing.composition import compose
from zarr_indexing.domain import IndexDomain
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import IndexTransform


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


def _storage_of(transform: IndexTransform, point: tuple[int, ...]) -> tuple[int, ...]:
    """The storage coordinates a transform assigns to one input point.

    The point is given in the transform's own domain coordinates; an index array
    carries the full input rank, singleton on the axes it does not vary over, and
    is addressed positionally from the domain origin.
    """
    coords: list[int] = []
    for m in transform.output:
        if isinstance(m, ConstantMap):
            coords.append(m.offset)
        elif isinstance(m, DimensionMap):
            coords.append(m.offset + m.stride * point[m.input_dimension])
        else:
            origin = transform.domain.inclusive_min
            idx = tuple(
                0 if m.index_array.shape[axis] == 1 else point[axis] - origin[axis]
                for axis in range(m.index_array.ndim)
            )
            coords.append(m.offset + m.stride * int(m.index_array[idx]))
    return tuple(coords)


def _assert_composes_pointwise(outer: IndexTransform, inner: IndexTransform) -> None:
    """`compose(outer, inner)` must agree with running the two in sequence."""
    composed = compose(outer, inner)
    assert composed.domain == outer.domain
    lo = outer.domain.inclusive_min
    hi = outer.domain.exclusive_max
    for coord in np.ndindex(*outer.domain.shape):
        point = tuple(int(c) + int(o) for c, o in zip(coord, lo, strict=True))
        assert all(point[d] < hi[d] for d in range(len(hi)))
        intermediate = _storage_of(outer, point)
        assert _storage_of(composed, point) == _storage_of(inner, intermediate)


class TestComposeOverANonZeroOriginDomain:
    """The outer domain need not start at 0 — a step-1 slice preserves its
    literal bounds and a negative step produces a negative origin — so the inner
    map has to be evaluated over the outer domain's real range."""

    def test_array_inner_sliced_outer(self) -> None:
        inner = IndexTransform.from_shape((10,)).oindex[np.array([3, 1, 4, 1, 5])]
        outer = IndexTransform.identity(IndexDomain.from_shape((5,)))[1:4]
        assert outer.domain.inclusive_min == (1,)
        result = compose(outer, inner)
        assert isinstance(result.output[0], ArrayMap)
        np.testing.assert_array_equal(
            result.output[0].index_array, np.array([1, 4, 1], dtype=np.intp)
        )
        _assert_composes_pointwise(outer, inner)

    def test_array_inner_reversed_outer(self) -> None:
        inner = IndexTransform.from_shape((10,)).oindex[np.array([3, 1, 4, 1, 5])]
        outer = IndexTransform.identity(IndexDomain.from_shape((5,)))[::-1]
        assert outer.domain.inclusive_min == (-4,)
        result = compose(outer, inner)
        assert isinstance(result.output[0], ArrayMap)
        np.testing.assert_array_equal(
            result.output[0].index_array, np.array([5, 1, 4, 1, 3], dtype=np.intp)
        )
        _assert_composes_pointwise(outer, inner)

    def test_array_inner_strided_outer(self) -> None:
        inner = IndexTransform.from_shape((10,)).oindex[np.array([3, 1, 4, 1, 5, 9])]
        outer = IndexTransform.identity(IndexDomain.from_shape((6,)))[1::2]
        _assert_composes_pointwise(outer, inner)

    def test_array_inner_translated_outer(self) -> None:
        inner = IndexTransform.from_shape((10,)).oindex[np.array([3, 1, 4, 1, 5])]
        outer = IndexTransform.identity(IndexDomain.from_shape((5,))).translate_domain_to((-2,))
        _assert_composes_pointwise(outer, inner)

    def test_array_inner_array_outer_over_a_shifted_domain(self) -> None:
        inner = IndexTransform.from_shape((10,)).oindex[np.array([3, 1, 4, 1, 5])]
        outer = IndexTransform.from_shape((5,)).oindex[np.array([4, 0, 2])]
        _assert_composes_pointwise(outer, inner)

    def test_array_inner_constant_outer_over_a_shifted_domain(self) -> None:
        inner = IndexTransform.from_shape((10,)).oindex[np.array([3, 1, 4, 1, 5])]
        outer = IndexTransform(
            domain=IndexDomain.from_shape((4,)),
            output=(ConstantMap(offset=3),),
        )
        _assert_composes_pointwise(outer, inner)


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
