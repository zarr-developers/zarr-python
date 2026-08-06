from __future__ import annotations

import numpy as np
import pytest

from zarr_indexing.composition import compose
from zarr_indexing.domain import IndexDomain
from zarr_indexing.errors import BoundsCheckError
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

    def test_dimension_inner_constant_outer_rejects_affine_overflow(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((1,)),
            output=(ConstantMap(offset=2**62),),
        )
        inner = IndexTransform(
            domain=IndexDomain((2**62,), (2**62 + 1,)),
            output=(DimensionMap(input_dimension=0, stride=4),),
        )

        with pytest.raises(OverflowError, match="outside np.intp"):
            compose(outer, inner)

    def test_dimension_inner_dimension_outer(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
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
        arr = np.array([0, 1, 2], dtype=np.intp)
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

    def test_array_inner_constant_outer_rejects_affine_overflow(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((1,)),
            output=(ConstantMap(offset=0),),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((1,)),
            output=(ArrayMap(np.array([2**62], dtype=np.intp), stride=4),),
        )

        with pytest.raises(OverflowError, match="outside np.intp"):
            compose(outer, inner)

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


class TestComposeInnerDomainValidation:
    @pytest.mark.parametrize(
        ("outer_map", "inner_lower"),
        [
            (ConstantMap(offset=10), 10),
            (ConstantMap(offset=14), 10),
            (DimensionMap(input_dimension=0, offset=10, stride=1), 10),
            (DimensionMap(input_dimension=0, offset=14, stride=-1), 10),
            (DimensionMap(input_dimension=0, offset=10**100, stride=1), 10**100),
        ],
        ids=["lower-bound", "upper-bound", "forward", "reverse", "arbitrary-integer"],
    )
    def test_valid_constant_and_affine_boundaries(
        self, outer_map: ConstantMap | DimensionMap, inner_lower: int
    ) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(outer_map,),
        )
        inner = IndexTransform(
            domain=IndexDomain((inner_lower,), (inner_lower + 5,)),
            output=(DimensionMap(input_dimension=0),),
        )

        result = compose(outer, inner)

        assert result.domain == outer.domain
        _assert_composes_pointwise(outer, inner)

    def test_rejects_constant_outer_coordinate_outside_inner_domain(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((1,)),
            output=(ConstantMap(offset=2),),
        )
        inner = IndexTransform.identity(IndexDomain((0,), (2,)))

        with pytest.raises(BoundsCheckError, match="outside.*inner.*domain"):
            compose(outer, inner)

    def test_rejects_affine_outer_range_crossing_inner_domain(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain((10**100,), (10**100 + 3,)),
            output=(DimensionMap(input_dimension=0),),
        )
        inner = IndexTransform.identity(IndexDomain((10**100,), (10**100 + 2,)))

        with pytest.raises(BoundsCheckError, match="outside.*inner.*domain"):
            compose(outer, inner)

    def test_empty_outer_domain_does_not_evaluate_nonexistent_points(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((0,)),
            output=(ConstantMap(offset=99),),
        )
        inner = IndexTransform.identity(IndexDomain((0,), (2,)))

        result = compose(outer, inner)

        assert result.domain.shape == (0,)
        assert result.output == (ConstantMap(offset=99),)


class TestComposeMultidimensionalArrayInner:
    def test_identity_gathers_a_two_dimensional_array_map(self) -> None:
        inner = IndexTransform(
            domain=IndexDomain.from_shape((2, 3)),
            output=(
                ArrayMap(
                    index_array=np.array([[11, 13, 17], [19, 23, 29]], dtype=np.intp),
                    offset=5,
                    stride=2,
                ),
            ),
        )
        outer = IndexTransform.identity(inner.domain)

        result = compose(outer, inner)

        assert result == inner
        _assert_composes_pointwise(outer, inner)

    def test_mixed_affine_and_constant_outputs_gather_with_metadata(self) -> None:
        inner = IndexTransform(
            domain=IndexDomain((4, 8), (7, 10)),
            output=(
                ArrayMap(
                    index_array=np.array([[2, 3], [5, 7], [11, 13]], dtype=np.intp),
                    offset=-3,
                    stride=4,
                ),
            ),
        )
        outer = IndexTransform(
            domain=IndexDomain((-2,), (0,)),
            output=(
                DimensionMap(input_dimension=0, offset=7, stride=1),
                ConstantMap(offset=9),
            ),
        )

        result = compose(outer, inner)

        assert result.domain == outer.domain
        assert len(result.output) == 1
        array_map = result.output[0]
        assert isinstance(array_map, ArrayMap)
        assert array_map.index_array.shape == (2,)
        assert array_map.offset == -3
        assert array_map.stride == 4
        np.testing.assert_array_equal(array_map.index_array, np.array([7, 13], dtype=np.intp))
        _assert_composes_pointwise(outer, inner)

    def test_out_of_bounds_affine_output_is_rejected_before_gather(self) -> None:
        outer = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(
                DimensionMap(input_dimension=0, offset=1, stride=1),
                ConstantMap(offset=0),
            ),
        )
        inner = IndexTransform(
            domain=IndexDomain.from_shape((3, 2)),
            output=(ArrayMap(np.arange(6, dtype=np.intp).reshape(3, 2)),),
        )

        with pytest.raises(BoundsCheckError, match="outside.*inner.*domain"):
            compose(outer, inner)

    def test_rank_zero_array_map_composition(self) -> None:
        domain = IndexDomain((), ())
        outer = IndexTransform.identity(domain)
        inner = IndexTransform(
            domain=domain,
            output=(ArrayMap(np.array(7, dtype=np.intp), offset=2, stride=3),),
        )

        result = compose(outer, inner)

        assert result.domain == domain
        assert result.output == (ConstantMap(offset=23),)


class TestComposeChain:
    def test_three_transforms(self) -> None:
        a = IndexTransform.from_shape((100,))
        b = IndexTransform(
            domain=IndexDomain.from_shape((100,)),
            output=(DimensionMap(input_dimension=0, offset=10, stride=1),),
        )
        c = IndexTransform(
            domain=IndexDomain.from_shape((110,)),
            output=(DimensionMap(input_dimension=0, offset=5, stride=2),),
        )
        bc = compose(b, c)
        abc = compose(a, bc)
        assert isinstance(abc.output[0], DimensionMap)
        assert abc.output[0].offset == 25
        assert abc.output[0].stride == 2


def test_composing_an_inner_array_with_a_broadcast_axis_wider_than_one_cell() -> None:
    """A non-dependency axis is a singleton that broadcasts, so its coordinate is 0.

    Indexing it by the raw intermediate coordinate walked off the end of an axis
    the array only has one entry for.
    """
    outer = IndexTransform(
        domain=IndexDomain(inclusive_min=(-2,), exclusive_max=(0,)),
        output=(ConstantMap(offset=1), ConstantMap(offset=0)),
    )
    inner = IndexTransform(
        domain=IndexDomain(inclusive_min=(0, 0), exclusive_max=(2, 1)),
        output=(ArrayMap(index_array=np.array([[13]], dtype=np.intp), offset=-2, stride=2),),
    )
    composed = compose(outer, inner)
    assert composed.output[0] == ConstantMap(offset=24)


def test_composing_a_one_dimensional_inner_array_under_a_higher_rank_outer() -> None:
    """The shortcut gated on the output rank but sized by the input rank.

    A rank-2 outer therefore built a rank-1 array for a rank-2 domain, which the
    engine's own invariant then rejected.
    """
    outer = IndexTransform(
        domain=IndexDomain.from_shape((2, 3)),
        output=(DimensionMap(input_dimension=0, offset=1, stride=1),),
    )
    inner = IndexTransform(
        domain=IndexDomain.from_shape((4,)),
        output=(ArrayMap(index_array=np.array([7, 3, 5, 1], dtype=np.intp)),),
    )
    composed = compose(outer, inner)
    array_map = composed.output[0]
    assert isinstance(array_map, ArrayMap)
    assert array_map.index_array.shape == (2, 1)
    np.testing.assert_array_equal(array_map.index_array, np.array([[3], [5]]))


def test_composing_out_of_the_inner_domain_is_refused() -> None:
    """An intermediate outside the inner domain wrapped NumPy-style and read a cell."""
    outer = IndexTransform(
        domain=IndexDomain.from_shape((1,)),
        output=(ConstantMap(offset=-3),),
    )
    inner = IndexTransform(
        domain=IndexDomain.from_shape((2,)),
        output=(ArrayMap(index_array=np.array([7, 3], dtype=np.intp)),),
    )
    with pytest.raises(BoundsCheckError, match="outside.*inner.*domain"):
        compose(outer, inner)
