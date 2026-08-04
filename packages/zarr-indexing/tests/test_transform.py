from __future__ import annotations

import numpy as np
import pytest

from zarr_indexing.domain import IndexDomain
from zarr_indexing.errors import BoundsCheckError
from zarr_indexing.lazy_array import LazyArray
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr_indexing.transform import (
    IndexTransform,
    array_map_dependent_axis,
    selection_to_transform,
)


class TestIndexTransformConstruction:
    def test_from_shape(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        assert t.input_rank == 2
        assert t.output_rank == 2
        assert t.domain.shape == (10, 20)
        assert t.domain.origin == (0, 0)
        for i, m in enumerate(t.output):
            assert isinstance(m, DimensionMap)
            assert m.input_dimension == i
            assert m.offset == 0
            assert m.stride == 1

    def test_identity(self) -> None:
        domain = IndexDomain(inclusive_min=(5,), exclusive_max=(15,))
        t = IndexTransform.identity(domain)
        assert t.input_rank == 1
        assert t.output_rank == 1
        assert t.domain == domain
        assert isinstance(t.output[0], DimensionMap)
        assert t.output[0].input_dimension == 0

    def test_from_shape_0d(self) -> None:
        t = IndexTransform.from_shape(())
        assert t.input_rank == 0
        assert t.output_rank == 0
        assert t.domain.shape == ()

    def test_custom_output_maps(self) -> None:
        domain = IndexDomain.from_shape((10,))
        maps = (ConstantMap(offset=42), DimensionMap(input_dimension=0, offset=5, stride=2))
        t = IndexTransform(domain=domain, output=maps)
        assert t.input_rank == 1
        assert t.output_rank == 2

    def test_validation_input_dimension_out_of_range(self) -> None:
        domain = IndexDomain.from_shape((10,))
        maps = (DimensionMap(input_dimension=5),)
        with pytest.raises(ValueError, match="input_dimension"):
            IndexTransform(domain=domain, output=maps)


class TestIndexTransformApply:
    @pytest.mark.parametrize(
        ("transform", "points", "expected"),
        [
            pytest.param(
                IndexTransform.identity(IndexDomain((-2, 5), (1, 8))),
                np.array([-2, 7], dtype=np.int64),
                np.array([-2, 7], dtype=np.intp),
                id="identity-negative-and-nonzero-origins",
            ),
            pytest.param(
                IndexTransform(
                    domain=IndexDomain((3,), (6,)),
                    output=(
                        ConstantMap(41),
                        DimensionMap(0, offset=10, stride=-2),
                    ),
                ),
                np.array([[3], [5]], dtype=np.int16),
                np.array([[41, 4], [41, 0]], dtype=np.intp),
                id="constant-and-negative-stride",
            ),
            pytest.param(
                IndexTransform(
                    domain=IndexDomain((-2, 5), (1, 8)),
                    output=(
                        ArrayMap(
                            np.array([[7], [11], [13]], dtype=np.intp),
                            offset=-1,
                            stride=2,
                            input_dimension=0,
                        ),
                    ),
                ),
                np.array(
                    [
                        [[-2, 5], [-1, 7]],
                        [[0, 6], [-2, 6]],
                    ],
                    dtype=np.intp,
                ),
                np.array([[[13], [21]], [[25], [13]]], dtype=np.intp),
                id="array-map-singleton-broadcast-multidimensional-batch",
            ),
            pytest.param(
                IndexTransform(IndexDomain((), ()), (ConstantMap(42),)),
                np.empty((2, 0), dtype=np.intp),
                np.array([[42], [42]], dtype=np.intp),
                id="rank-zero-input",
            ),
            pytest.param(
                IndexTransform(IndexDomain((-1,), (2,)), ()),
                np.array([[-1], [1]], dtype=np.intp),
                np.empty((2, 0), dtype=np.intp),
                id="rank-zero-output",
            ),
            pytest.param(
                IndexTransform.identity(IndexDomain.from_shape((2,))),
                np.empty((0, 1), dtype=np.intp),
                np.empty((0, 1), dtype=np.intp),
                id="empty-batch",
            ),
        ],
    )
    def test_apply_many_maps_integer_point_batches(
        self,
        transform: IndexTransform,
        points: np.ndarray,
        expected: np.ndarray,
    ) -> None:
        result = transform.apply_many(points)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.dtype(np.intp)
        assert result.flags.owndata

    def test_apply_maps_one_point(self) -> None:
        transform = IndexTransform(
            IndexDomain((-2, 4), (1, 7)),
            (
                DimensionMap(1, offset=3, stride=-1),
                DimensionMap(0, offset=2, stride=2),
            ),
        )

        assert transform.apply((-1, 6)) == (-3, 0)

    def test_apply_rejects_a_point_with_the_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match=r"point must have shape \(2,\), got \(1,\)"):
            IndexTransform.from_shape((2, 3)).apply((1,))

    def test_apply_rejects_an_explicitly_floating_rank_zero_point(self) -> None:
        transform = IndexTransform(IndexDomain((), ()), ())
        with pytest.raises(TypeError, match="integer dtype"):
            transform.apply(np.array([], dtype=np.float64))

    @pytest.mark.parametrize(
        "points",
        [
            pytest.param(np.array(1, dtype=np.intp), id="no-coordinate-axis"),
            pytest.param(np.zeros((4, 3), dtype=np.intp), id="wrong-trailing-size"),
        ],
    )
    def test_apply_many_rejects_an_invalid_coordinate_axis(self, points: np.ndarray) -> None:
        with pytest.raises(ValueError, match="trailing coordinate axis"):
            IndexTransform.from_shape((2, 3)).apply_many(points)

    @pytest.mark.parametrize(
        "points",
        [
            pytest.param(np.array([[True]], dtype=np.bool_), id="bool"),
            pytest.param(np.array([[1.0]], dtype=np.float64), id="float"),
            pytest.param(np.array([["1"]], dtype=np.str_), id="string"),
            pytest.param(np.array([[1]], dtype=object), id="object"),
        ],
    )
    def test_apply_many_rejects_non_integer_coordinates(self, points: np.ndarray) -> None:
        with pytest.raises(TypeError, match="integer dtype"):
            IndexTransform.from_shape((2,)).apply_many(points)

    def test_apply_many_reports_the_first_out_of_bounds_coordinate(self) -> None:
        transform = IndexTransform.identity(IndexDomain((-2, 10), (2, 13)))
        points = np.array(
            [
                [[-2, 10], [-1, 20]],
                [[9, 11], [0, 12]],
            ],
            dtype=np.intp,
        )

        with pytest.raises(BoundsCheckError) as error:
            transform.apply_many(points)

        assert str(error.value) == (
            "point at batch position (0, 1) has input dimension 1 coordinate 20 outside [10, 13)"
        )

    @pytest.mark.parametrize(
        "beyond_intp",
        [
            pytest.param(int(np.iinfo(np.intp).max) + 1, id="first-uint64-coordinate"),
            pytest.param(int(np.iinfo(np.uint64).max), id="maximum-uint64-coordinate"),
        ],
    )
    def test_apply_many_maps_large_literal_coordinates_exactly(self, beyond_intp: int) -> None:
        transform = IndexTransform(
            IndexDomain((beyond_intp,), (beyond_intp + 1,)),
            (DimensionMap(0, offset=-beyond_intp),),
        )

        result = transform.apply_many(np.array([[beyond_intp]], dtype=np.uint64))

        np.testing.assert_array_equal(result, np.array([[0]], dtype=np.intp))
        assert result.dtype == np.dtype(np.intp)
        assert result.flags.owndata

    @pytest.mark.parametrize(
        ("transform", "points"),
        [
            pytest.param(
                IndexTransform(
                    IndexDomain.from_shape((1,)),
                    (ConstantMap(np.iinfo(np.intp).max + 1),),
                ),
                [[0]],
                id="constant",
            ),
            pytest.param(
                IndexTransform(
                    IndexDomain.from_shape((2,)),
                    (DimensionMap(0, offset=np.iinfo(np.intp).max),),
                ),
                [[1]],
                id="dimension",
            ),
            pytest.param(
                IndexTransform(
                    IndexDomain.from_shape((1,)),
                    (
                        ArrayMap(
                            np.array([1], dtype=np.intp),
                            offset=np.iinfo(np.intp).max,
                        ),
                    ),
                ),
                [[0]],
                id="array",
            ),
            pytest.param(
                IndexTransform.identity(
                    IndexDomain(
                        (int(np.iinfo(np.intp).max) + 1,),
                        (int(np.iinfo(np.intp).max) + 2,),
                    )
                ),
                np.array([[int(np.iinfo(np.intp).max) + 1]], dtype=np.uint64),
                id="large-input-identity",
            ),
        ],
    )
    def test_apply_many_rejects_mapped_coordinates_outside_intp(
        self, transform: IndexTransform, points: list[list[int]] | np.ndarray
    ) -> None:
        with pytest.raises(OverflowError, match="output coordinate.*np.intp"):
            transform.apply_many(points)

    def test_apply_many_rejects_affine_coordinate_overflow(self) -> None:
        transform = IndexTransform(
            domain=IndexDomain.from_shape((1,)),
            output=(ArrayMap(np.array([2**62], dtype=np.intp), stride=4),),
        )
        with pytest.raises(OverflowError, match="outside np.intp"):
            transform.apply_many(np.array([[0]], dtype=np.intp))

    def test_apply_rejects_affine_coordinate_overflow(self) -> None:
        transform = IndexTransform(
            domain=IndexDomain.from_shape((1,)),
            output=(ArrayMap(np.array([2**62], dtype=np.intp), stride=4),),
        )
        with pytest.raises(OverflowError, match="outside np.intp"):
            transform.apply((0,))


class TestIndexTransformInverted:
    @pytest.mark.parametrize(
        ("transform", "points"),
        [
            pytest.param(
                IndexTransform(
                    IndexDomain((-3, 4), (1, 7)),
                    (
                        DimensionMap(1, offset=10),
                        DimensionMap(0, offset=2, stride=-1),
                    ),
                ),
                np.array([[-3, 4], [0, 6]], dtype=np.intp),
                id="permutation-translation-reversal-nonzero-origin",
            ),
            pytest.param(
                IndexTransform(
                    IndexDomain((5, -2), (8, -1)),
                    (DimensionMap(0, offset=3), ConstantMap(99)),
                ),
                np.array([[5, -2], [7, -2]], dtype=np.intp),
                id="constant-and-unreferenced-singleton",
            ),
            pytest.param(
                IndexTransform(IndexDomain((), ()), ()),
                np.empty((1, 0), dtype=np.intp),
                id="rank-zero",
            ),
        ],
    )
    def test_inverted_round_trips_points(
        self, transform: IndexTransform, points: np.ndarray
    ) -> None:
        inverse = transform.inverted()
        mapped = transform.apply_many(points)

        np.testing.assert_array_equal(inverse.apply_many(mapped), points)
        assert inverse.apply(transform.apply(tuple(points[0]))) == tuple(points[0])
        assert inverse.inverted() == transform

    def test_inverted_rejects_unequal_ranks(self) -> None:
        transform = IndexTransform(IndexDomain.from_shape((2,)), (ConstantMap(1), ConstantMap(2)))
        with pytest.raises(ValueError, match="input rank must equal output rank"):
            transform.inverted()

    def test_inverted_rejects_an_array_map(self) -> None:
        transform = IndexTransform(
            IndexDomain.from_shape((2,)),
            (ArrayMap(np.array([1, 0], dtype=np.intp)),),
        )
        with pytest.raises(ValueError, match="ArrayMap"):
            transform.inverted()

    def test_inverted_rejects_a_non_unit_stride(self) -> None:
        transform = IndexTransform(
            IndexDomain.from_shape((2,)),
            (DimensionMap(0, stride=2),),
        )
        with pytest.raises(ValueError, match=r"stride must be \+1 or -1"):
            transform.inverted()

    def test_inverted_rejects_a_repeated_input_dimension(self) -> None:
        transform = IndexTransform(
            IndexDomain.from_shape((2, 1)),
            (DimensionMap(0), DimensionMap(0, offset=5)),
        )
        with pytest.raises(ValueError, match="referenced more than once"):
            transform.inverted()

    def test_inverted_rejects_an_unreferenced_non_singleton_dimension(self) -> None:
        transform = IndexTransform(
            IndexDomain.from_shape((2, 2)),
            (DimensionMap(0), ConstantMap(7)),
        )
        with pytest.raises(ValueError, match="unreferenced input dimension 1.*extent 2"):
            transform.inverted()

    def test_inverted_rejects_input_labels_that_cannot_be_preserved(self) -> None:
        transform = IndexTransform.identity(IndexDomain((0,), (2,), labels=("row",)))
        with pytest.raises(ValueError, match="input labels cannot be represented"):
            transform.inverted()


class TestIndexTransformBasicIndexing:
    def test_slice_identity(self) -> None:
        """slice(None) on identity transform is a no-op."""
        t = IndexTransform.from_shape((10, 20))
        result = t[slice(None), slice(None)]
        assert result.domain.shape == (10, 20)
        assert result.input_rank == 2
        assert result.output_rank == 2

    def test_slice_narrows(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = t[2:8, 5:15]
        # Domains are preserved (TensorStore): the slice keeps its literal
        # coordinates, so the map stays the identity (out = in).
        assert result.domain.shape == (6, 10)
        assert result.domain.origin == (2, 5)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 0
        assert result.output[0].stride == 1
        assert result.output[0].input_dimension == 0
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].offset == 0
        assert result.output[1].input_dimension == 1

    def test_strided_slice(self) -> None:
        t = IndexTransform.from_shape((10,))
        result = t[::2]
        assert result.domain.shape == (5,)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 0
        assert result.output[0].stride == 2

    def test_strided_slice_with_start(self) -> None:
        t = IndexTransform.from_shape((10,))
        result = t[1:9:3]
        # indices: 1, 4, 7 -> 3 elements
        assert result.domain.shape == (3,)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 1
        assert result.output[0].stride == 3

    def test_int_drops_dimension(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = t[3]
        assert result.input_rank == 1
        assert result.output_rank == 2
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == 3
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].input_dimension == 0

    def test_int_middle_dimension(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30))
        result = t[:, 5, :]
        assert result.input_rank == 2
        assert result.output_rank == 3
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].input_dimension == 0
        assert isinstance(result.output[1], ConstantMap)
        assert result.output[1].offset == 5
        assert isinstance(result.output[2], DimensionMap)
        assert result.output[2].input_dimension == 1

    def test_ellipsis(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30))
        result = t[2:8, ...]
        assert result.input_rank == 3
        assert result.domain.shape == (6, 20, 30)

    def test_newaxis(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = t[np.newaxis, :, :]
        assert result.input_rank == 3
        assert result.domain.shape == (1, 10, 20)
        assert result.output_rank == 2
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].input_dimension == 1
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].input_dimension == 2

    def test_int_out_of_bounds(self) -> None:
        t = IndexTransform.from_shape((10,))
        with pytest.raises(IndexError):
            t[10]

    def test_negative_int_is_literal(self) -> None:
        """Negative indices are literal coordinates (TensorStore convention),
        not 'from the end' like NumPy."""
        t = IndexTransform.from_shape((10,))
        with pytest.raises(IndexError):
            t[-1]  # -1 is out of bounds for domain [0, 10)

    def test_negative_int_valid_with_negative_origin(self) -> None:
        """Negative index is valid if the domain includes negative coordinates."""
        domain = IndexDomain(inclusive_min=(-5,), exclusive_max=(5,))
        t = IndexTransform.identity(domain)
        result = t[-3]
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == -3

    def test_composition_of_slices(self) -> None:
        """Slicing a sliced transform re-selects in literal domain coordinates."""
        t = IndexTransform.from_shape((100,))
        result = t[10:50][15:30]
        assert result.domain.shape == (15,)
        assert result.domain.origin == (15,)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 0
        assert result.output[0].stride == 1

    def test_composition_of_strides(self) -> None:
        t = IndexTransform.from_shape((100,))
        result = t[::2][::3]
        # t[::2] -> shape (50,), offset=0, stride=2
        # [::3] -> shape ceil(50/3)=17, offset=0, stride=2*3=6
        assert result.domain.shape == (17,)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].stride == 6

    def test_bare_int(self) -> None:
        """Non-tuple selection."""
        t = IndexTransform.from_shape((10, 20))
        result = t[3]
        assert result.input_rank == 1

    def test_bare_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = t[2:8]
        assert result.domain.shape == (6, 20)


class TestBasicIndexingOnArrayMaps:
    """When a transform already has ArrayMap outputs, basic indexing must
    apply the corresponding operation to the index_array's axes."""

    def test_int_on_array_map_drops_axis(self) -> None:
        """Integer index on a dimension referenced by an ArrayMap should
        index into the array on that axis."""
        arr = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.intp)
        # 2D input domain (3, 2), one ArrayMap output
        t = IndexTransform(
            domain=IndexDomain.from_shape((3, 2)),
            output=(ArrayMap(index_array=arr),),
        )
        # Index with int on dim 0 -> pick row 1 -> arr[1, :] = [30, 40]
        result = t[1]
        assert result.input_rank == 1
        assert result.domain.shape == (2,)
        assert isinstance(result.output[0], ArrayMap)
        np.testing.assert_array_equal(result.output[0].index_array, np.array([30, 40]))

    def test_slice_on_array_map(self) -> None:
        """Slice on a dimension referenced by an ArrayMap should slice the array."""
        arr = np.array([10, 20, 30, 40, 50], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(ArrayMap(index_array=arr),),
        )
        result = t[1:4]
        assert result.domain.shape == (3,)
        assert isinstance(result.output[0], ArrayMap)
        np.testing.assert_array_equal(result.output[0].index_array, np.array([20, 30, 40]))

    def test_strided_slice_on_array_map(self) -> None:
        """Strided slice on ArrayMap should stride the array."""
        arr = np.array([10, 20, 30, 40, 50], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(ArrayMap(index_array=arr),),
        )
        result = t[::2]
        assert result.domain.shape == (3,)
        assert isinstance(result.output[0], ArrayMap)
        np.testing.assert_array_equal(result.output[0].index_array, np.array([10, 30, 50]))

    def test_newaxis_on_array_map(self) -> None:
        """Newaxis should insert an axis in the index_array."""
        arr = np.array([10, 20, 30], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=arr),),
        )
        result = t[np.newaxis, :]
        assert result.input_rank == 2
        assert result.domain.shape == (1, 3)
        assert isinstance(result.output[0], ArrayMap)
        assert result.output[0].index_array.shape == (1, 3)
        np.testing.assert_array_equal(result.output[0].index_array, np.array([[10, 20, 30]]))

    def test_int_drops_one_of_two_array_dims(self) -> None:
        """2D array map, int on dim 0, slice on dim 1."""
        arr = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((2, 3)),
            output=(ArrayMap(index_array=arr),),
        )
        result = t[0, 1:3]
        assert result.input_rank == 1
        assert result.domain.shape == (2,)
        assert isinstance(result.output[0], ArrayMap)
        # arr[0, 1:3] = [20, 30]
        np.testing.assert_array_equal(result.output[0].index_array, np.array([20, 30]))


class TestIndexTransformOindex:
    def test_oindex_int_array(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        idx = np.array([1, 3, 5], dtype=np.intp)
        result = t.oindex[idx, :]
        assert result.input_rank == 2
        assert result.domain.shape == (3, 20)
        assert isinstance(result.output[0], ArrayMap)
        # Full input rank: the array varies along its own axis (0), singleton on 1.
        assert result.output[0].index_array.shape == (3, 1)
        np.testing.assert_array_equal(result.output[0].index_array, idx.reshape(3, 1))
        assert result.output[0].offset == 0
        assert result.output[0].stride == 1
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].input_dimension == 1

    def test_oindex_bool_array(self) -> None:
        t = IndexTransform.from_shape((5,))
        mask = np.array([True, False, True, False, True])
        result = t.oindex[mask]
        assert result.domain.shape == (3,)
        assert isinstance(result.output[0], ArrayMap)
        np.testing.assert_array_equal(
            result.output[0].index_array, np.array([0, 2, 4], dtype=np.intp)
        )

    def test_oindex_mixed(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        idx = np.array([2, 4], dtype=np.intp)
        result = t.oindex[idx, 5:15]
        assert result.input_rank == 2
        assert result.domain.shape == (2, 10)
        # fancy dim: fresh zero-origin; slice dim: preserved literal coords
        assert result.domain.origin == (0, 5)
        assert isinstance(result.output[0], ArrayMap)
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].offset == 0

    def test_oindex_multiple_arrays(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30))
        idx0 = np.array([1, 3], dtype=np.intp)
        idx1 = np.array([5, 10, 15], dtype=np.intp)
        result = t.oindex[idx0, :, idx1]
        assert result.input_rank == 3
        assert result.domain.shape == (2, 20, 3)
        assert isinstance(result.output[0], ArrayMap)
        assert isinstance(result.output[1], DimensionMap)
        assert isinstance(result.output[2], ArrayMap)

    def test_oindex_multiple_arrays_preserves_independent_axes(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = t.oindex[np.array([1, 3]), np.array([2, 4, 6])]
        assert result.domain.shape == (2, 3)
        assert isinstance(result.output[0], ArrayMap)
        assert isinstance(result.output[1], ArrayMap)
        assert result.output[0].index_array.shape == (2, 1)
        assert result.output[1].index_array.shape == (1, 3)


class TestIndexTransformVindex:
    def test_vindex_single_array(self) -> None:
        t = IndexTransform.from_shape((10,))
        idx = np.array([1, 3, 5], dtype=np.intp)
        result = t.vindex[idx]
        assert result.input_rank == 1
        assert result.domain.shape == (3,)
        assert isinstance(result.output[0], ArrayMap)
        np.testing.assert_array_equal(result.output[0].index_array, idx)

    def test_vindex_broadcast(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        idx0 = np.array([[1, 2], [3, 4]], dtype=np.intp)
        idx1 = np.array([[10, 11], [12, 13]], dtype=np.intp)
        result = t.vindex[idx0, idx1]
        assert result.input_rank == 2
        assert result.domain.shape == (2, 2)
        assert isinstance(result.output[0], ArrayMap)
        assert isinstance(result.output[1], ArrayMap)
        np.testing.assert_array_equal(result.output[0].index_array, idx0)
        np.testing.assert_array_equal(result.output[1].index_array, idx1)

    def test_vindex_with_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20, 30))
        idx = np.array([1, 3, 5], dtype=np.intp)
        result = t.vindex[idx, :, :]
        assert result.input_rank == 3
        assert result.domain.shape == (3, 20, 30)
        assert isinstance(result.output[0], ArrayMap)

    def test_vindex_bool_mask(self) -> None:
        t = IndexTransform.from_shape((5,))
        mask = np.array([True, False, True, False, True])
        result = t.vindex[mask]
        assert result.domain.shape == (3,)
        assert isinstance(result.output[0], ArrayMap)

    def test_vindex_broadcast_different_shapes(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        idx0 = np.array([1, 2, 3], dtype=np.intp)
        idx1 = np.array([[10], [11]], dtype=np.intp)
        result = t.vindex[idx0, idx1]
        assert result.input_rank == 2
        assert result.domain.shape == (2, 3)

    def test_vindex_multiple_arrays_preserves_shared_axes(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = t.vindex[np.array([1, 3]), np.array([2, 4])]
        assert result.domain.shape == (2,)
        assert isinstance(result.output[0], ArrayMap)
        assert isinstance(result.output[1], ArrayMap)
        assert result.output[0].index_array.shape == (2,)
        assert result.output[1].index_array.shape == (2,)


@pytest.mark.parametrize("mode", ["oindex", "vindex"])
def test_direct_advanced_index_rejects_float_arrays(mode: str) -> None:
    helper = getattr(IndexTransform.from_shape((5,)), mode)
    with pytest.raises(IndexError, match="integer or boolean"):
        helper[np.array([1.9, 3.2])]


@pytest.mark.parametrize("mode", ["oindex", "vindex"])
def test_direct_advanced_index_rejects_wrong_length_boolean_mask(mode: str) -> None:
    helper = getattr(IndexTransform.from_shape((5,)), mode)
    with pytest.raises(IndexError, match="boolean index.*dimension 5"):
        helper[np.array([True, False])]


class TestSelectionToTransform:
    def test_basic_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = selection_to_transform((slice(2, 8), slice(5, 15)), t, "basic")
        assert result.domain.shape == (6, 10)
        assert result.domain.origin == (2, 5)  # preserved literal coordinates
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 0

    def test_basic_int(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = selection_to_transform((3, slice(None)), t, "basic")
        assert result.input_rank == 1
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == 3

    def test_basic_ellipsis(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = selection_to_transform(Ellipsis, t, "basic")
        assert result.domain.shape == (10, 20)

    def test_orthogonal(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        idx = np.array([1, 3, 5], dtype=np.intp)
        result = selection_to_transform((idx, slice(None)), t, "orthogonal")
        assert result.domain.shape == (3, 20)
        assert isinstance(result.output[0], ArrayMap)

    def test_vectorized(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        idx0 = np.array([1, 3], dtype=np.intp)
        idx1 = np.array([5, 7], dtype=np.intp)
        result = selection_to_transform((idx0, idx1), t, "vectorized")
        assert result.domain.shape == (2,)
        assert isinstance(result.output[0], ArrayMap)
        assert isinstance(result.output[1], ArrayMap)

    def test_composition_with_non_identity(self) -> None:
        """Indexing a sliced transform uses literal domain coordinates.

        The slice [10:50] preserves its domain, so a follow-up [15:30]
        re-selects coordinates 15..29 of the base (TensorStore semantics), and
        the composed map stays the identity (out = in).
        """
        t = IndexTransform.from_shape((100,))[10:50]
        result = selection_to_transform(slice(15, 30), t, "basic")
        assert (result.domain.inclusive_min, result.domain.exclusive_max) == ((15,), (30,))
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 0
        assert result.output[0].stride == 1


class TestIndexTransformIntersect:
    def test_constant_inside(self) -> None:
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ConstantMap(offset=5),),
        )
        result = t.intersect(IndexDomain(inclusive_min=(0,), exclusive_max=(10,)))
        assert result is not None
        restricted, surviving = result
        assert isinstance(restricted.output[0], ConstantMap)
        assert restricted.output[0].offset == 5
        assert surviving is None

    def test_constant_outside(self) -> None:
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ConstantMap(offset=5),),
        )
        result = t.intersect(IndexDomain(inclusive_min=(10,), exclusive_max=(20,)))
        assert result is None

    def test_dimension_partial(self) -> None:
        """DimensionMap over [0,10) intersected with [5,15) narrows input to [5,10)."""
        t = IndexTransform.from_shape((10,))
        result = t.intersect(IndexDomain(inclusive_min=(5,), exclusive_max=(15,)))
        assert result is not None
        restricted, surviving = result
        assert restricted.domain.inclusive_min == (5,)
        assert restricted.domain.exclusive_max == (10,)
        assert surviving is None

    def test_dimension_no_overlap(self) -> None:
        t = IndexTransform.from_shape((10,))
        result = t.intersect(IndexDomain(inclusive_min=(20,), exclusive_max=(30,)))
        assert result is None

    def test_dimension_strided(self) -> None:
        """stride=2, offset=1 over [0,5): storage 1,3,5,7,9. Chunk [4,8)."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((5,)),
            output=(DimensionMap(input_dimension=0, offset=1, stride=2),),
        )
        result = t.intersect(IndexDomain(inclusive_min=(4,), exclusive_max=(8,)))
        assert result is not None
        restricted, _surviving = result
        # input 2->5, input 3->7. Both in [4,8).
        assert restricted.domain.inclusive_min == (2,)
        assert restricted.domain.exclusive_max == (4,)

    @pytest.mark.parametrize(
        ("input_domain", "output_domain", "output_map", "expected"),
        [
            (
                (2**53, 2**53 + 3),
                (2**53 + 1, 2**53 + 2),
                DimensionMap(input_dimension=0),
                (2**53 + 1, 2**53 + 2),
            ),
            (
                (-(2**53) - 2, -(2**53) + 1),
                (-(2**53) - 1, -(2**53)),
                DimensionMap(input_dimension=0),
                (-(2**53) - 1, -(2**53)),
            ),
            (
                (-(2**53) - 2, -(2**53) + 1),
                (2**53 + 1, 2**53 + 2),
                DimensionMap(input_dimension=0, stride=-1),
                (-(2**53) - 1, -(2**53)),
            ),
            (
                (2**53, 2**53 + 3),
                (-(2**53) - 2, -(2**53) - 1),
                DimensionMap(input_dimension=0, stride=-1),
                (2**53 + 2, 2**53 + 3),
            ),
        ],
        ids=[
            "positive-coordinates-positive-stride",
            "negative-coordinates-positive-stride",
            "positive-coordinates-negative-stride",
            "negative-coordinates-negative-stride",
        ],
    )
    def test_dimension_intersection_is_exact_above_float_precision(
        self,
        input_domain: tuple[int, int],
        output_domain: tuple[int, int],
        output_map: DimensionMap,
        expected: tuple[int, int],
    ) -> None:
        transform = IndexTransform(
            domain=IndexDomain((input_domain[0],), (input_domain[1],)),
            output=(output_map,),
        )

        result = transform.intersect(IndexDomain((output_domain[0],), (output_domain[1],)))

        assert result is not None
        restricted, _surviving = result
        assert restricted.domain == IndexDomain((expected[0],), (expected[1],))

    def test_dimension_intersection_accepts_unbounded_python_integer_precision(self) -> None:
        huge = 10**400
        transform = IndexTransform(
            domain=IndexDomain((huge,), (huge + 2,)),
            output=(DimensionMap(input_dimension=0),),
        )

        result = transform.intersect(IndexDomain((huge + 1,), (huge + 2,)))

        assert result is not None
        restricted, _surviving = result
        assert restricted.domain == IndexDomain((huge + 1,), (huge + 2,))

    def test_array_partial(self) -> None:
        arr = np.array([3, 8, 15, 22], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((4,)),
            output=(ArrayMap(index_array=arr),),
        )
        result = t.intersect(IndexDomain(inclusive_min=(5,), exclusive_max=(20,)))
        assert result is not None
        restricted, surviving = result
        assert isinstance(restricted.output[0], ArrayMap)
        np.testing.assert_array_equal(restricted.output[0].index_array, np.array([8, 15]))
        assert surviving is not None
        np.testing.assert_array_equal(surviving, np.array([1, 2]))

    def test_array_none_inside(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((3,)),
            output=(ArrayMap(index_array=arr),),
        )
        assert t.intersect(IndexDomain(inclusive_min=(10,), exclusive_max=(20,))) is None

    def test_2d_mixed(self) -> None:
        """2D: ConstantMap on dim 0, DimensionMap on dim 1."""
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(
                ConstantMap(offset=5),
                DimensionMap(input_dimension=0, offset=0, stride=1),
            ),
        )
        chunk = IndexDomain(inclusive_min=(0, 5), exclusive_max=(10, 15))
        result = t.intersect(chunk)
        assert result is not None
        restricted, _ = result
        assert isinstance(restricted.output[0], ConstantMap)
        assert restricted.output[0].offset == 5
        assert isinstance(restricted.output[1], DimensionMap)
        assert restricted.domain.inclusive_min == (5,)
        assert restricted.domain.exclusive_max == (10,)


class TestIndexTransformTranslate:
    def test_translate_constant(self) -> None:
        t = IndexTransform(
            domain=IndexDomain.from_shape((10,)),
            output=(ConstantMap(offset=5),),
        )
        result = t.translate((-5,))
        assert isinstance(result.output[0], ConstantMap)
        assert result.output[0].offset == 0

    def test_translate_dimension(self) -> None:
        t = IndexTransform.from_shape((10,))
        result = t.translate((-3,))
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == -3
        assert result.output[0].stride == 1

    def test_translate_array(self) -> None:
        arr = np.array([5, 10], dtype=np.intp)
        t = IndexTransform(
            domain=IndexDomain.from_shape((2,)),
            output=(ArrayMap(index_array=arr, offset=3),),
        )
        result = t.translate((-3,))
        assert isinstance(result.output[0], ArrayMap)
        assert result.output[0].offset == 0
        np.testing.assert_array_equal(result.output[0].index_array, arr)

    def test_translate_2d(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = t.translate((-5, -10))
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == -5
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].offset == -10


class TestArrayMapDependencyAxes:
    """`_array_map_dependency_axes` derives the input axes an array varies on
    from its (full-rank) shape: non-singleton axes vary, singleton axes do not."""

    def test_orthogonal_single_axis(self) -> None:
        from zarr_indexing.transform import _array_map_dependency_axes

        t = IndexTransform.from_shape((10, 20)).oindex[np.array([1, 3]), np.array([2, 4, 6])]
        m0, m1 = t.output[0], t.output[1]
        assert isinstance(m0, ArrayMap)
        assert isinstance(m1, ArrayMap)
        assert _array_map_dependency_axes(m0.index_array) == (0,)
        assert _array_map_dependency_axes(m1.index_array) == (1,)

    def test_vectorized_shares_axes(self) -> None:
        from zarr_indexing.transform import _array_map_dependency_axes

        t = IndexTransform.from_shape((10, 20)).vindex[np.array([1, 3]), np.array([2, 4])]
        m0, m1 = t.output[0], t.output[1]
        assert isinstance(m0, ArrayMap)
        assert isinstance(m1, ArrayMap)
        assert _array_map_dependency_axes(m0.index_array) == (0,)
        assert _array_map_dependency_axes(m1.index_array) == (0,)

    def test_scalar_array_has_no_dependency(self) -> None:
        from zarr_indexing.transform import _array_map_dependency_axes

        assert _array_map_dependency_axes(np.ones((1, 1), dtype=np.intp)) == ()

    def test_zero_length_axis_has_no_dependency(self) -> None:
        """An axis of size 0 carries no dependency either: it selects nothing, so
        the array does not vary along it any more than along a singleton."""
        from zarr_indexing.transform import _array_map_dependency_axes

        assert _array_map_dependency_axes(np.zeros((0, 4), dtype=np.intp)) == (1,)
        assert _array_map_dependency_axes(np.zeros((3, 0), dtype=np.intp)) == (0,)
        assert _array_map_dependency_axes(np.zeros((0, 1), dtype=np.intp)) == ()

    def test_zero_length_axis_does_not_make_a_map_correlated(self) -> None:
        """An empty orthogonal selection is legal, so it must classify as one."""
        m = ArrayMap(index_array=np.zeros((0, 4), dtype=np.intp), input_dimension=1)
        assert array_map_dependent_axis(m) == 1


class TestIntersectArrayMapClassification:
    """`_intersect` must distinguish orthogonal (outer-product) ArrayMaps from
    correlated (vectorized) ones by their dependency axes, keep surviving arrays
    at full input rank, and preserve residual (slice) dimensions."""

    def test_orthogonal_outer_product_keeps_full_rank(self) -> None:
        """Two arrays on distinct axes narrow independently and stay full rank;
        out_indices is a per-output-dim dict of surviving positions."""
        t = IndexTransform.from_shape((10, 10)).oindex[np.array([1, 3, 8]), np.array([2, 6, 9])]
        # Chunk covering storage [0,5) x [0,5): rows 1,3 survive (out pos 0,1),
        # cols 2 survives (out pos 0).
        chunk = IndexDomain(inclusive_min=(0, 0), exclusive_max=(5, 5))
        result = t.intersect(chunk)
        assert result is not None
        restricted, out_indices = result
        assert isinstance(restricted.output[0], ArrayMap)
        assert isinstance(restricted.output[1], ArrayMap)
        # Full input rank preserved (not raveled to 1-D).
        assert restricted.output[0].index_array.ndim == 2
        assert restricted.output[1].index_array.ndim == 2
        assert restricted.domain.ndim == 2
        assert isinstance(out_indices, dict)
        np.testing.assert_array_equal(out_indices[0], np.array([0, 1]))
        np.testing.assert_array_equal(out_indices[1], np.array([0]))

    def test_correlated_with_residual_slice_preserves_slice_dim(self) -> None:
        """A vindex transform with two correlated arrays plus a residual slice
        dim intersects without a rank error and keeps the DimensionMap."""
        t = IndexTransform.from_shape((4, 3, 5)).vindex[np.array([1, 3]), np.array([2, 0])]
        # Chunk covering storage [0,2) x [2,3) x [0,5): only point (1,2,*) is in
        # bounds on both array dims -> one surviving broadcast point.
        chunk = IndexDomain(inclusive_min=(0, 2, 0), exclusive_max=(2, 3, 5))
        result = t.intersect(chunk)
        assert result is not None
        restricted, out_indices = result
        # A DimensionMap for the residual slice dim survives (no post-init error).
        assert any(isinstance(m, DimensionMap) for m in restricted.output)
        assert out_indices is not None

    def test_length1_orthogonal_not_treated_as_correlated(self) -> None:
        """A length-1 orthogonal array (all-singleton shape) is still an outer
        product with the length-3 axis: out_indices is a dict, not a flat array."""
        t = IndexTransform.from_shape((6, 6)).oindex[np.array([2]), np.array([1, 3, 5])]
        chunk = IndexDomain(inclusive_min=(0, 0), exclusive_max=(6, 6))
        result = t.intersect(chunk)
        assert result is not None
        _restricted, out_indices = result
        assert isinstance(out_indices, dict)


class TestDerivedMapDependency:
    """A map's `input_dimension` must describe the array it is built with.

    Three separate failures came from one stale value: a vectorized index applied
    to an orthogonal map makes it correlated, but the old dependency was carried
    onto the new array anyway. Readers fall back to that field when the shape
    alone cannot say, so the wrong axis was believed much later — by a scatter
    that filed positions under it, which is why the answer depended on how the
    read was partitioned.
    """

    def test_a_vindex_over_a_fancy_view_is_marked_correlated(self) -> None:
        base = np.arange(6)
        view = (
            LazyArray(base)
            .lazy.oindex[np.array([0, 1])]
            .lazy.vindex[np.array([[0, 1, 0], [1, 0, 1]])]
        )
        np.testing.assert_array_equal(
            np.asarray(view.result()), base[[0, 1]][[[0, 1, 0], [1, 0, 1]]]
        )

    def test_the_same_view_resolves_alike_however_it_is_partitioned(self) -> None:
        base = np.arange(36).reshape(6, 6)

        def build(array: LazyArray) -> LazyArray:
            return array.lazy.oindex[np.array([-3, -6, -4]), -4].lazy.vindex[np.array([[-2, -3]])]

        unpartitioned = np.asarray(build(LazyArray(base)).result())
        partitioned = np.asarray(build(LazyArray(base).with_parts((3, 3))).result())
        np.testing.assert_array_equal(partitioned, unpartitioned)
        np.testing.assert_array_equal(unpartitioned, np.array([[2, 20]]))

    def test_an_array_map_claiming_an_axis_it_does_not_vary_over_is_rejected(self) -> None:
        """The validation that would have caught the two above at their source."""
        with pytest.raises(ValueError, match="varies over"):
            IndexTransform(
                domain=IndexDomain.from_shape((2, 3)),
                output=(
                    ArrayMap(index_array=np.array([[0, 1, 2]], dtype=np.intp), input_dimension=0),
                ),
            )

    def test_an_array_map_input_dimension_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            IndexTransform(
                domain=IndexDomain.from_shape((3,)),
                output=(ArrayMap(index_array=np.array([0], dtype=np.intp), input_dimension=99),),
            )


def test_an_orthogonal_step_over_a_correlated_view_is_an_outer_product() -> None:
    """`oindex` after `vindex` means the outer product, not a joint gather.

    The reindexing applied its index tuple positionally, which is NumPy's
    *vectorized* rule, so two arrays collapsed into one axis and the result came
    back a rank short of what was asked for.
    """
    base = np.arange(14).reshape(7, 2)
    view = LazyArray(base).lazy.vindex[
        np.array([[5, 5], [1, 2], [0, 4]]), np.array([[1, 1], [1, 0], [1, 0]])
    ]
    result = np.asarray(view.lazy.oindex[np.array([1, 1, 0]), np.array([1, 1, 0, 1])].result())
    assert result.shape == (3, 4)
    np.testing.assert_array_equal(result, np.array([[4, 4, 3, 4], [4, 4, 3, 4], [11, 11, 11, 11]]))


@pytest.mark.parametrize(
    ("value", "description"),
    [(1, "one below the lower bound"), (10, "the exclusive upper bound itself")],
)
def test_an_index_array_value_just_outside_the_domain_is_refused(
    value: int, description: str
) -> None:
    """The bound checks are probed at the boundary, not comfortably past it.

    Both were only ever exercised from well outside the domain, so relaxing
    either by one — `lo - 1` instead of `lo` — went unnoticed while letting a
    view read a cell it does not address.
    """
    transform = IndexTransform.from_shape((12,))[2:10]
    with pytest.raises(BoundsCheckError, match="out of bounds"):
        transform.oindex[np.array([value, 3])]


def test_an_index_array_value_at_each_end_of_the_domain_is_accepted() -> None:
    """The other side of the same boundary: the extremes themselves are in range."""
    transform = IndexTransform.from_shape((12,))[2:10]
    array_map = transform.oindex[np.array([2, 9])].output[0]
    assert isinstance(array_map, ArrayMap)
    np.testing.assert_array_equal(array_map.index_array, np.array([2, 9]))
