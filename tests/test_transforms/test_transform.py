from __future__ import annotations

import numpy as np
import pytest

from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap
from zarr.core.transforms.transform import IndexTransform, selection_to_transform


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
        assert result.domain.shape == (6, 10)
        assert result.domain.origin == (0, 0)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 2
        assert result.output[0].stride == 1
        assert result.output[0].input_dimension == 0
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].offset == 5
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
        """Slicing a sliced transform should compose offsets."""
        t = IndexTransform.from_shape((100,))
        result = t[10:50][5:20]
        assert result.domain.shape == (15,)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 15
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
        np.testing.assert_array_equal(result.output[0].index_array, idx)
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
        assert isinstance(result.output[0], ArrayMap)
        assert isinstance(result.output[1], DimensionMap)
        assert result.output[1].offset == 5

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


class TestSelectionToTransform:
    def test_basic_slice(self) -> None:
        t = IndexTransform.from_shape((10, 20))
        result = selection_to_transform((slice(2, 8), slice(5, 15)), t, "basic")
        assert result.domain.shape == (6, 10)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 2

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
        """Indexing a sliced transform composes offsets."""
        t = IndexTransform.from_shape((100,))[10:50]
        result = selection_to_transform(slice(5, 20), t, "basic")
        assert result.domain.shape == (15,)
        assert isinstance(result.output[0], DimensionMap)
        assert result.output[0].offset == 15


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
