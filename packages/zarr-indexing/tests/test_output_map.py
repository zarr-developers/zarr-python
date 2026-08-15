from __future__ import annotations

import pickle

import numpy as np
import pytest

from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap


class TestConstantMap:
    def test_construction(self) -> None:
        m = ConstantMap(offset=42)
        assert m.offset == 42

    def test_default_offset(self) -> None:
        m = ConstantMap()
        assert m.offset == 0

    def test_frozen(self) -> None:
        m = ConstantMap(offset=5)
        assert isinstance(m, ConstantMap)


class TestDimensionMap:
    def test_construction(self) -> None:
        m = DimensionMap(input_dimension=3, offset=5, stride=2)
        assert m.input_dimension == 3
        assert m.offset == 5
        assert m.stride == 2

    def test_defaults(self) -> None:
        m = DimensionMap(input_dimension=0)
        assert m.offset == 0
        assert m.stride == 1

    def test_frozen(self) -> None:
        m = DimensionMap(input_dimension=0)
        assert isinstance(m, DimensionMap)


class TestArrayMap:
    def test_construction(self) -> None:
        arr = np.array([1, 3, 5], dtype=np.intp)
        m = ArrayMap(index_array=arr, offset=10, stride=2)
        assert m.offset == 10
        assert m.stride == 2
        np.testing.assert_array_equal(m.index_array, arr)

    def test_defaults(self) -> None:
        arr = np.array([0, 1], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        assert m.offset == 0
        assert m.stride == 1

    def test_frozen(self) -> None:
        arr = np.array([0], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        assert isinstance(m, ArrayMap)

    def test_owns_index_array_and_keeps_hash_stable(self) -> None:
        arr = np.array([1, 3, 5], dtype=np.intp)
        m = ArrayMap(index_array=arr)
        lookup = {m: "value"}

        arr[:] = 9

        np.testing.assert_array_equal(m.index_array, [1, 3, 5])
        assert lookup[m] == "value"

    def test_pickle_round_trip_keeps_index_array_read_only(self) -> None:
        restored = pickle.loads(pickle.dumps(ArrayMap(index_array=np.array([1, 3, 5]))))

        assert not restored.index_array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            restored.index_array[0] = 9

    def test_index_array_cannot_be_made_writeable(self) -> None:
        m = ArrayMap(index_array=np.array([1, 3, 5]))

        with pytest.raises(ValueError):
            m.index_array.flags.writeable = True

    def test_equal_array_maps_have_equal_hashes_across_integer_dtypes(self) -> None:
        left = ArrayMap(np.array([1, 2], dtype=np.int32))
        right = ArrayMap(np.array([1, 2], dtype=np.int64))
        assert left == right
        assert hash(left) == hash(right)
        assert left.index_array.dtype == np.dtype(np.intp)

    def test_rejects_non_integer_index_array(self) -> None:
        with pytest.raises(TypeError, match="index_array must have an integer dtype"):
            ArrayMap(np.array([1.5, 2.0], dtype=np.float64))

    def test_rejects_unsigned_index_array_value_outside_intp(self) -> None:
        outside_intp = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)

        with pytest.raises(OverflowError, match="outside np.intp range"):
            ArrayMap(outside_intp)
