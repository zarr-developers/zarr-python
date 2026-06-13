from __future__ import annotations

import numpy as np

from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap


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
