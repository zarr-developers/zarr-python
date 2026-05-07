from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import zarr
from zarr.storage import MemoryStore


@pytest.fixture
def arr() -> zarr.Array[Any]:
    """Create a 2D array with known data."""
    store = MemoryStore()
    a = zarr.create(shape=(20, 30), chunks=(5, 10), dtype="i4", store=store)
    data = np.arange(600, dtype="i4").reshape(20, 30)
    a[...] = data
    return a


@pytest.fixture
def data() -> np.ndarray[Any, Any]:
    return np.arange(600, dtype="i4").reshape(20, 30)


class TestEagerRead:
    def test_basic_slice(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        result = arr[2:8, 5:15]
        np.testing.assert_array_equal(result, data[2:8, 5:15])

    def test_basic_int(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        result = arr[3]
        np.testing.assert_array_equal(result, data[3])

    def test_basic_int_scalar(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        result = arr[3, 5]
        assert result == data[3, 5]

    def test_ellipsis(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        result = arr[...]
        np.testing.assert_array_equal(result, data)

    def test_strided_slice(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        result = arr[::2, ::3]
        np.testing.assert_array_equal(result, data[::2, ::3])

    def test_oindex(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        idx = np.array([1, 5, 10], dtype=np.intp)
        result = arr.oindex[idx, :]
        np.testing.assert_array_equal(result, data[idx, :])

    def test_vindex(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        idx0 = np.array([1, 5, 10], dtype=np.intp)
        idx1 = np.array([2, 8, 15], dtype=np.intp)
        result = arr.vindex[idx0, idx1]
        np.testing.assert_array_equal(result, data[idx0, idx1])

    def test_slice_across_chunks(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        """Slice that spans multiple chunks."""
        result = arr[3:17, 8:22]
        np.testing.assert_array_equal(result, data[3:17, 8:22])

    def test_single_element(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        result = arr[0:1, 0:1]
        np.testing.assert_array_equal(result, data[0:1, 0:1])

    def test_full_read(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        result = arr[:]
        np.testing.assert_array_equal(result, data)


class TestEagerWrite:
    def test_write_slice(self, arr: zarr.Array[Any]) -> None:
        arr[2:5, 10:20] = np.ones((3, 10), dtype="i4") * 99
        result = arr[2:5, 10:20]
        np.testing.assert_array_equal(result, np.ones((3, 10), dtype="i4") * 99)

    def test_write_scalar(self, arr: zarr.Array[Any]) -> None:
        arr[0, 0] = 42
        assert arr[0, 0] == 42

    def test_roundtrip(self, arr: zarr.Array[Any]) -> None:
        new_data = np.random.randint(0, 100, size=(20, 30), dtype="i4")
        arr[...] = new_data
        np.testing.assert_array_equal(arr[...], new_data)

    def test_write_across_chunks(self, arr: zarr.Array[Any]) -> None:
        """Write spanning multiple chunks."""
        val = np.ones((14, 14), dtype="i4") * 77
        arr[3:17, 8:22] = val
        result = arr[3:17, 8:22]
        np.testing.assert_array_equal(result, val)


class TestLazyRead:
    def test_lazy_shape(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        v = arr.z[2:8, 5:15]
        assert isinstance(v, zarr.Array)
        assert v.shape == (6, 10)

    def test_lazy_resolve(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        v = arr.z[2:8, 5:15]
        result = v[...]
        np.testing.assert_array_equal(result, data[2:8, 5:15])

    def test_lazy_np_asarray(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        v = arr.z[2:8]
        result = np.asarray(v)
        np.testing.assert_array_equal(result, data[2:8])

    def test_lazy_composition(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        v = arr.z[2:12].z[3:8]
        assert v.shape == (5, 30)
        result = v[...]
        np.testing.assert_array_equal(result, data[5:10])

    def test_lazy_oindex(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        idx = np.array([1, 5, 10], dtype=np.intp)
        v = arr.z.oindex[idx, :]
        assert isinstance(v, zarr.Array)
        assert v.shape == (3, 30)
        result = v[...]
        np.testing.assert_array_equal(result, data[idx, :])

    def test_lazy_vindex(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        idx0 = np.array([1, 5, 10], dtype=np.intp)
        idx1 = np.array([2, 8, 15], dtype=np.intp)
        v = arr.z.vindex[idx0, idx1]
        assert isinstance(v, zarr.Array)
        assert v.shape == (3,)
        result = v[...]
        np.testing.assert_array_equal(result, data[idx0, idx1])

    def test_lazy_resolve_method(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        v = arr.z[2:8]
        result = v.resolve()
        np.testing.assert_array_equal(result, data[2:8])

    def test_lazy_across_chunks(self, arr: zarr.Array[Any], data: np.ndarray[Any, Any]) -> None:
        """Lazy slice spanning multiple chunks resolves correctly."""
        v = arr.z[3:17, 8:22]
        result = v[...]
        np.testing.assert_array_equal(result, data[3:17, 8:22])


class TestLazyWrite:
    def test_lazy_write(self, arr: zarr.Array[Any]) -> None:
        arr.z[2:5, 10:20] = np.ones((3, 10), dtype="i4") * 99
        result = arr[2:5, 10:20]
        np.testing.assert_array_equal(result, np.ones((3, 10), dtype="i4") * 99)

    def test_lazy_oindex_write(self, arr: zarr.Array[Any]) -> None:
        idx = np.array([0, 5, 10], dtype=np.intp)
        arr.z.oindex[idx, :] = np.zeros((3, 30), dtype="i4")
        result = arr.oindex[idx, :]
        np.testing.assert_array_equal(result, np.zeros((3, 30), dtype="i4"))

    def test_lazy_vindex_write(self, arr: zarr.Array[Any]) -> None:
        idx0 = np.array([0, 5, 10], dtype=np.intp)
        idx1 = np.array([0, 5, 10], dtype=np.intp)
        arr.z.vindex[idx0, idx1] = np.array([77, 88, 99], dtype="i4")
        result = arr.vindex[idx0, idx1]
        np.testing.assert_array_equal(result, np.array([77, 88, 99], dtype="i4"))
