"""Regression tests: the engine facade must not force a host (numpy) conversion
on a device buffer (e.g. cupy/torch) whose implicit `np.asarray` coercion is
refused.

A `_DeviceArray` stand-in wraps a numpy array but raises `TypeError` from
`__array__`, exactly like cupy refusing an implicit host copy. Everything else
(indexing, `astype`, `copy`, and numpy's `__array_function__` protocol) is
delegated to the wrapped array and re-wrapped, so operations that *stay in the
array's own namespace* keep working while any `np.asarray`/`np.array` coercion
raises. Driving `Array.__setitem__`/`__getitem__` through a stub engine that
returns such buffers therefore fails before the facade's namespace fixes and
passes after them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

import zarr
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.abc.engine import Region
    from zarr.core.buffer import BufferPrototype, NDBuffer
    from zarr.core.metadata import ArrayMetadata


class _DeviceArray:
    """A minimal `NDArrayLike` whose implicit host conversion is refused.

    Mimics a cupy array: indexing / `astype` / `copy` and numpy function
    dispatch (`__array_function__`) all work in-namespace, but `__array__`
    (what `np.asarray`/`np.array` call) raises, so any host coercion blows up.
    """

    def __init__(self, data: npt.NDArray[Any]) -> None:
        self._a: npt.NDArray[Any] = data

    # --- host coercion is forbidden -------------------------------------
    def __array__(self, dtype: Any = None) -> npt.NDArray[Any]:
        raise TypeError("implicit conversion to a host numpy array is not allowed")

    # --- in-namespace numpy-function dispatch ---------------------------
    def __array_function__(
        self, func: Any, types: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        unwrapped = tuple(a._a if isinstance(a, _DeviceArray) else a for a in args)
        result = func(*unwrapped, **kwargs)
        if isinstance(result, np.ndarray):
            return _DeviceArray(result)
        return result

    # --- ndarray-like surface -------------------------------------------
    @property
    def dtype(self) -> np.dtype[Any]:
        return self._a.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._a.shape

    @property
    def ndim(self) -> int:
        return self._a.ndim

    @property
    def size(self) -> int:
        return self._a.size

    def __len__(self) -> int:
        return len(self._a)

    def __getitem__(self, key: Any) -> _DeviceArray:
        return _DeviceArray(self._a[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        self._a[key] = value._a if isinstance(value, _DeviceArray) else value

    def astype(self, dtype: Any, order: Any = "K", *, copy: bool = True) -> _DeviceArray:
        return _DeviceArray(self._a.astype(dtype, order=order, copy=copy))

    def copy(self) -> _DeviceArray:
        return _DeviceArray(self._a.copy())

    def reshape(self, *args: Any, **kwargs: Any) -> _DeviceArray:
        return _DeviceArray(self._a.reshape(*args, **kwargs))


def _region_to_index(region: Region) -> tuple[slice, ...]:
    return tuple(slice(s, e) for s, e in zip(region.start, region.end_exclusive, strict=True))


class _DeviceEngine:
    """A synchronous `ArrayEngine` backed by numpy that hands the facade
    `_DeviceArray` buffers (on read) and unwraps them (on write)."""

    def __init__(self, data: np.ndarray[Any, Any]) -> None:
        self._data = data

    def read_selection(self, selection: Region, *, prototype: BufferPrototype) -> _DeviceArray:
        return _DeviceArray(self._data[_region_to_index(selection)].copy())

    def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        arr: object = value.as_ndarray_like()
        assert isinstance(arr, _DeviceArray), (
            "facade coerced the box off the device namespace before writing"
        )
        self._data[_region_to_index(selection)] = arr._a

    def with_metadata(self, metadata: ArrayMetadata) -> _DeviceEngine:
        return self


class _AsyncDeviceEngine:
    """Async mirror of `_DeviceEngine` for the `AsyncArray` data path."""

    def __init__(self, data: npt.NDArray[Any]) -> None:
        self._data = data

    async def read_selection(
        self, selection: Region, *, prototype: BufferPrototype
    ) -> _DeviceArray:
        return _DeviceArray(self._data[_region_to_index(selection)].copy())

    async def write_selection(
        self, selection: Region, value: NDBuffer, *, prototype: BufferPrototype
    ) -> None:
        arr: object = value.as_ndarray_like()
        assert isinstance(arr, _DeviceArray), (
            "facade coerced the box off the device namespace before writing"
        )
        self._data[_region_to_index(selection)] = arr._a

    def with_metadata(self, metadata: ArrayMetadata) -> _AsyncDeviceEngine:
        return self


def _array_on_device_engine() -> tuple[zarr.Array[Any], _DeviceEngine]:
    z = zarr.create_array(MemoryStore(), shape=(8,), chunks=(4,), dtype="int64")
    engine = _DeviceEngine(np.zeros(8, dtype="int64"))
    # inject the device engine as this array's cached sync engine
    z._engine = engine  # type: ignore[assignment]
    return z, engine


def _array_2d_on_device_engine() -> tuple[zarr.Array[Any], _DeviceEngine]:
    z = zarr.create_array(MemoryStore(), shape=(4, 4), chunks=(4, 4), dtype="int64")
    engine = _DeviceEngine(np.arange(16, dtype="int64").reshape(4, 4))
    z._engine = engine  # type: ignore[assignment]
    return z, engine


def test_identity_setitem_keeps_device_buffer() -> None:
    # full-box write: `_prepare_set_selection` broadcasts the value into the box
    # without a read; it must not `np.asarray` the device value.
    z, engine = _array_on_device_engine()
    value = _DeviceArray(np.arange(8, dtype="int64"))
    z[:] = value
    np.testing.assert_array_equal(engine._data, np.arange(8, dtype="int64"))


def test_strided_setitem_reads_and_patches_on_device() -> None:
    # strided write is a read-modify-write: the engine box read comes back as a
    # `_DeviceArray`, and `_patch_selection_box` must patch it in-namespace
    # (`raw.copy()`), never `np.asarray(raw)`.
    z, engine = _array_on_device_engine()
    z[::2] = np.array([10, 20, 30, 40], dtype="int64")
    expected = np.zeros(8, dtype="int64")
    expected[::2] = [10, 20, 30, 40]
    np.testing.assert_array_equal(engine._data, expected)


def test_strided_getitem_keeps_device_buffer() -> None:
    # strided read: `_finish_get_selection` applies the post index and reorders
    # the result in the device namespace, returning a `_DeviceArray` untouched
    # by `np.asarray`.
    z, engine = _array_on_device_engine()
    engine._data[:] = np.arange(8, dtype="int64")
    result: object = z[::2]
    assert isinstance(result, _DeviceArray), "strided read coerced off the device namespace"
    np.testing.assert_array_equal(result._a, np.arange(8, dtype="int64")[::2])


def test_basic_selection_scalar_read_keeps_device_buffer() -> None:
    # all-integer basic read scalarizes a 0-d result; `_finalize_result` must
    # extract the element in the device namespace, not via `np.asarray(result)[()]`.
    z, engine = _array_2d_on_device_engine()
    result: object = z.get_basic_selection((1, 2))
    assert isinstance(result, _DeviceArray), "scalar read coerced off the device namespace"
    np.testing.assert_array_equal(result._a, engine._data[1, 2])


async def test_async_getitem_scalar_read_keeps_device_buffer() -> None:
    z = zarr.create_array(MemoryStore(), shape=(4, 4), chunks=(4, 4), dtype="int64")
    aa = z.async_array
    engine = _AsyncDeviceEngine(np.arange(16, dtype="int64").reshape(4, 4))
    object.__setattr__(aa, "engine", engine)
    result: object = await aa.getitem((1, 2))
    assert isinstance(result, _DeviceArray), "async scalar read coerced off the device namespace"
    np.testing.assert_array_equal(result._a, engine._data[1, 2])


def test_oindex_set_integer_and_array_axis_keeps_device_buffer() -> None:
    # orthogonal write with a dropped integer axis widens the value with a
    # `np.newaxis`; `_widen_value_for_squeeze` must index the value in its own
    # namespace instead of `np.asarray(value)[...]`.
    z, engine = _array_2d_on_device_engine()
    value = _DeviceArray(np.array([10, 20], dtype="int64"))
    z.oindex[1, np.array([0, 2])] = value
    expected = np.arange(16, dtype="int64").reshape(4, 4)
    expected[1, [0, 2]] = [10, 20]
    np.testing.assert_array_equal(engine._data, expected)


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_scalar_bytes_read_returns_numpy_scalar() -> None:
    # A fixed/vlen-bytes scalar read produces a numpy scalar (`np.bytes_`), not a
    # 0-d ndarray; the device-namespace `result[()]` branch must not swallow it
    # (`np.bytes_[()]` raises "byte indices must be integers"). numpy scalars go
    # through the `np.asarray` path, staying bit-identical to the historical
    # `as_scalar()` return. (Distilled from a `test_basic_indexing` hypothesis
    # failure; uses the real default engine, no device stand-in.)
    z = zarr.create_array(MemoryStore(), shape=(1,), chunks=(1,), dtype="S4")
    z[:] = np.array([b"ab"], dtype="S4")
    result = z.get_basic_selection(0)
    assert isinstance(result, np.bytes_)
    assert result == b"ab"


def test_facade_never_coerces_device_buffer_to_host() -> None:
    # Belt-and-braces: a bare `np.asarray` on the stand-in must raise, proving the
    # tests above would trip any host coercion the facade performed.
    with pytest.raises(TypeError):
        np.asarray(_DeviceArray(np.arange(4)))
