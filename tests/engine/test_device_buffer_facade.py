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


def _array_on_device_engine() -> tuple[zarr.Array[Any], _DeviceEngine]:
    z = zarr.create_array(MemoryStore(), shape=(8,), chunks=(4,), dtype="int64")
    engine = _DeviceEngine(np.zeros(8, dtype="int64"))
    # inject the device engine as this array's cached sync engine
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


def test_facade_never_coerces_device_buffer_to_host() -> None:
    # Belt-and-braces: a bare `np.asarray` on the stand-in must raise, proving the
    # tests above would trip any host coercion the facade performed.
    with pytest.raises(TypeError):
        np.asarray(_DeviceArray(np.arange(4)))
