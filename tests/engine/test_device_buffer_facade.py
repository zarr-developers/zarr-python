"""Regression tests: neither `Array` nor `AsyncArray` may force a host (numpy)
conversion on a device buffer (e.g. cupy/torch) whose implicit `np.asarray`
coercion is refused.

The engine owns the whole data path, so what these tests pin is the *facade's*
end of the contract: the value handed to `write_selection` reaches the engine as
the caller wrote it, and whatever `read_selection` returns reaches the caller
unchanged. A `_DeviceArray` stand-in wraps a numpy array but raises `TypeError`
from `__array__`, exactly like cupy refusing an implicit host copy. Everything
else (indexing, `astype`, `copy`, and numpy's `__array_function__` protocol) is
delegated to the wrapped array and re-wrapped, so operations that *stay in the
array's own namespace* keep working while any `np.asarray`/`np.array` coercion
raises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

import zarr
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.abc.engine import SelectionRequest
    from zarr.core.buffer import BufferPrototype, NDBuffer
    from zarr.core.indexing import Fields
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


def _numpy_key(request: SelectionRequest) -> Any:
    """The numpy key equivalent to `request`'s raw selection.

    These tests only use basic selections and orthogonal ones with a single
    advanced index, for which numpy's own indexing already means what zarr's
    does -- so the stub engine can apply the raw selection directly and stays
    free of any resolution logic of its own.
    """
    assert request.kind in ("basic", "orthogonal")
    return request.selection


class _DeviceEngine:
    """A synchronous `ArrayEngine` backed by numpy that hands the facade
    `_DeviceArray` buffers (on read) and asserts it is handed them (on write)."""

    def __init__(self, data: npt.NDArray[Any]) -> None:
        self._data = data

    def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> _DeviceArray:
        # `np.asarray` keeps a rank-0 read a 0-d array rather than a numpy
        # scalar, so even a scalar read stays in the device namespace.
        return _DeviceArray(np.asarray(self._data[_numpy_key(request)]).copy())

    def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        assert isinstance(value, _DeviceArray), (
            "facade coerced the value off the device namespace before writing"
        )
        self._data[_numpy_key(request)] = value._a

    def with_metadata(self, metadata: ArrayMetadata) -> _DeviceEngine:
        return self


class _AsyncDeviceEngine:
    """Async mirror of `_DeviceEngine` for the `AsyncArray` data path."""

    def __init__(self, data: npt.NDArray[Any]) -> None:
        self._data = data
        self._sync = _DeviceEngine(data)

    async def read_selection(
        self,
        request: SelectionRequest,
        *,
        prototype: BufferPrototype,
        out: NDBuffer | None = None,
        fields: Fields | None = None,
    ) -> _DeviceArray:
        return self._sync.read_selection(request, prototype=prototype, out=out, fields=fields)

    async def write_selection(
        self,
        request: SelectionRequest,
        value: npt.ArrayLike,
        *,
        prototype: BufferPrototype,
        fields: Fields | None = None,
    ) -> None:
        self._sync.write_selection(request, value, prototype=prototype, fields=fields)

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


def test_full_box_setitem_keeps_device_buffer() -> None:
    z, engine = _array_on_device_engine()
    z[:] = _DeviceArray(np.arange(8, dtype="int64"))
    np.testing.assert_array_equal(engine._data, np.arange(8, dtype="int64"))


def test_strided_setitem_keeps_device_buffer() -> None:
    # a strided write is a read-modify-write inside the engine; the facade must
    # still hand the value over untouched rather than normalizing it first.
    z, engine = _array_on_device_engine()
    z[::2] = _DeviceArray(np.array([10, 20, 30, 40], dtype="int64"))
    expected = np.zeros(8, dtype="int64")
    expected[::2] = [10, 20, 30, 40]
    np.testing.assert_array_equal(engine._data, expected)


def test_strided_getitem_keeps_device_buffer() -> None:
    z, engine = _array_on_device_engine()
    engine._data[:] = np.arange(8, dtype="int64")
    result: object = z[::2]
    assert isinstance(result, _DeviceArray), "strided read coerced off the device namespace"
    np.testing.assert_array_equal(result._a, np.arange(8, dtype="int64")[::2])


def test_basic_selection_scalar_read_keeps_device_buffer() -> None:
    # an all-integer basic read is where zarr's own engine scalarizes; the
    # facade must not repeat that on an engine that returned a device buffer.
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
    # an orthogonal write with a dropped integer axis has to widen the value
    # back to the selection's rank -- work the engine does, on a value the
    # facade must not have converted on the way in.
    z, engine = _array_2d_on_device_engine()
    z.oindex[1, np.array([0, 2])] = _DeviceArray(np.array([10, 20], dtype="int64"))
    expected = np.arange(16, dtype="int64").reshape(4, 4)
    expected[1, [0, 2]] = [10, 20]
    np.testing.assert_array_equal(engine._data, expected)


@pytest.mark.filterwarnings("ignore::zarr.core.dtype.common.UnstableSpecificationWarning")
def test_scalar_bytes_read_returns_numpy_scalar() -> None:
    # A fixed/vlen-bytes scalar read produces a numpy scalar (`np.bytes_`), not a
    # 0-d ndarray; nothing in the read path may swallow it (`np.bytes_[()]`
    # raises "byte indices must be integers"). Uses the real default engine, no
    # device stand-in. (Distilled from a `test_basic_indexing` hypothesis
    # failure.)
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
