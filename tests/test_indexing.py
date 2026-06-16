from __future__ import annotations

import itertools
from collections import Counter
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

import zarr
from tests.conftest import Expect, ExpectFail
from zarr import Array
from zarr.core.buffer import default_buffer_prototype
from zarr.core.indexing import (
    BasicSelection,
    CoordinateSelection,
    OrthogonalSelection,
    Selection,
    _ArrayIndexingOrder,
    _iter_grid,
    _iter_regions,
    ceildiv,
    make_slice_selection,
    normalize_integer_selection,
    oindex,
    oindex_set,
    replace_ellipsis,
)
from zarr.registry import get_ndbuffer_class
from zarr.storage import MemoryStore, StorePath

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import BufferPrototype
    from zarr.core.buffer.core import Buffer


@pytest.fixture
async def store() -> AsyncGenerator[StorePath]:
    return StorePath(await MemoryStore.open())


def zarr_array_from_numpy_array(
    store: StorePath,
    a: npt.NDArray[Any],
    chunk_shape: tuple[int, ...] | None = None,
) -> zarr.Array:
    z = zarr.create_array(
        store=store / str(uuid4()),
        shape=a.shape,
        dtype=a.dtype,
        chunks=chunk_shape or a.shape,
        chunk_key_encoding={"name": "v2", "separator": "."},
    )
    z[()] = a
    return z


class CountingDict(MemoryStore):
    counter: Counter[tuple[str, str]]

    @classmethod
    async def open(cls) -> CountingDict:
        store = await super().open()
        store.counter = Counter()
        return store

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: tuple[int | None, int | None] | None = None,
    ) -> Buffer | None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__getitem__", key_suffix] += 1
        return await super().get(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__setitem__", key_suffix] += 1
        return await super().set(key, value, byte_range)

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__getitem__", key_suffix] += 1
        return super().get_sync(key, prototype=prototype, byte_range=byte_range)

    def set_sync(self, key: str, value: Buffer) -> None:
        key_suffix = "/".join(key.split("/")[1:])
        self.counter["__setitem__", key_suffix] += 1
        return super().set_sync(key, value)


def test_normalize_integer_selection() -> None:
    """normalize_integer_selection handles positive/negative indices and raises IndexError for out-of-bounds values."""
    assert 1 == normalize_integer_selection(1, 100)
    assert 99 == normalize_integer_selection(-1, 100)
    with pytest.raises(IndexError):
        normalize_integer_selection(100, 100)
    with pytest.raises(IndexError):
        normalize_integer_selection(1000, 100)
    with pytest.raises(IndexError):
        normalize_integer_selection(-1000, 100)


def test_replace_ellipsis() -> None:
    """replace_ellipsis expands Ellipsis to full slice(None) selections for 1D and 2D shapes."""
    # 1D, single item
    assert (0,) == replace_ellipsis(0, (100,))

    # 1D
    assert (slice(None),) == replace_ellipsis(Ellipsis, (100,))
    assert (slice(None),) == replace_ellipsis(slice(None), (100,))
    assert (slice(None, 100),) == replace_ellipsis(slice(None, 100), (100,))
    assert (slice(0, None),) == replace_ellipsis(slice(0, None), (100,))
    assert (slice(None),) == replace_ellipsis((slice(None), Ellipsis), (100,))
    assert (slice(None),) == replace_ellipsis((Ellipsis, slice(None)), (100,))

    # 2D, single item
    assert (0, 0) == replace_ellipsis((0, 0), (100, 100))
    assert (-1, 1) == replace_ellipsis((-1, 1), (100, 100))

    # 2D, single col/row
    assert (0, slice(None)) == replace_ellipsis((0, slice(None)), (100, 100))
    assert (0, slice(None)) == replace_ellipsis((0,), (100, 100))
    assert (slice(None), 0) == replace_ellipsis((slice(None), 0), (100, 100))

    # 2D slice
    assert (slice(None), slice(None)) == replace_ellipsis(Ellipsis, (100, 100))
    assert (slice(None), slice(None)) == replace_ellipsis(slice(None), (100, 100))
    assert (slice(None), slice(None)) == replace_ellipsis((slice(None), slice(None)), (100, 100))
    assert (slice(None), slice(None)) == replace_ellipsis((Ellipsis, slice(None)), (100, 100))
    assert (slice(None), slice(None)) == replace_ellipsis((slice(None), Ellipsis), (100, 100))
    assert (slice(None), slice(None)) == replace_ellipsis(
        (slice(None), Ellipsis, slice(None)), (100, 100)
    )
    assert (slice(None), slice(None)) == replace_ellipsis(
        (Ellipsis, slice(None), slice(None)), (100, 100)
    )
    assert (slice(None), slice(None)) == replace_ellipsis(
        (slice(None), slice(None), Ellipsis), (100, 100)
    )


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (42, "uint8"),
        pytest.param(
            (b"aaa", 1, 4.2),
            [("foo", "S3"), ("bar", "i4"), ("baz", "f8")],
            marks=pytest.mark.xfail(reason="structured dtypes (fields) are not supported in v3"),
        ),
    ],
)
@pytest.mark.parametrize("use_out", [True, False])
def test_get_basic_selection_0d(store: StorePath, use_out: bool, value: Any, dtype: Any) -> None:
    """get_basic_selection on a 0-dimensional array returns the scalar value via Ellipsis and (), including the `out` buffer path."""
    # setup
    arr_np = np.array(value, dtype=dtype)
    arr_z = zarr_array_from_numpy_array(store, arr_np)

    assert_array_equal(arr_np, arr_z.get_basic_selection(Ellipsis))
    assert_array_equal(arr_np, arr_z[...])
    assert value == arr_z.get_basic_selection(())
    assert value == arr_z[()]

    if use_out:
        # test out param
        b = default_buffer_prototype().nd_buffer.from_numpy_array(np.zeros_like(arr_np))
        arr_z.get_basic_selection(Ellipsis, out=b)
        assert_array_equal(arr_np, b.as_ndarray_like())

    # todo: uncomment the structured array tests when we can make them pass,
    # or delete them if we formally decide not to support structured dtypes.

    # test structured array
    # value = (b"aaa", 1, 4.2)
    # a = np.array(value, dtype=[("foo", "S3"), ("bar", "i4"), ("baz", "f8")])
    # z = zarr_array_from_numpy_array(store, a)
    # z[()] = value
    # assert_array_equal(a, z.get_basic_selection(Ellipsis))
    # assert_array_equal(a, z[...])
    # assert a[()] == z.get_basic_selection(())
    # assert a[()] == z[()]
    # assert b"aaa" == z.get_basic_selection((), fields="foo")
    # assert b"aaa" == z["foo"]
    # assert a[["foo", "bar"]] == z.get_basic_selection((), fields=["foo", "bar"])
    # assert a[["foo", "bar"]] == z["foo", "bar"]
    # # test out param
    # b = NDBuffer.from_numpy_array(np.zeros_like(a))
    # z.get_basic_selection(Ellipsis, out=b)
    # assert_array_equal(a, b)
    # c = NDBuffer.from_numpy_array(np.zeros_like(a[["foo", "bar"]]))
    # z.get_basic_selection(Ellipsis, out=c, fields=["foo", "bar"])
    # assert_array_equal(a[["foo", "bar"]], c)


_BASIC_1D_CASES: list[Expect[BasicSelection, None]] = [
    Expect(input=5, output=None, id="single-positive"),
    Expect(input=-1, output=None, id="single-negative"),
    Expect(input=slice(3, 18), output=None, id="bounded-slice"),
    Expect(input=slice(0, 100), output=None, id="over-bounds-slice"),
    Expect(input=slice(-18, -3), output=None, id="negative-slice"),
    Expect(input=slice(0, 0), output=None, id="empty-slice"),
    Expect(input=slice(-1, 0), output=None, id="empty-negative-slice"),
    Expect(input=slice(None), output=None, id="full-slice"),
    Expect(input=Ellipsis, output=None, id="ellipsis"),
    Expect(input=(), output=None, id="empty-tuple"),
    Expect(input=(Ellipsis, slice(None)), output=None, id="ellipsis-slice"),
    Expect(input=slice(None, None, 3), output=None, id="stride-3"),
    Expect(input=slice(3, 27, 5), output=None, id="bounded-stride"),
]

_BASIC_1D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(
        input=slice(None, None, -1),
        exception=IndexError,
        id="negative-step",
        msg="only slices with step >= 1 are supported",
    ),
    ExpectFail(
        input=2.3,
        exception=IndexError,
        id="float",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'float'>",
        escape=True,
    ),
    # get_basic_selection and z[...] word their errors differently for a string
    # selection, so this case asserts only the exception type (msg=None).
    ExpectFail(
        input="foo",
        exception=IndexError,
        id="string",
        msg=None,
    ),
    ExpectFail(
        input=b"xxx",
        exception=IndexError,
        id="bytes",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'bytes'>",
        escape=True,
    ),
    ExpectFail(
        input=None,
        exception=IndexError,
        id="none",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'NoneType'>",
        escape=True,
    ),
    ExpectFail(
        input=(0, 0),
        exception=IndexError,
        id="tuple-too-many",
        msg="too many indices for array; expected 1, got 2",
    ),
    ExpectFail(
        input=(slice(None), slice(None)),
        exception=IndexError,
        id="two-slices",
        msg="too many indices for array; expected 1, got 2",
    ),
]


def _test_get_basic_selection(
    a: npt.NDArray[Any] | Array, z: Array, selection: BasicSelection
) -> None:
    expect = a[selection]
    actual = z.get_basic_selection(selection)
    assert_array_equal(expect, actual)
    actual = z[selection]
    assert_array_equal(expect, actual)

    # test out param
    b = default_buffer_prototype().nd_buffer.from_numpy_array(
        np.empty(shape=expect.shape, dtype=expect.dtype)
    )
    z.get_basic_selection(selection, out=b)
    assert_array_equal(expect, b.as_numpy_array())


@pytest.mark.parametrize("case", _BASIC_1D_CASES, ids=lambda c: c.id)
def test_get_basic_selection_1d(store: StorePath, case: Expect[BasicSelection, None]) -> None:
    """Basic getitem on a 1D array matches numpy for ints, slices, strides, and full selections."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_get_basic_selection(a, z, case.input)


@pytest.mark.parametrize("case", _BASIC_1D_BAD_CASES, ids=lambda c: c.id)
def test_get_basic_selection_1d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """Basic getitem on a 1D array rejects negative steps and invalid index types with IndexError."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.get_basic_selection(case.input)
    with case.raises():
        z[case.input]


_BASIC_2D_CASES: list[Expect[BasicSelection, None]] = [
    Expect(input=5, output=None, id="single-row"),
    Expect(input=-1, output=None, id="single-row-neg"),
    Expect(input=(5, slice(None)), output=None, id="row-and-full-col"),
    Expect(input=(slice(None), 3), output=None, id="single-col"),
    Expect(input=(slice(None), -1), output=None, id="single-col-neg"),
    Expect(input=slice(None), output=None, id="full"),
    Expect(input=slice(2, 9), output=None, id="row-slice"),
    Expect(input=slice(0, 0), output=None, id="empty-row-slice"),
    Expect(input=(slice(2, 9), slice(1, 4)), output=None, id="2d-slice"),
    Expect(input=(slice(0, 12, 3), slice(0, 5, 2)), output=None, id="strided-2d-slice"),
    Expect(input=Ellipsis, output=None, id="ellipsis"),
    Expect(input=(), output=None, id="empty-tuple"),
]

_BASIC_2D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(
        input=2.3,
        exception=IndexError,
        id="float",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'float'>",
        escape=True,
    ),
    ExpectFail(
        input="foo",
        exception=IndexError,
        id="string",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'str'>",
        escape=True,
    ),
    ExpectFail(
        input=None,
        exception=IndexError,
        id="none",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'NoneType'>",
        escape=True,
    ),
    ExpectFail(
        input=(2.3, slice(None)),
        exception=IndexError,
        id="float-in-tuple",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'float'>",
        escape=True,
    ),
    ExpectFail(
        input=slice(None, None, -1),
        exception=IndexError,
        id="negative-step",
        msg="only slices with step >= 1 are supported",
    ),
    ExpectFail(
        input=(slice(None), slice(None), slice(None)),
        exception=IndexError,
        id="too-many-dims",
        msg="too many indices for array; expected 2, got 3",
    ),
    ExpectFail(
        input=[0, 1],
        exception=IndexError,
        id="integer-list",
        msg="unsupported selection item for basic indexing; expected integer or slice, got <class 'list'>",
        escape=True,
    ),
]


@pytest.mark.parametrize("case", _BASIC_2D_CASES, ids=lambda c: c.id)
def test_get_basic_selection_2d(store: StorePath, case: Expect[BasicSelection, None]) -> None:
    """Basic getitem on a 2D array matches numpy for rows, cols, slices, and strides."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_get_basic_selection(a, z, case.input)


@pytest.mark.parametrize("case", _BASIC_2D_BAD_CASES, ids=lambda c: c.id)
def test_get_basic_selection_2d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """Basic getitem on a 2D array rejects malformed selections with IndexError."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    with case.raises():
        z.get_basic_selection(case.input)


def test_basic_2d_fancy_fallback(store: StorePath) -> None:
    """Indexing a 2D array with paired integer lists falls back to fancy (vectorized) indexing."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    np.testing.assert_array_equal(z[([0, 1], [0, 1])], a[([0, 1], [0, 1])])


def test_get_basic_selection_1d_rejects_integer_list(store: StorePath) -> None:
    """get_basic_selection on a 1D array rejects an integer list (basic indexing is int/slice only)."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with pytest.raises(IndexError, match="unsupported selection item for basic indexing"):
        z.get_basic_selection([1, 0])


def test_get_basic_selection_2d_rejects_list_in_tuple(store: StorePath) -> None:
    """get_basic_selection on a 2D array rejects a list nested in an index tuple."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    with pytest.raises(IndexError, match="unsupported selection item for basic indexing"):
        z.get_basic_selection((slice(None), [0, 1]))


def test_fancy_indexing_fallback_on_get_setitem(store: StorePath) -> None:
    """Paired integer-list indexing falls back to vectorized (fancy) get and set via `__getitem__`/`__setitem__`."""
    z = zarr_array_from_numpy_array(store, np.zeros((20, 20)))
    z[[1, 2, 3], [1, 2, 3]] = 1
    np.testing.assert_array_equal(
        z[:4, :4],
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    np.testing.assert_array_equal(z[[1, 2, 3], [1, 2, 3]], 1)
    # test broadcasting
    np.testing.assert_array_equal(z[1, [1, 2, 3]], [1, 0, 0])
    # test 1D fancy indexing
    z2 = zarr_array_from_numpy_array(store, np.zeros(5))
    z2[[1, 2, 3]] = 1
    np.testing.assert_array_equal(z2[:], [0, 1, 1, 1, 0])


@pytest.mark.parametrize(
    ("index", "expected_result"),
    [
        # Single iterable of integers
        ([0, 1], [[0, 1, 2], [3, 4, 5]]),
        # List first, then slice
        (([0, 1], slice(None)), [[0, 1, 2], [3, 4, 5]]),
        # List first, then slice
        (([0, 1], slice(1, None)), [[1, 2], [4, 5]]),
        # Slice first, then list
        ((slice(0, 2), [0, 2]), [[0, 2], [3, 5]]),
        # Slices only
        ((slice(0, 2), slice(0, 2)), [[0, 1], [3, 4]]),
        # List with repeated index
        (([1, 0, 1], slice(1, None)), [[4, 5], [1, 2], [4, 5]]),
        # 1D indexing
        (([1, 0, 1]), [[3, 4, 5], [0, 1, 2], [3, 4, 5]]),
    ],
)
def test_orthogonal_indexing_fallback_on_getitem_2d(
    store: StorePath, index: Selection, expected_result: npt.ArrayLike
) -> None:
    """
    Tests the orthogonal indexing fallback on __getitem__ for a 2D matrix.

    In addition to checking expected behavior, all indexing
    is also checked against numpy.
    """
    # [0, 1, 2],
    # [3, 4, 5],
    # [6, 7, 8]
    a = np.arange(9).reshape(3, 3)
    z = zarr_array_from_numpy_array(store, a)

    np.testing.assert_array_equal(z[index], a[index], err_msg="Indexing disagrees with numpy")
    np.testing.assert_array_equal(z[index], expected_result)


def test_setitem_zarr_array_as_value() -> None:
    """Assigning a zarr array as a value in `__setitem__` does not raise a SyncError (regression for GH3611)."""
    # Regression test for https://github.com/zarr-developers/zarr-python/issues/3611
    # Assigning a zarr Array as the value used to raise
    # SyncError("Calling sync() from within a running loop") because the codec
    # pipeline tried to index the zarr array inside an already-running async loop.
    src = zarr.array(np.arange(10), chunks=(5,))
    dst = zarr.zeros(10, chunks=(5,), dtype=src.dtype)

    # Full assignment
    dst[:] = src
    assert_array_equal(dst[:], np.arange(10))

    # Slice assignment
    dst2 = zarr.zeros(10, chunks=(5,), dtype=src.dtype)
    dst2[2:7] = src[2:7]
    assert_array_equal(dst2[2:7], np.arange(2, 7))


@pytest.mark.skip(reason="fails on ubuntu, windows; numpy=2.2; in CI")
def test_setitem_repeated_index():
    """oindex assignment with repeated indices writes the last value for each duplicated index position."""
    array = zarr.array(data=np.zeros((4,)), chunks=(1,))
    indexer = np.array([-1, -1, 0, 0])
    array.oindex[(indexer,)] = [0, 1, 2, 3]
    np.testing.assert_array_equal(array[:], np.array([3, 0, 0, 1]))

    indexer = np.array([-1, 0, 0, -1])
    array.oindex[(indexer,)] = [0, 1, 2, 3]
    np.testing.assert_array_equal(array[:], np.array([2, 0, 0, 3]))


Index = list[int] | tuple[slice | int | list[int], ...]


@pytest.mark.parametrize(
    ("index", "expected_result"),
    [
        # Single iterable of integers
        ([0, 1], [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14], [15, 16, 17]]]),
        # One slice, two integers
        ((slice(0, 2), 1, 1), [4, 13]),
        # One integer, two slices
        ((slice(0, 2), 1, slice(0, 2)), [[3, 4], [12, 13]]),
        # Two slices and a list
        ((slice(0, 2), [1, 2], slice(0, 2)), [[[3, 4], [6, 7]], [[12, 13], [15, 16]]]),
    ],
)
def test_orthogonal_indexing_fallback_on_getitem_3d(
    store: StorePath, index: Selection, expected_result: npt.ArrayLike
) -> None:
    """
    Tests the orthogonal indexing fallback on __getitem__ for a 3D matrix.

    In addition to checking expected behavior, all indexing
    is also checked against numpy.
    """
    #   [[[ 0,  1,  2],
    #     [ 3,  4,  5],
    #     [ 6,  7,  8]],

    #    [[ 9, 10, 11],
    #     [12, 13, 14],
    #     [15, 16, 17]],

    #    [[18, 19, 20],
    #     [21, 22, 23],
    #     [24, 25, 26]]]
    a = np.arange(27).reshape(3, 3, 3)
    z = zarr_array_from_numpy_array(store, a)

    np.testing.assert_array_equal(z[index], a[index], err_msg="Indexing disagrees with numpy")
    np.testing.assert_array_equal(z[index], expected_result)


@pytest.mark.parametrize(
    ("index", "expected_result"),
    [
        # Single iterable of integers
        ([0, 1], [[1, 1, 1], [1, 1, 1], [0, 0, 0]]),
        # List and slice combined
        (([0, 1], slice(1, 3)), [[0, 1, 1], [0, 1, 1], [0, 0, 0]]),
        # Index repetition is ignored on setitem
        (([0, 1, 1, 1, 1, 1, 1], slice(1, 3)), [[0, 1, 1], [0, 1, 1], [0, 0, 0]]),
        # Slice with step
        (([0, 2], slice(None, None, 2)), [[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
    ],
)
def test_orthogonal_indexing_fallback_on_setitem_2d(
    store: StorePath, index: Selection, expected_result: npt.ArrayLike
) -> None:
    """
    Tests the orthogonal indexing fallback on __setitem__ for a 3D matrix.

    In addition to checking expected behavior, all indexing
    is also checked against numpy.
    """
    # Slice + fancy index
    a = np.zeros((3, 3))
    z = zarr_array_from_numpy_array(store, a)
    z[index] = 1
    a[index] = 1
    np.testing.assert_array_equal(z[:], expected_result)
    np.testing.assert_array_equal(z[:], a, err_msg="Indexing disagrees with numpy")


def test_fancy_indexing_doesnt_mix_with_implicit_slicing(store: StorePath) -> None:
    """Fancy indexing that would require implicit slicing over an unspecified axis raises IndexError on a 3D array."""
    z2 = zarr_array_from_numpy_array(store, np.zeros((5, 5, 5)))
    with pytest.raises(IndexError):
        z2[[1, 2, 3], [1, 2, 3]] = 2
    with pytest.raises(IndexError):
        np.testing.assert_array_equal(z2[[1, 2, 3], [1, 2, 3]], 0)
    with pytest.raises(IndexError):
        z2[..., [1, 2, 3]] = 2  # type: ignore[index]
    with pytest.raises(IndexError):
        np.testing.assert_array_equal(z2[..., [1, 2, 3]], 0)  # type: ignore[index]


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (42, "uint8"),
        pytest.param(
            (b"aaa", 1, 4.2),
            [("foo", "S3"), ("bar", "i4"), ("baz", "f8")],
            marks=pytest.mark.xfail(reason="structured dtypes (fields) are not supported in v3"),
        ),
    ],
)
def test_set_basic_selection_0d(
    store: StorePath, value: Any, dtype: str | list[tuple[str, str]]
) -> None:
    """set_basic_selection and `__setitem__` write scalar values correctly to a 0-dimensional array."""
    arr_np = np.array(value, dtype=dtype)
    arr_np_zeros = np.zeros_like(arr_np, dtype=dtype)
    arr_z = zarr_array_from_numpy_array(store, arr_np_zeros)
    assert_array_equal(arr_np_zeros, arr_z)

    arr_z.set_basic_selection(Ellipsis, value)
    assert_array_equal(value, arr_z)
    arr_z[...] = 0
    assert_array_equal(arr_np_zeros, arr_z)
    arr_z[...] = value
    assert_array_equal(value, arr_z)

    # todo: uncomment the structured array tests when we can make them pass,
    # or delete them if we formally decide not to support structured dtypes.

    # arr_z.set_basic_selection(Ellipsis, v["foo"], fields="foo")
    # assert v["foo"] == arr_z["foo"]
    # assert arr_np_zeros["bar"] == arr_z["bar"]
    # assert arr_np_zeros["baz"] == arr_z["baz"]
    # arr_z["bar"] = v["bar"]
    # assert v["foo"] == arr_z["foo"]
    # assert v["bar"] == arr_z["bar"]
    # assert arr_np_zeros["baz"] == arr_z["baz"]
    # # multiple field assignment not supported
    # with pytest.raises(IndexError):
    #     arr_z.set_basic_selection(Ellipsis, v[["foo", "bar"]], fields=["foo", "bar"])
    # with pytest.raises(IndexError):
    #     arr_z[..., "foo", "bar"] = v[["foo", "bar"]]


def _test_get_orthogonal_selection(
    a: npt.NDArray[Any], z: Array, selection: OrthogonalSelection
) -> None:
    expect = oindex(a, selection)
    actual = z.get_orthogonal_selection(selection)
    assert_array_equal(expect, actual)
    actual = z.oindex[selection]
    assert_array_equal(expect, actual)


_ORTHO_1D_BOOL_CASES: list[Expect[OrthogonalSelection, None]] = [
    Expect(input=np.zeros(30, dtype=bool), output=None, id="empty-mask"),
    Expect(input=np.ones(30, dtype=bool), output=None, id="full-mask"),
    Expect(input=np.arange(30) % 2 == 0, output=None, id="alternating-mask"),
    Expect(input=np.arange(30) == 7, output=None, id="single-true"),
    Expect(
        input=np.isin(np.arange(30), [0, 1, 8, 15, 29]),
        output=None,
        id="sparse-cross-chunk",
    ),
]

_ORTHO_1D_BOOL_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(
        input=np.zeros(5, dtype=bool),
        exception=IndexError,
        id="mask-too-short",
        msg="wrong length for dimension; expected 30, got 5",
    ),
    ExpectFail(
        input=np.zeros(50, dtype=bool),
        exception=IndexError,
        id="mask-too-long",
        msg="wrong length for dimension; expected 30, got 50",
    ),
    ExpectFail(
        input=[[True, False], [False, True]],
        exception=IndexError,
        id="mask-too-many-dims",
        msg="must be 1-dimensional only",
    ),
]


@pytest.mark.parametrize("case", _ORTHO_1D_BOOL_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_1d_bool(
    store: StorePath, case: Expect[OrthogonalSelection, None]
) -> None:
    """oindex with a 1D boolean mask matches numpy across chunk boundaries."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_get_orthogonal_selection(a, z, case.input)


@pytest.mark.parametrize("case", _ORTHO_1D_BOOL_BAD_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_1d_bool_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """oindex rejects masks of the wrong length or dimensionality with IndexError."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.oindex[case.input]


_ORTHO_1D_INT_CASES: list[Expect[OrthogonalSelection, None]] = [
    Expect(input=[0, 8, 15, 29], output=None, id="sorted"),
    Expect(input=[3, 29, 1, 16], output=None, id="unsorted"),
    Expect(input=[2, 2, 8, 8], output=None, id="duplicates"),
    Expect(input=[0, 3, 10, -23, -12, -1], output=None, id="wraparound"),
    Expect(input=[15], output=None, id="single"),
]

_ORTHO_1D_INT_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(
        input=[31],
        exception=IndexError,
        id="out-of-bounds-high",
        msg="index out of bounds for dimension with length 30",
    ),
    ExpectFail(
        input=[-31],
        exception=IndexError,
        id="out-of-bounds-low",
        msg="index out of bounds for dimension with length 30",
    ),
    ExpectFail(
        input=[[2, 4], [6, 8]],
        exception=IndexError,
        id="too-many-dims",
        msg="integer arrays in an orthogonal selection must be 1-dimensional only",
    ),
]


@pytest.mark.parametrize("case", _ORTHO_1D_INT_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_1d_int(
    store: StorePath, case: Expect[OrthogonalSelection, None]
) -> None:
    """oindex with a 1D integer array matches numpy, including wraparound and duplicates."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_get_orthogonal_selection(a, z, case.input)


@pytest.mark.parametrize("case", _ORTHO_1D_INT_BAD_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_1d_int_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """oindex rejects out-of-bounds or multi-dimensional integer selections with IndexError."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.get_orthogonal_selection(case.input)
    with case.raises():
        z.oindex[case.input]


_ORTHO_2D_IX0_BOOL = np.isin(np.arange(12), [0, 5, 11])  # rows 0, 5, 11
_ORTHO_2D_IX1_BOOL = np.array([True, False, True, False, True])  # cols 0, 2, 4
_ORTHO_2D_IX0_INT = np.array([0, 5, 11])
_ORTHO_2D_IX1_INT = np.array([0, 2, 4])

_ORTHO_2D_CASES: list[Expect[OrthogonalSelection, None]] = [
    Expect(input=(_ORTHO_2D_IX0_BOOL, _ORTHO_2D_IX1_BOOL), output=None, id="both-bool"),
    Expect(input=(_ORTHO_2D_IX0_BOOL, slice(1, 4)), output=None, id="bool-slice"),
    Expect(input=(_ORTHO_2D_IX0_BOOL, slice(0, 5, 2)), output=None, id="bool-strided-slice"),
    Expect(input=(slice(2, 9), _ORTHO_2D_IX1_BOOL), output=None, id="slice-bool"),
    Expect(input=(slice(0, 12, 4), _ORTHO_2D_IX1_BOOL), output=None, id="strided-slice-bool"),
    Expect(input=(_ORTHO_2D_IX0_BOOL, 3), output=None, id="bool-int"),
    Expect(input=(7, _ORTHO_2D_IX1_BOOL), output=None, id="int-bool"),
    Expect(input=(_ORTHO_2D_IX0_INT, _ORTHO_2D_IX1_INT), output=None, id="both-int"),
    Expect(input=(_ORTHO_2D_IX0_INT, _ORTHO_2D_IX1_BOOL), output=None, id="int-array-bool-array"),
    Expect(input=(_ORTHO_2D_IX0_BOOL, _ORTHO_2D_IX1_INT), output=None, id="bool-array-int-array"),
    Expect(input=7, output=None, id="single-row"),
    Expect(input=(slice(None), 3), output=None, id="single-col"),
    Expect(input=(slice(None), slice(None)), output=None, id="full"),
    Expect(input=slice(2, 9), output=None, id="row-slice"),
]

_ORTHO_2D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(
        input=2.3,
        exception=IndexError,
        id="float-index",
        msg="unsupported selection item for orthogonal indexing",
    ),
    # get_orthogonal_selection and oindex raise different messages for a string
    # selection, so assert only the exception type.
    ExpectFail(
        input="foo",
        exception=IndexError,
        id="string-index",
        msg=None,
    ),
    ExpectFail(
        input=None,
        exception=IndexError,
        id="none-index",
        msg="unsupported selection item for orthogonal indexing",
    ),
    ExpectFail(
        input=slice(None, None, -1),
        exception=IndexError,
        id="negative-step",
        msg="only slices with step >= 1 are supported",
    ),
    ExpectFail(
        input=(0, 0, 0),
        exception=IndexError,
        id="too-many-dims",
        msg="too many indices for array",
    ),
]


@pytest.mark.parametrize("case", _ORTHO_2D_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_2d(
    store: StorePath, case: Expect[OrthogonalSelection, None]
) -> None:
    """oindex on a 2D array matches numpy for array/slice/int combinations per axis."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_get_orthogonal_selection(a, z, case.input)


@pytest.mark.parametrize("case", _ORTHO_2D_BAD_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_2d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """oindex on a 2D array rejects malformed selections with IndexError."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    with case.raises():
        z.get_orthogonal_selection(case.input)
    with case.raises():
        z.oindex[case.input]


_ORTHO_3D_IX0_BOOL = np.isin(np.arange(7), [0, 3, 6])  # axis 0
_ORTHO_3D_IX1_BOOL = np.isin(np.arange(6), [0, 2, 5])  # axis 1
_ORTHO_3D_IX2_BOOL = np.isin(np.arange(10), [0, 4, 9])  # axis 2
_ORTHO_3D_IX0_INT = np.array([0, 3, 6])
_ORTHO_3D_IX1_INT = np.array([0, 2, 5])
_ORTHO_3D_IX2_INT = np.array([0, 4, 9])

_ORTHO_3D_CASES: list[Expect[OrthogonalSelection, None]] = [
    # single value
    Expect(input=(5, 3, 8), output=None, id="single-value"),
    Expect(input=(-1, -1, -1), output=None, id="all-negative"),
    # index all axes with arrays
    Expect(
        input=(_ORTHO_3D_IX0_BOOL, _ORTHO_3D_IX1_BOOL, _ORTHO_3D_IX2_BOOL),
        output=None,
        id="three-bool-arrays",
    ),
    Expect(
        input=(_ORTHO_3D_IX0_INT, _ORTHO_3D_IX1_INT, _ORTHO_3D_IX2_INT),
        output=None,
        id="three-int-arrays",
    ),
    # mixed indexing with single array / slices
    Expect(
        input=(_ORTHO_3D_IX0_BOOL, slice(1, 5), slice(2, 9)), output=None, id="array-slice-slice"
    ),
    Expect(
        input=(slice(1, 6), _ORTHO_3D_IX1_BOOL, slice(2, 9)), output=None, id="slice-array-slice"
    ),
    Expect(
        input=(slice(1, 6), slice(1, 5), _ORTHO_3D_IX2_BOOL), output=None, id="slice-slice-array"
    ),
    Expect(
        input=(_ORTHO_3D_IX0_BOOL, slice(0, 6, 2), slice(0, 10, 3)),
        output=None,
        id="array-strided-strided",
    ),
    Expect(
        input=(slice(0, 7, 2), _ORTHO_3D_IX1_BOOL, slice(0, 10, 3)),
        output=None,
        id="strided-array-strided",
    ),
    Expect(
        input=(slice(0, 7, 2), slice(0, 6, 2), _ORTHO_3D_IX2_BOOL),
        output=None,
        id="strided-strided-array",
    ),
    # mixed indexing with single array / ints
    Expect(input=(_ORTHO_3D_IX0_BOOL, 3, 8), output=None, id="array-int-int"),
    Expect(input=(5, _ORTHO_3D_IX1_BOOL, 8), output=None, id="int-array-int"),
    Expect(input=(5, 3, _ORTHO_3D_IX2_BOOL), output=None, id="int-int-array"),
    # mixed indexing with single array / slice / int
    Expect(input=(_ORTHO_3D_IX0_BOOL, slice(1, 5), 8), output=None, id="array-slice-int"),
    Expect(input=(5, _ORTHO_3D_IX1_BOOL, slice(2, 9)), output=None, id="int-array-slice"),
    Expect(input=(slice(1, 6), 3, _ORTHO_3D_IX2_BOOL), output=None, id="slice-int-array"),
    # mixed indexing with two arrays / slice
    Expect(
        input=(_ORTHO_3D_IX0_BOOL, _ORTHO_3D_IX1_BOOL, slice(2, 9)),
        output=None,
        id="two-arrays-slice",
    ),
    Expect(
        input=(slice(1, 6), _ORTHO_3D_IX1_BOOL, _ORTHO_3D_IX2_BOOL),
        output=None,
        id="slice-two-arrays",
    ),
    Expect(
        input=(_ORTHO_3D_IX0_BOOL, slice(1, 5), _ORTHO_3D_IX2_BOOL),
        output=None,
        id="array-slice-array",
    ),
    # mixed indexing with two arrays / integer
    Expect(input=(_ORTHO_3D_IX0_BOOL, _ORTHO_3D_IX1_BOOL, 8), output=None, id="two-arrays-int"),
    Expect(input=(5, _ORTHO_3D_IX1_BOOL, _ORTHO_3D_IX2_BOOL), output=None, id="int-two-arrays"),
    Expect(input=(_ORTHO_3D_IX0_BOOL, 3, _ORTHO_3D_IX2_BOOL), output=None, id="array-int-array"),
]


@pytest.mark.parametrize("case", _ORTHO_3D_CASES, ids=lambda c: c.id)
def test_get_orthogonal_selection_3d(
    store: StorePath, case: Expect[OrthogonalSelection, None]
) -> None:
    """oindex on a 3D array matches numpy for array/slice/int combinations per axis."""
    a = np.arange(420, dtype=int).reshape(7, 6, 10)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(3, 2, 4))
    _test_get_orthogonal_selection(a, z, case.input)


def test_orthogonal_indexing_edge_cases(store: StorePath) -> None:
    """oindex on a shape-(1, 2, 3) array correctly handles mixing integer, slice, int-list, and bool-list indexers per axis."""
    a = np.arange(6).reshape(1, 2, 3)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(1, 2, 3))

    expect = oindex(a, (0, slice(None), [0, 1, 2]))
    actual = z.oindex[0, :, [0, 1, 2]]
    assert_array_equal(expect, actual)

    expect = oindex(a, (0, slice(None), [True, True, True]))
    actual = z.oindex[0, :, [True, True, True]]
    assert_array_equal(expect, actual)


def _test_set_orthogonal_selection(
    v: npt.NDArray[np.int_], a: npt.NDArray[Any], z: Array, selection: OrthogonalSelection
) -> None:
    for value in 42, oindex(v, selection), oindex(v, selection).tolist():
        if isinstance(value, list) and value == []:
            # skip these cases as cannot preserve all dimensions
            continue
        # setup expectation
        a[:] = 0
        oindex_set(a, selection, value)
        # long-form API
        z[:] = 0
        z.set_orthogonal_selection(selection, value)
        assert_array_equal(a, z[:])
        # short-form API
        z[:] = 0
        z.oindex[selection] = value
        assert_array_equal(a, z[:])


@pytest.mark.parametrize("case", _ORTHO_1D_BOOL_CASES + _ORTHO_1D_INT_CASES, ids=lambda c: c.id)
def test_set_orthogonal_selection_1d(
    store: StorePath, case: Expect[OrthogonalSelection, None]
) -> None:
    """set_orthogonal_selection on a 1D array round-trips through numpy for masks and int arrays."""
    v = np.arange(30, dtype=int)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_set_orthogonal_selection(v, a, z, case.input)


def test_set_item_1d_last_two_chunks(store: StorePath):
    """Regression for GH2849: `__setitem__` correctly writes to the last two chunks of a 1D array and to 0-dimensional scalar arrays."""
    # regression test for GH2849
    g = zarr.open_group(store=store, zarr_format=3, mode="w")
    a = g.create_array("bar", shape=(10,), chunks=(3,), dtype=int)
    data = np.array([7, 8, 9])
    a[slice(7, 10)] = data
    np.testing.assert_array_equal(a[slice(7, 10)], data)

    z = zarr.open_group(store=store, mode="w")
    z.create_array("zoo", dtype=float, shape=())
    z["zoo"][...] = np.array(1)  # why doesn't [:] work?
    np.testing.assert_equal(z["zoo"][()], np.array(1))

    z = zarr.open_group(store=store, mode="w")
    z.create_array("zoo", dtype=float, shape=())
    z["zoo"][...] = 1  # why doesn't [:] work?
    np.testing.assert_equal(z["zoo"][()], np.array(1))


@pytest.mark.parametrize("case", _ORTHO_2D_CASES, ids=lambda c: c.id)
def test_set_orthogonal_selection_2d(
    store: StorePath, case: Expect[OrthogonalSelection, None]
) -> None:
    """set_orthogonal_selection on a 2D array round-trips through numpy."""
    v = np.arange(60, dtype=int).reshape(12, 5)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_set_orthogonal_selection(v, a, z, case.input)


@pytest.mark.parametrize("case", _ORTHO_3D_CASES, ids=lambda c: c.id)
def test_set_orthogonal_selection_3d(
    store: StorePath, case: Expect[OrthogonalSelection, None]
) -> None:
    """set_orthogonal_selection on a 3D array round-trips through numpy."""
    v = np.arange(420, dtype=int).reshape(7, 6, 10)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(3, 2, 4))
    _test_set_orthogonal_selection(v, a, z, case.input)


def test_orthogonal_indexing_fallback_on_get_setitem(store: StorePath) -> None:
    """Paired integer-list indexing on a 2D array falls back to orthogonal get and set via `__getitem__`/`__setitem__`."""
    z = zarr_array_from_numpy_array(store, np.zeros((20, 20)))
    z[[1, 2, 3], [1, 2, 3]] = 1
    np.testing.assert_array_equal(
        z[:4, :4],
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    np.testing.assert_array_equal(z[[1, 2, 3], [1, 2, 3]], 1)
    # test broadcasting
    np.testing.assert_array_equal(z[1, [1, 2, 3]], [1, 0, 0])
    # test 1D fancy indexing
    z2 = zarr_array_from_numpy_array(store, np.zeros(5))
    z2[[1, 2, 3]] = 1
    np.testing.assert_array_equal(z2[:], [0, 1, 1, 1, 0])


def _test_get_coordinate_selection(
    a: npt.NDArray, z: Array, selection: CoordinateSelection
) -> None:
    expect = a[selection]
    actual = z.get_coordinate_selection(selection)
    assert_array_equal(expect, actual)
    actual = z.vindex[selection]
    assert_array_equal(expect, actual)


_COORD_1D_CASES: list[Expect[CoordinateSelection, None]] = [
    Expect(input=5, output=None, id="single"),
    Expect(input=-1, output=None, id="single-negative"),
    Expect(input=[0, 3, 10, -23, -12, -1], output=None, id="wraparound"),
    Expect(input=[3, 25, 8, 17], output=None, id="out-of-order"),
    Expect(input=[1, 8, 15, 29], output=None, id="sorted"),
    Expect(input=[29, 15, 8, 1], output=None, id="reversed"),
    Expect(input=[2, 2, 8, 8], output=None, id="duplicates"),
    Expect(input=np.array([[2, 4], [6, 8]]), output=None, id="multi-dim"),
]

# get_coordinate_selection and vindex word their errors differently for these
# invalid-type inputs, so these cases assert only the exception type (msg=None).
_COORD_1D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(input=slice(5, 15), exception=IndexError, id="slice", msg=None),
    ExpectFail(input=slice(None), exception=IndexError, id="full-slice", msg=None),
    ExpectFail(input=Ellipsis, exception=IndexError, id="ellipsis", msg=None),
    ExpectFail(input=2.3, exception=IndexError, id="float", msg=None),
    ExpectFail(input="foo", exception=IndexError, id="string", msg=None),
    ExpectFail(input=b"xxx", exception=IndexError, id="bytes", msg=None),
    ExpectFail(input=None, exception=IndexError, id="none", msg=None),
    ExpectFail(input=(0, 0), exception=IndexError, id="tuple-pair", msg=None),
    ExpectFail(input=(slice(None), slice(None)), exception=IndexError, id="two-slices", msg=None),
    ExpectFail(
        input=[31],
        exception=IndexError,
        id="out-of-bounds-high",
        msg="index out of bounds for dimension with length 30",
    ),
    ExpectFail(
        input=[-31],
        exception=IndexError,
        id="out-of-bounds-low",
        msg="index out of bounds for dimension with length 30",
    ),
]

_COORD_2D_IX0 = np.array([0, 5, 11, 2, 8])
_COORD_2D_IX1 = np.array([1, 3, 4, 0, 2])

_COORD_2D_CASES: list[Expect[CoordinateSelection, None]] = [
    Expect(input=(5, 4), output=None, id="single"),
    Expect(input=(-1, -1), output=None, id="single-negative"),
    Expect(input=(_COORD_2D_IX0, _COORD_2D_IX1), output=None, id="both-arrays"),
    # scalar broadcasts in coordinate indexing (numpy and zarr agree)
    Expect(input=(np.array([0, 5, 11]), 4), output=None, id="array-int"),
    Expect(input=(7, np.array([0, 2, 4])), output=None, id="int-array"),
    Expect(input=([3, 3, 4, 2, 5], [1, 3, 4, 0, 2]), output=None, id="not-monotonic-first"),
    Expect(input=([1, 1, 2, 2, 5], [1, 3, 2, 1, 0]), output=None, id="not-monotonic-second"),
    Expect(
        input=(np.array([[1, 1, 2], [2, 2, 5]]), np.array([[1, 3, 2], [1, 0, 0]])),
        output=None,
        id="multi-dim",
    ),
]

_COORD_2D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(
        input=(slice(5, 15), [1, 2, 3]),
        exception=IndexError,
        id="slice-with-array",
        msg=None,
    ),
    ExpectFail(
        input=([1, 2, 3], slice(5, 15)),
        exception=IndexError,
        id="array-with-slice",
        msg=None,
    ),
    ExpectFail(
        input=(Ellipsis, [1, 2, 3]),
        exception=IndexError,
        id="ellipsis-with-array",
        msg=None,
    ),
    ExpectFail(input=Ellipsis, exception=IndexError, id="ellipsis", msg=None),
    ExpectFail(
        input=(np.array([12]), np.array([0])),
        exception=IndexError,
        id="out-of-bounds-axis0",
        msg="index out of bounds for dimension with length 12",
    ),
    ExpectFail(
        input=(np.array([0]), np.array([5])),
        exception=IndexError,
        id="out-of-bounds-axis1",
        msg="index out of bounds for dimension with length 5",
    ),
]


@pytest.mark.parametrize("case", _COORD_1D_CASES, ids=lambda c: c.id)
def test_get_coordinate_selection_1d(
    store: StorePath, case: Expect[CoordinateSelection, None]
) -> None:
    """vindex and get_coordinate_selection on a 1D array match numpy for int, list, and multi-dim selections."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_get_coordinate_selection(a, z, case.input)


@pytest.mark.parametrize("case", _COORD_1D_BAD_CASES, ids=lambda c: c.id)
def test_get_coordinate_selection_1d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """get_coordinate_selection and vindex both raise IndexError for invalid 1D selections."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.get_coordinate_selection(case.input)  # type: ignore[arg-type]
    with case.raises():
        z.vindex[case.input]  # type: ignore[index]


@pytest.mark.parametrize("case", _COORD_2D_CASES, ids=lambda c: c.id)
def test_get_coordinate_selection_2d(
    store: StorePath, case: Expect[CoordinateSelection, None]
) -> None:
    """vindex and get_coordinate_selection on a 2D array match numpy for coordinate selections."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_get_coordinate_selection(a, z, case.input)


@pytest.mark.parametrize("case", _COORD_2D_BAD_CASES, ids=lambda c: c.id)
def test_get_coordinate_selection_2d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """get_coordinate_selection raises IndexError when slices or Ellipsis appear in a 2D coordinate selection."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    with case.raises():
        z.get_coordinate_selection(case.input)  # type: ignore[arg-type]


def _test_set_coordinate_selection(
    v: npt.NDArray, a: npt.NDArray, z: Array, selection: CoordinateSelection
) -> None:
    for value in 42, v[selection], v[selection].tolist():
        # setup expectation
        a[:] = 0
        a[selection] = value
        # test long-form API
        z[:] = 0
        z.set_coordinate_selection(selection, value)
        assert_array_equal(a, z[:])
        # test short-form API
        z[:] = 0
        z.vindex[selection] = value
        assert_array_equal(a, z[:])


@pytest.mark.parametrize("case", _COORD_1D_CASES, ids=lambda c: c.id)
def test_set_coordinate_selection_1d(
    store: StorePath, case: Expect[CoordinateSelection, None]
) -> None:
    """set_coordinate_selection and vindex assignment on a 1D array round-trip through numpy."""
    v = np.arange(30, dtype=int)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_set_coordinate_selection(v, a, z, case.input)


@pytest.mark.parametrize("case", _COORD_2D_CASES, ids=lambda c: c.id)
def test_set_coordinate_selection_2d(
    store: StorePath, case: Expect[CoordinateSelection, None]
) -> None:
    """set_coordinate_selection and vindex assignment on a 2D array round-trip through numpy."""
    v = np.arange(60, dtype=int).reshape(12, 5)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_set_coordinate_selection(v, a, z, case.input)


def _test_get_block_selection(
    a: npt.NDArray[Any],
    z: Array,
    selection: BasicSelection,
    expected_idx: slice | tuple[slice, ...],
) -> None:
    expect = a[expected_idx]
    actual = z.get_block_selection(selection)
    assert_array_equal(expect, actual)
    actual = z.blocks[selection]
    assert_array_equal(expect, actual)


_BLOCK_1D_CASES: list[Expect[BasicSelection, slice]] = [
    Expect(input=0, output=slice(0, 7), id="block-0"),
    Expect(input=2, output=slice(14, 21), id="block-mid"),
    Expect(input=4, output=slice(28, 30), id="block-last"),
    Expect(input=-1, output=slice(28, 30), id="block-neg-1"),
    Expect(input=-2, output=slice(21, 28), id="block-neg-2"),
    Expect(input=slice(3), output=slice(0, 21), id="slice-to-3"),
    Expect(input=slice(None, 2), output=slice(0, 14), id="slice-none-2"),
    Expect(input=slice(1, 2), output=slice(7, 14), id="slice-1-2"),
    Expect(input=slice(-2, -1), output=slice(21, 28), id="slice-neg"),
    Expect(input=slice(None), output=slice(0, 30), id="full"),
]

_BLOCK_1D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(input=slice(3, 8, 2), exception=IndexError, id="strided-slice"),
    ExpectFail(input=2.3, exception=IndexError, id="float"),
    ExpectFail(input=b"xxx", exception=IndexError, id="bytes"),
    ExpectFail(input=None, exception=IndexError, id="none"),
    ExpectFail(input=(0, 0), exception=IndexError, id="tuple-pair"),
    ExpectFail(input=(slice(None), slice(None)), exception=IndexError, id="two-slices"),
    ExpectFail(input=[0, 5, 3], exception=IndexError, id="int-list"),
    ExpectFail(input=5, exception=IndexError, id="out-of-bounds-high"),
    ExpectFail(input=-6, exception=IndexError, id="out-of-bounds-low"),
]

_BLOCK_2D_CASES: list[Expect[BasicSelection, tuple[slice, slice]]] = [
    Expect(input=(0, 0), output=(slice(0, 5), slice(0, 2)), id="single-00"),
    Expect(input=(1, 1), output=(slice(5, 10), slice(2, 4)), id="single-mid"),
    Expect(input=(-1, -1), output=(slice(10, 12), slice(4, 5)), id="neg"),
    Expect(input=(slice(0, 2), 0), output=(slice(0, 10), slice(0, 2)), id="slice-rows"),
    Expect(input=(2, slice(1, 3)), output=(slice(10, 12), slice(2, 5)), id="slice-cols"),
    Expect(input=(slice(0, 2), slice(0, 2)), output=(slice(0, 10), slice(0, 4)), id="both-slices"),
    Expect(input=(slice(None), slice(None)), output=(slice(0, 12), slice(0, 5)), id="full"),
]

_BLOCK_2D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(input=(slice(5, 15), [1, 2, 3]), exception=IndexError, id="slice-with-array"),
    ExpectFail(input=(Ellipsis, [1, 2, 3]), exception=IndexError, id="ellipsis-with-array"),
    ExpectFail(input=(slice(15, 20), slice(None)), exception=IndexError, id="out-of-bounds"),
]


@pytest.mark.parametrize("case", _BLOCK_1D_CASES, ids=lambda c: c.id)
def test_get_block_selection_1d(store: StorePath, case: Expect[BasicSelection, slice]) -> None:
    """get_block_selection / .blocks on a 1D array selects whole chunks matching the array slice."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_get_block_selection(a, z, case.input, case.output)


@pytest.mark.parametrize("case", _BLOCK_1D_BAD_CASES, ids=lambda c: c.id)
def test_get_block_selection_1d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """get_block_selection / .blocks on a 1D array rejects invalid block selections with IndexError."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.get_block_selection(case.input)
    with case.raises():
        z.blocks[case.input]


@pytest.mark.parametrize("case", _BLOCK_2D_CASES, ids=lambda c: c.id)
def test_get_block_selection_2d(
    store: StorePath, case: Expect[BasicSelection, tuple[slice, slice]]
) -> None:
    """get_block_selection / .blocks on a 2D array selects whole chunk regions matching the array slices."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_get_block_selection(a, z, case.input, case.output)


@pytest.mark.parametrize("case", _BLOCK_2D_BAD_CASES, ids=lambda c: c.id)
def test_get_block_selection_2d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """get_block_selection on a 2D array rejects invalid or out-of-bounds block selections with IndexError."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    with case.raises():
        z.get_block_selection(case.input)


def _test_set_block_selection(
    v: npt.NDArray[Any],
    a: npt.NDArray[Any],
    z: zarr.Array,
    selection: BasicSelection,
    expected_idx: slice | tuple[slice, ...],
) -> None:
    for value in 42, v[expected_idx], v[expected_idx].tolist():
        # setup expectation
        a[:] = 0
        a[expected_idx] = value
        # test long-form API
        z[:] = 0
        z.set_block_selection(selection, value)
        assert_array_equal(a, z[:])
        # test short-form API
        z[:] = 0
        z.blocks[selection] = value
        assert_array_equal(a, z[:])


@pytest.mark.parametrize("case", _BLOCK_1D_CASES, ids=lambda c: c.id)
def test_set_block_selection_1d(store: StorePath, case: Expect[BasicSelection, slice]) -> None:
    """set_block_selection / .blocks assignment on a 1D array round-trips through numpy for each block selection."""
    v = np.arange(30, dtype=int)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_set_block_selection(v, a, z, case.input, case.output)


@pytest.mark.parametrize("case", _BLOCK_1D_BAD_CASES, ids=lambda c: c.id)
def test_set_block_selection_1d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """set_block_selection / .blocks assignment on a 1D array rejects invalid block selections with IndexError."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.set_block_selection(case.input, 42)
    with case.raises():
        z.blocks[case.input] = 42


@pytest.mark.parametrize("case", _BLOCK_2D_CASES, ids=lambda c: c.id)
def test_set_block_selection_2d(
    store: StorePath, case: Expect[BasicSelection, tuple[slice, slice]]
) -> None:
    """set_block_selection / .blocks assignment on a 2D array round-trips through numpy for each block selection."""
    v = np.arange(60, dtype=int).reshape(12, 5)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_set_block_selection(v, a, z, case.input, case.output)


@pytest.mark.parametrize("case", _BLOCK_2D_BAD_CASES, ids=lambda c: c.id)
def test_set_block_selection_2d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """set_block_selection on a 2D array rejects invalid or out-of-bounds block selections with IndexError."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    with case.raises():
        z.set_block_selection(case.input, 42)


def _test_get_mask_selection(a: npt.NDArray[Any], z: Array, selection: npt.NDArray) -> None:
    expect = a[selection]
    actual = z.get_mask_selection(selection)
    assert_array_equal(expect, actual)
    actual = z.vindex[selection]
    assert_array_equal(expect, actual)
    actual = z[selection]
    assert_array_equal(expect, actual)


_MASK_1D_CASES: list[Expect[Any, None]] = [
    Expect(input=np.zeros(30, dtype=bool), output=None, id="all-false"),
    Expect(input=np.ones(30, dtype=bool), output=None, id="all-true"),
    Expect(input=np.arange(30) % 2 == 0, output=None, id="alternating"),
    Expect(
        input=np.isin(np.arange(30), [0, 7, 14, 29]),
        output=None,
        id="sparse-cross-chunk",
    ),
]

# msg=None for all 1d bad cases: get_mask_selection and vindex raise different
# messages for the same input, so no single substring satisfies both assertions.
_MASK_1D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(input=slice(5, 15), exception=IndexError, id="slice"),
    ExpectFail(input=slice(None), exception=IndexError, id="full-slice"),
    ExpectFail(input=Ellipsis, exception=IndexError, id="ellipsis"),
    ExpectFail(input=2.3, exception=IndexError, id="float"),
    ExpectFail(input="foo", exception=IndexError, id="string"),
    ExpectFail(input=b"xxx", exception=IndexError, id="bytes"),
    ExpectFail(input=None, exception=IndexError, id="none"),
    ExpectFail(input=(0, 0), exception=IndexError, id="tuple-pair"),
    ExpectFail(input=(slice(None), slice(None)), exception=IndexError, id="two-slices"),
    ExpectFail(input=np.zeros(5, dtype=bool), exception=IndexError, id="mask-too-short"),
    ExpectFail(input=np.zeros(50, dtype=bool), exception=IndexError, id="mask-too-long"),
    ExpectFail(input=[[True, False], [False, True]], exception=IndexError, id="too-many-dims"),
]


def _make_sparse_2d_mask() -> npt.NDArray[np.bool_]:
    """Build a deterministic sparse (12, 5) boolean mask with Trues at (0,0), (5,2), (11,4), (2,3)."""
    mask = np.zeros((12, 5), dtype=bool)
    for r, c in [(0, 0), (5, 2), (11, 4), (2, 3)]:
        mask[r, c] = True
    return mask


_MASK_2D_CASES: list[Expect[Any, None]] = [
    Expect(input=np.zeros((12, 5), dtype=bool), output=None, id="all-false"),
    Expect(input=np.ones((12, 5), dtype=bool), output=None, id="all-true"),
    Expect(
        input=(np.add.outer(np.arange(12), np.arange(5)) % 2).astype(bool),
        output=None,
        id="checkerboard",
    ),
    Expect(
        input=_make_sparse_2d_mask(),
        output=None,
        id="sparse",
    ),
]

_MASK_2D_BAD_CASES: list[ExpectFail[Any]] = [
    ExpectFail(input=np.zeros((12, 3), dtype=bool), exception=IndexError, id="too-few-cols"),
    ExpectFail(input=np.zeros((20, 5), dtype=bool), exception=IndexError, id="too-many-rows"),
    ExpectFail(input=[True, False], exception=IndexError, id="wrong-ndim"),
]


@pytest.mark.parametrize("case", _MASK_1D_CASES, ids=lambda c: c.id)
def test_get_mask_selection_1d(store: StorePath, case: Expect[Any, None]) -> None:
    """get_mask_selection / vindex / getitem on a 1D array match numpy for boolean masks."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_get_mask_selection(a, z, case.input)


@pytest.mark.parametrize("case", _MASK_1D_BAD_CASES, ids=lambda c: c.id)
def test_get_mask_selection_1d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """get_mask_selection / vindex on a 1D array reject non-boolean-mask and mis-shaped selections."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.get_mask_selection(case.input)  # type: ignore[arg-type]
    with case.raises():
        z.vindex[case.input]  # type: ignore[index]


@pytest.mark.parametrize("case", _MASK_2D_CASES, ids=lambda c: c.id)
def test_get_mask_selection_2d(store: StorePath, case: Expect[Any, None]) -> None:
    """get_mask_selection / vindex / getitem on a 2D array match numpy for boolean masks."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_get_mask_selection(a, z, case.input)


@pytest.mark.parametrize("case", _MASK_2D_BAD_CASES, ids=lambda c: c.id)
def test_get_mask_selection_2d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """vindex on a 2D array rejects masks of the wrong shape or dimensionality."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    with case.raises():
        z.vindex[case.input]  # type: ignore[index]


def _test_set_mask_selection(
    v: npt.NDArray, a: npt.NDArray, z: Array, selection: npt.NDArray
) -> None:
    a[:] = 0
    z[:] = 0
    a[selection] = v[selection]
    z.set_mask_selection(selection, v[selection])
    assert_array_equal(a, z[:])
    z[:] = 0
    z.vindex[selection] = v[selection]
    assert_array_equal(a, z[:])
    z[:] = 0
    z[selection] = v[selection]
    assert_array_equal(a, z[:])


@pytest.mark.parametrize("case", _MASK_1D_CASES, ids=lambda c: c.id)
def test_set_mask_selection_1d(store: StorePath, case: Expect[Any, None]) -> None:
    """set_mask_selection / vindex / setitem on a 1D array match numpy for boolean masks."""
    v = np.arange(30, dtype=int)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    _test_set_mask_selection(v, a, z, case.input)


@pytest.mark.parametrize("case", _MASK_1D_BAD_CASES, ids=lambda c: c.id)
def test_set_mask_selection_1d_raises(store: StorePath, case: ExpectFail[Any]) -> None:
    """set_mask_selection / vindex on a 1D array reject non-boolean-mask and mis-shaped selections."""
    a = np.arange(30, dtype=int)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))
    with case.raises():
        z.set_mask_selection(case.input, 42)  # type: ignore[arg-type]
    with case.raises():
        z.vindex[case.input] = 42  # type: ignore[index]


@pytest.mark.parametrize("case", _MASK_2D_CASES, ids=lambda c: c.id)
def test_set_mask_selection_2d(store: StorePath, case: Expect[Any, None]) -> None:
    """set_mask_selection / vindex / setitem on a 2D array match numpy for boolean masks."""
    v = np.arange(60, dtype=int).reshape(12, 5)
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    _test_set_mask_selection(v, a, z, case.input)


def test_get_selection_out(store: StorePath) -> None:
    """get_*_selection writes results into a provided out buffer, matching numpy."""
    # basic selections
    a = np.arange(30)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(7,))

    selections = [
        slice(5, 15),
        slice(0, 30),
        slice(1, 2),
    ]
    for selection in selections:
        expect = a[selection]
        out = get_ndbuffer_class().from_numpy_array(np.empty(expect.shape))
        z.get_basic_selection(selection, out=out)
        assert_array_equal(expect, out.as_numpy_array()[:])

    with pytest.raises(TypeError):
        z.get_basic_selection(Ellipsis, out=[])  # type: ignore[arg-type]

    # orthogonal selections
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    selections = [
        # index both axes with bool array
        (_ORTHO_2D_IX0_BOOL, _ORTHO_2D_IX1_BOOL),
        # mixed indexing with bool array / slice
        (_ORTHO_2D_IX0_BOOL, slice(1, 4)),
        (slice(2, 9), _ORTHO_2D_IX1_BOOL),
        # mixed indexing with bool array / int
        (_ORTHO_2D_IX0_BOOL, 3),
        (7, _ORTHO_2D_IX1_BOOL),
        # mixed int array / bool array
        (_ORTHO_2D_IX0_BOOL, _ORTHO_2D_IX1_INT),
        (_ORTHO_2D_IX0_INT, _ORTHO_2D_IX1_BOOL),
    ]
    for selection in selections:
        expect = oindex(a, selection)
        out = get_ndbuffer_class().from_numpy_array(np.zeros(expect.shape, dtype=expect.dtype))
        z.get_orthogonal_selection(selection, out=out)
        assert_array_equal(expect, out.as_numpy_array()[:])

    # coordinate selections
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    selections = [
        # index both axes with array
        (np.array([0, 5, 11]), np.array([0, 2, 4])),
        # mixed indexing with array / int
        (np.array([0, 5, 11]), 3),
        (7, np.array([0, 2, 4])),
    ]
    for selection in selections:
        expect = a[selection]
        out = get_ndbuffer_class().from_numpy_array(np.zeros(expect.shape, dtype=expect.dtype))
        z.get_coordinate_selection(selection, out=out)
        assert_array_equal(expect, out.as_numpy_array()[:])


@pytest.mark.xfail(reason="fields are not supported in v3")
def test_get_selections_with_fields(store: StorePath) -> None:
    """Would verify that basic, orthogonal, coordinate, and mask selections with structured-array `fields` arguments return the correct sub-fields (xfail: fields unsupported in v3)."""
    a = np.array(
        [("aaa", 1, 4.2), ("bbb", 2, 8.4), ("ccc", 3, 12.6)],
        dtype=[("foo", "S3"), ("bar", "i4"), ("baz", "f8")],
    )
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(2,))

    fields_fixture: list[str | list[str]] = [
        "foo",
        ["foo"],
        ["foo", "bar"],
        ["foo", "baz"],
        ["bar", "baz"],
        ["foo", "bar", "baz"],
        ["bar", "foo"],
        ["baz", "bar", "foo"],
    ]

    for fields in fields_fixture:
        # total selection
        expect = a[fields]
        actual = z.get_basic_selection(Ellipsis, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z[fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[fields[0], fields[1]]
            assert_array_equal(expect, actual)
        if isinstance(fields, str):
            actual = z[..., fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[..., fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # basic selection with slice
        expect = a[fields][0:2]
        actual = z.get_basic_selection(slice(0, 2), fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z[0:2, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[0:2, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # basic selection with single item
        expect = a[fields][1]
        actual = z.get_basic_selection(1, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z[1, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z[1, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # orthogonal selection
        ix = [0, 2]
        expect = a[fields][ix]
        actual = z.get_orthogonal_selection(ix, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z.oindex[ix, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z.oindex[ix, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # coordinate selection
        ix = [0, 2]
        expect = a[fields][ix]
        actual = z.get_coordinate_selection(ix, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z.vindex[ix, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z.vindex[ix, fields[0], fields[1]]
            assert_array_equal(expect, actual)

        # mask selection
        ix = [True, False, True]
        expect = a[fields][ix]
        actual = z.get_mask_selection(ix, fields=fields)
        assert_array_equal(expect, actual)
        # alternative API
        if isinstance(fields, str):
            actual = z.vindex[ix, fields]
            assert_array_equal(expect, actual)
        elif len(fields) == 2:
            actual = z.vindex[ix, fields[0], fields[1]]
            assert_array_equal(expect, actual)

    # missing/bad fields
    with pytest.raises(IndexError):
        z.get_basic_selection(Ellipsis, fields=["notafield"])
    with pytest.raises(IndexError):
        z.get_basic_selection(Ellipsis, fields=slice(None))  # type: ignore[arg-type]


@pytest.mark.xfail(reason="fields are not supported in v3")
def test_set_selections_with_fields(store: StorePath) -> None:
    """Would verify that basic, orthogonal, coordinate, and mask set-selections with structured-array `fields` correctly write individual fields and reject multi-field assignment (xfail: fields unsupported in v3)."""
    v = np.array(
        [("aaa", 1, 4.2), ("bbb", 2, 8.4), ("ccc", 3, 12.6)],
        dtype=[("foo", "S3"), ("bar", "i4"), ("baz", "f8")],
    )
    a = np.empty_like(v)
    z = zarr_array_from_numpy_array(store, v, chunk_shape=(2,))

    fields_fixture: list[str | list[str]] = [
        "foo",
        [],
        ["foo"],
        ["foo", "bar"],
        ["foo", "baz"],
        ["bar", "baz"],
        ["foo", "bar", "baz"],
        ["bar", "foo"],
        ["baz", "bar", "foo"],
    ]

    for fields in fields_fixture:
        # currently multi-field assignment is not supported in numpy, so we won't support
        # it either
        if isinstance(fields, list) and len(fields) > 1:
            with pytest.raises(IndexError):
                z.set_basic_selection(Ellipsis, v, fields=fields)
            with pytest.raises(IndexError):
                z.set_orthogonal_selection([0, 2], v, fields=fields)  # type: ignore[arg-type]
            with pytest.raises(IndexError):
                z.set_coordinate_selection([0, 2], v, fields=fields)
            with pytest.raises(IndexError):
                z.set_mask_selection([True, False, True], v, fields=fields)  # type: ignore[arg-type]

        else:
            if isinstance(fields, list) and len(fields) == 1:
                # work around numpy does not support multi-field assignment even if there
                # is only one field
                key = fields[0]
            elif isinstance(fields, list) and len(fields) == 0:
                # work around numpy ambiguity about what is a field selection
                key = Ellipsis
            else:
                key = fields

            # setup expectation
            a[:] = ("", 0, 0)
            z[:] = ("", 0, 0)
            assert_array_equal(a, z[:])
            a[key] = v[key]
            # total selection
            z.set_basic_selection(Ellipsis, v[key], fields=fields)
            assert_array_equal(a, z[:])

            # basic selection with slice
            a[:] = ("", 0, 0)
            z[:] = ("", 0, 0)
            a[key][0:2] = v[key][0:2]
            z.set_basic_selection(slice(0, 2), v[key][0:2], fields=fields)
            assert_array_equal(a, z[:])

            # orthogonal selection
            a[:] = ("", 0, 0)
            z[:] = ("", 0, 0)
            ix = [0, 2]
            a[key][ix] = v[key][ix]
            z.set_orthogonal_selection(ix, v[key][ix], fields=fields)
            assert_array_equal(a, z[:])

            # coordinate selection
            a[:] = ("", 0, 0)
            z[:] = ("", 0, 0)
            ix = [0, 2]
            a[key][ix] = v[key][ix]
            z.set_coordinate_selection(ix, v[key][ix], fields=fields)
            assert_array_equal(a, z[:])

            # mask selection
            a[:] = ("", 0, 0)
            z[:] = ("", 0, 0)
            ix = [True, False, True]
            a[key][ix] = v[key][ix]
            z.set_mask_selection(ix, v[key][ix], fields=fields)
            assert_array_equal(a, z[:])


def test_slice_selection_uints() -> None:
    """make_slice_selection accepts unsigned integer indices without error and produces correct shape."""
    arr = np.arange(24).reshape((4, 6))
    idx = np.uint64(3)
    slice_sel = make_slice_selection((idx,))
    assert arr[tuple(slice_sel)].shape == (1, 6)


def test_numpy_int_indexing(store: StorePath) -> None:
    """Indexing with a plain Python int and with `np.int64` both return the correct scalar element."""
    a = np.arange(1050)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(100,))
    assert a[42] == z[42]
    assert a[np.int64(42)] == z[np.int64(42)]


@pytest.mark.parametrize(
    ("shape", "chunks", "ops"),
    [
        # 1D test cases
        ((1070,), (50,), [("__getitem__", (slice(200, 400),))]),
        ((1070,), (50,), [("__getitem__", (slice(200, 400, 100),))]),
        (
            (1070,),
            (50,),
            [
                ("__getitem__", (slice(200, 400),)),
                ("__setitem__", (slice(200, 400, 100),)),
            ],
        ),
        # 2D test cases
        (
            (40, 50),
            (5, 8),
            [
                ("__getitem__", (slice(6, 37, 13), (slice(4, 10)))),
                ("__setitem__", (slice(None), (slice(None)))),
            ],
        ),
    ],
)
async def test_accessed_chunks(
    shape: tuple[int, ...], chunks: tuple[int, ...], ops: list[tuple[str, tuple[slice, ...]]]
) -> None:
    """Only the chunks intersected by a slice selection are read or written, verified via a `CountingDict` store."""
    # Test that only the required chunks are accessed during basic selection operations
    # shape: array shape
    # chunks: chunk size
    # ops: list of tuples with (optype, tuple of slices)
    # optype = "__getitem__" or "__setitem__", tuple length must match number of dims

    # Use a counting dict as the backing store so we can track the items access
    store = await CountingDict.open()
    z = zarr_array_from_numpy_array(StorePath(store), np.zeros(shape), chunk_shape=chunks)

    for ii, (optype, slices) in enumerate(ops):
        # Resolve the slices into the accessed chunks for each dimension
        chunks_per_dim = []
        for N, C, sl in zip(shape, chunks, slices, strict=True):
            chunk_ind = np.arange(N, dtype=int)[sl] // C
            chunks_per_dim.append(np.unique(chunk_ind))

        # Combine and generate the cartesian product to determine the chunks keys that
        # will be accessed
        chunks_accessed = [".".join(map(str, comb)) for comb in itertools.product(*chunks_per_dim)]

        counts_before = store.counter.copy()

        # Perform the operation
        if optype == "__getitem__":
            z[slices]
        else:
            z[slices] = ii

        # Get the change in counts
        delta_counts = store.counter - counts_before

        # Check that the access counts for the operation have increased by one for all
        # the chunks we expect to be included
        for ci in chunks_accessed:
            assert delta_counts.pop((optype, ci)) == 1

            # If the chunk was partially written to it will also have been read once. We
            # don't determine if the chunk was actually partial here, just that the
            # counts are consistent that this might have happened
            if optype == "__setitem__":
                assert ("__getitem__", ci) not in delta_counts or delta_counts.pop(
                    ("__getitem__", ci)
                ) == 1
        # Check that no other chunks were accessed
        assert len(delta_counts) == 0


@pytest.mark.parametrize(
    "selection",
    [
        # basic selection
        [...],
        [1, ...],
        [slice(None)],
        [1, 3],
        [[1, 2, 3], 4],
        [np.arange(12)],
        [slice(2, 9)],
        [slice(1, 3), 3],
        [[1, 3]],
        # mask selection
        [np.tile([True, False, True, False, True], (12, 1))],
        [np.full((12, 5), False)],
        # coordinate selection
        [[1, 2, 3, 4], [0, 1, 2, 3]],
        [[10, 11, 5], [4, 0, 2]],
    ],
)
def test_indexing_equals_numpy(store: StorePath, selection: Selection) -> None:
    """Indexing a zarr array with assorted basic/mask/coordinate selections matches numpy."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    # note: in python 3.10 a[*selection] is not valid unpacking syntax
    expected = a[*selection,]
    actual = z[*selection,]
    assert_array_equal(expected, actual, err_msg=f"selection: {selection}")


@pytest.mark.parametrize(
    "selection",
    [
        [np.tile([True, False], 6), np.tile([True, False, True, False, True], 1)],
        [np.full(12, False), np.array([True, False, True, False, True])],
        [np.full(12, True), np.full(5, True)],
        [np.full(12, True), [True, False, True, False, True]],
    ],
)
def test_orthogonal_bool_indexing_like_numpy_ix(
    store: StorePath, selection: list[npt.ArrayLike]
) -> None:
    """Orthogonal boolean indexing on each axis matches numpy's np.ix_ semantics."""
    a = np.arange(60, dtype=int).reshape(12, 5)
    z = zarr_array_from_numpy_array(store, a, chunk_shape=(5, 2))
    expected = a[np.ix_(*selection)]
    # note: in python 3.10 z[*selection] is not valid unpacking syntax
    actual = z[*selection,]
    assert_array_equal(expected, actual, err_msg=f"{selection=}")


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("origin_0d", [None, (0,), (1,)])
@pytest.mark.parametrize("selection_shape_0d", [None, (2,), (3,)])
def test_iter_grid(
    ndim: int, origin_0d: tuple[int] | None, selection_shape_0d: tuple[int] | None
) -> None:
    """
    Test that iter_grid works as expected for 1, 2, and 3 dimensions.
    """
    grid_shape = (10, 5, 7)[:ndim]

    if origin_0d is not None:
        origin_kwarg = origin_0d * ndim
        origin = origin_kwarg
    else:
        origin_kwarg = None
        origin = (0,) * ndim

    if selection_shape_0d is not None:
        selection_shape_kwarg = selection_shape_0d * ndim
        selection_shape = selection_shape_kwarg
    else:
        selection_shape_kwarg = None
        selection_shape = tuple(gs - o for gs, o in zip(grid_shape, origin, strict=False))

    observed = tuple(
        _iter_grid(grid_shape, origin=origin_kwarg, selection_shape=selection_shape_kwarg)
    )

    # generate a numpy array of indices, and index it
    coord_array = np.array(list(itertools.product(*[range(s) for s in grid_shape]))).reshape(
        (*grid_shape, ndim)
    )
    coord_array_indexed = coord_array[
        tuple(slice(o, o + s, 1) for o, s in zip(origin, selection_shape, strict=False))
        + (range(ndim),)
    ]

    expected = tuple(map(tuple, coord_array_indexed.reshape(-1, ndim).tolist()))
    assert observed == expected


def test_iter_grid_invalid() -> None:
    """
    Ensure that a selection_shape that exceeds the grid_shape + origin produces an indexing error.
    """
    with pytest.raises(IndexError):
        list(_iter_grid((5,), origin=(0,), selection_shape=(10,)))


def test_indexing_with_zarr_array(store: StorePath) -> None:
    """Regression for GH2133: indexing a zarr array with another zarr array (boolean or integer) as the indexer produces the same result as indexing with the equivalent numpy array."""
    # regression test for https://github.com/zarr-developers/zarr-python/issues/2133
    a = np.arange(10)
    za = zarr.array(a, chunks=2, store=store, path="a")
    ix = [False, True, False, True, False, True, False, True, False, True]
    ii = [0, 2, 4, 5]

    zix = zarr.array(ix, chunks=2, store=store, dtype="bool", path="ix")
    zii = zarr.array(ii, chunks=2, store=store, dtype="i4", path="ii")
    assert_array_equal(a[ix], za[zix])
    assert_array_equal(a[ix], za.oindex[zix])
    assert_array_equal(a[ix], za.vindex[zix])

    assert_array_equal(a[ii], za[zii])
    assert_array_equal(a[ii], za.oindex[zii])


@pytest.mark.parametrize("store", ["local", "memory"], indirect=["store"])
@pytest.mark.parametrize("shape", [(0, 2, 3), (0,), (3, 0)])
def test_zero_sized_chunks(store: StorePath, shape: list[int]) -> None:
    """Arrays with zero-extent dimensions can be created and indexed without error; reading back returns the fill value."""
    # Chunk sizes must be >= 1 per spec; use 1 for zero-extent dimensions.
    chunks = tuple(max(1, s) for s in shape)
    z = zarr.create_array(store=store, shape=shape, chunks=chunks, zarr_format=3, dtype="f8")
    z[...] = 42
    assert_array_equal(z[...], np.zeros(shape, dtype="f8"))


@pytest.mark.parametrize("store", ["memory"], indirect=["store"])
def test_vectorized_indexing_incompatible_shape(store) -> None:
    """Regression for GH2469: vectorized set-indexing raises ValueError when the value shape is incompatible with the indexer shape."""
    # GH2469
    shape = (4, 4)
    chunks = (2, 2)
    fill_value = 32767
    arr = zarr.create(
        shape,
        store=store,
        chunks=chunks,
        dtype=np.int16,
        fill_value=fill_value,
        codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()],
    )
    with pytest.raises(ValueError, match="Attempting to set"):
        arr[np.array([1, 2]), np.array([1, 2])] = np.array([[-1, -2], [-3, -4]])


def test_iter_chunk_regions():
    """_iter_chunk_regions yields slices that exactly cover each chunk, and reading/writing each region round-trips correctly."""
    chunks = (2, 3)
    a = zarr.create((10, 10), chunks=chunks)
    a[:] = 1
    for region in a._iter_chunk_regions():
        assert_array_equal(a[region], np.ones_like(a[region]))
        a[region] = 0
        assert_array_equal(a[region], np.zeros_like(a[region]))


@pytest.mark.parametrize(
    ("domain_shape", "region_shape", "origin", "selection_shape"),
    [
        ((9,), (1,), None, (9,)),
        ((9,), (1,), (0,), (9,)),
        ((3,), (2,), (0,), (1,)),
        ((9,), (2,), (2,), (2,)),
        ((9, 9), (2, 1), None, None),
        ((9, 9), (4, 1), None, None),
    ],
)
@pytest.mark.parametrize("order", ["lexicographic"])
@pytest.mark.parametrize("trim_excess", [True, False])
def test_iter_regions(
    domain_shape: tuple[int, ...],
    region_shape: tuple[int, ...],
    origin: tuple[int, ...] | None,
    selection_shape: tuple[int, ...] | None,
    order: _ArrayIndexingOrder,
    trim_excess: bool,
) -> None:
    """
    Test that iter_regions properly iterates over contiguous regions of a gridded domain.
    """
    expected_slices_by_dim: list[list[slice]] = []
    origin_parsed: tuple[int, ...]
    selection_shape_parsed: tuple[int, ...]
    if origin is None:
        origin_parsed = (0,) * len(domain_shape)
    else:
        origin_parsed = origin
    if selection_shape is None:
        selection_shape_parsed = tuple(
            ceildiv(ds, rs) - o
            for ds, o, rs in zip(domain_shape, origin_parsed, region_shape, strict=True)
        )
    else:
        selection_shape_parsed = selection_shape
    for d_s, r_s, o, ss in zip(
        domain_shape, region_shape, origin_parsed, selection_shape_parsed, strict=True
    ):
        _expected_slices: list[slice] = []
        start = o * r_s
        for incr in range(start, start + ss * r_s, r_s):
            if trim_excess:
                term = min(incr + r_s, d_s)
            else:
                term = incr + r_s
            _expected_slices.append(slice(incr, term, 1))
        expected_slices_by_dim.append(_expected_slices)

    expected = tuple(itertools.product(*expected_slices_by_dim))
    observed = tuple(
        _iter_regions(
            domain_shape,
            region_shape,
            origin=origin,
            selection_shape=selection_shape,
            order=order,
            trim_excess=trim_excess,
        )
    )
    assert observed == expected


class TestAsync:
    @pytest.mark.parametrize(
        ("indexer", "expected"),
        [
            # int
            ((0,), np.array([1, 2])),
            ((1,), np.array([3, 4])),
            ((0, 1), np.array(2)),
            # slice
            ((slice(None),), np.array([[1, 2], [3, 4]])),
            ((slice(0, 1),), np.array([[1, 2]])),
            ((slice(1, 2),), np.array([[3, 4]])),
            ((slice(0, 2),), np.array([[1, 2], [3, 4]])),
            ((slice(0, 0),), np.empty(shape=(0, 2), dtype="i8")),
            # ellipsis
            ((...,), np.array([[1, 2], [3, 4]])),
            ((0, ...), np.array([1, 2])),
            ((..., 0), np.array([1, 3])),
            ((0, 1, ...), np.array(2)),
            # combined
            ((0, slice(None)), np.array([1, 2])),
            ((slice(None), 0), np.array([1, 3])),
            ((slice(None), slice(None)), np.array([[1, 2], [3, 4]])),
            # array of ints
            (([0]), np.array([[1, 2]])),
            (([1]), np.array([[3, 4]])),
            (([0], [1]), np.array(2)),
            (([0, 1], [0]), np.array([[1], [3]])),
            (([0, 1], [0, 1]), np.array([[1, 2], [3, 4]])),
            # boolean array
            (np.array([True, True]), np.array([[1, 2], [3, 4]])),
            (np.array([True, False]), np.array([[1, 2]])),
            (np.array([False, True]), np.array([[3, 4]])),
            (np.array([False, False]), np.empty(shape=(0, 2), dtype="i8")),
        ],
    )
    async def test_async_oindex(self, store, indexer, expected):
        """The async `oindex.getitem` interface returns the correct orthogonally-indexed result for int, slice, ellipsis, array, and boolean indexers."""
        z = zarr.create_array(store=store, shape=(2, 2), chunks=(1, 1), zarr_format=3, dtype="i8")
        z[...] = np.array([[1, 2], [3, 4]])
        async_zarr = z._async_array

        result = await async_zarr.oindex.getitem(indexer)
        assert_array_equal(result, expected)

    async def test_async_oindex_with_zarr_array(self, store):
        """The async `oindex.getitem` interface accepts a zarr boolean array as the indexer and returns the correct rows."""
        group = zarr.create_group(store=store, zarr_format=3)

        z1 = group.create_array(name="z1", shape=(2, 2), chunks=(1, 1), dtype="i8")
        z1[...] = np.array([[1, 2], [3, 4]])
        async_zarr = z1._async_array

        # create boolean zarr array to index with
        z2 = group.create_array(name="z2", shape=(2,), chunks=(1,), dtype="?")
        z2[...] = np.array([True, False])

        result = await async_zarr.oindex.getitem(z2)
        expected = np.array([[1, 2]])
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        ("indexer", "expected"),
        [
            (([0], [0]), np.array(1)),
            (([0, 1], [0, 1]), np.array([1, 4])),
            (np.array([[False, True], [False, True]]), np.array([2, 4])),
        ],
    )
    async def test_async_vindex(self, store, indexer, expected):
        """The async `vindex.getitem` interface returns the correct vectorized-indexed result for coordinate and boolean indexers."""
        z = zarr.create_array(store=store, shape=(2, 2), chunks=(1, 1), zarr_format=3, dtype="i8")
        z[...] = np.array([[1, 2], [3, 4]])
        async_zarr = z._async_array

        result = await async_zarr.vindex.getitem(indexer)
        assert_array_equal(result, expected)

    async def test_async_vindex_with_zarr_array(self, store):
        """The async `vindex.getitem` interface accepts a zarr 2D boolean array as the indexer and returns the correct elements."""
        group = zarr.create_group(store=store, zarr_format=3)

        z1 = group.create_array(name="z1", shape=(2, 2), chunks=(1, 1), dtype="i8")
        z1[...] = np.array([[1, 2], [3, 4]])
        async_zarr = z1._async_array

        # create boolean zarr array to index with
        z2 = group.create_array(name="z2", shape=(2, 2), chunks=(1, 1), dtype="?")
        z2[...] = np.array([[False, True], [False, True]])

        result = await async_zarr.vindex.getitem(z2)
        expected = np.array([2, 4])
        assert_array_equal(result, expected)

    async def test_async_invalid_indexer(self, store):
        """The async `vindex.getitem` and `oindex.getitem` interfaces raise IndexError when given an unsupported indexer type."""
        z = zarr.create_array(store=store, shape=(2, 2), chunks=(1, 1), zarr_format=3, dtype="i8")
        z[...] = np.array([[1, 2], [3, 4]])
        async_zarr = z._async_array

        with pytest.raises(IndexError):
            await async_zarr.vindex.getitem("invalid_indexer")

        with pytest.raises(IndexError):
            await async_zarr.oindex.getitem("invalid_indexer")
