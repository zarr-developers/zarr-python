from __future__ import annotations

import math
from typing import Any, get_type_hints

import numpy as np
import numpy.typing as npt
import pytest

import zarr_indexing.reader as reader_module
from zarr_indexing import (
    ArrayMap,
    ChunkProjection,
    DimensionMap,
    IndexDomain,
    IndexTransform,
    ReadContext,
)
from zarr_indexing.reader import basic_reader, numpy_reader


class BasicOnlySource:
    def __init__(self, data: np.ndarray[Any, Any]) -> None:
        self.data = data
        self.keys: list[tuple[Any, ...]] = []

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    def __getitem__(self, key: tuple[Any, ...]) -> np.ndarray[Any, Any]:
        assert all(isinstance(item, slice) and (item.step or 1) > 0 for item in key)
        self.keys.append(key)
        return self.data[key]


SOURCE = np.arange(6 * 7 * 8).reshape(6, 7, 8)
BASE = IndexTransform.from_shape(SOURCE.shape)


def test_read_context_public_annotations_resolve() -> None:
    assert get_type_hints(ReadContext)["projection"] == ChunkProjection | None


READER_CASES = (
    pytest.param(
        SOURCE,
        BASE[1:6:2, 2, ::-2],
        np.array(
            [
                [79, 77, 75, 73],
                [191, 189, 187, 185],
                [303, 301, 299, 297],
            ]
        ),
        id="strided-reversed-nonzero-origin",
    ),
    pytest.param(
        SOURCE,
        BASE.oindex[[5, 1, 1], slice(1, 6), [7, 2]],
        np.array(
            [
                [[295, 290], [303, 298], [311, 306], [319, 314], [327, 322]],
                [[71, 66], [79, 74], [87, 82], [95, 90], [103, 98]],
                [[71, 66], [79, 74], [87, 82], [95, 90], [103, 98]],
            ]
        ),
        id="orthogonal-nonzero-origin",
    ),
    pytest.param(
        SOURCE,
        BASE.vindex[np.array([[5], [1]]), np.array([[2, 4, 0]])],
        np.array(
            [
                [
                    [296, 297, 298, 299, 300, 301, 302, 303],
                    [312, 313, 314, 315, 316, 317, 318, 319],
                    [280, 281, 282, 283, 284, 285, 286, 287],
                ],
                [
                    [72, 73, 74, 75, 76, 77, 78, 79],
                    [88, 89, 90, 91, 92, 93, 94, 95],
                    [56, 57, 58, 59, 60, 61, 62, 63],
                ],
            ]
        ),
        id="correlated-broadcast",
    ),
    pytest.param(
        SOURCE,
        BASE.vindex[[5, 1], [2, 2]],
        np.array(
            [
                [296, 297, 298, 299, 300, 301, 302, 303],
                [72, 73, 74, 75, 76, 77, 78, 79],
            ]
        ),
        id="correlated-vector",
    ),
    pytest.param(
        SOURCE,
        BASE[:, 0:0, :],
        np.empty((6, 0, 8), dtype=SOURCE.dtype),
        id="empty",
    ),
    pytest.param(SOURCE, BASE[2, 3, 4], np.array(140), id="scalar"),
    pytest.param(
        np.arange(8),
        IndexTransform(
            domain=IndexDomain((4,), (7,)),
            output=(DimensionMap(input_dimension=0, offset=2, stride=0),),
        ),
        np.array([2, 2, 2]),
        id="zero-stride-nonzero-origin",
    ),
)


@pytest.mark.parametrize(("source_data", "transform", "expected"), READER_CASES)
@pytest.mark.parametrize("reader_name", ["basic", "numpy"])
def test_builtin_readers_match_the_transform(
    source_data: np.ndarray[Any, Any],
    transform: IndexTransform,
    expected: np.ndarray[Any, Any],
    reader_name: str,
) -> None:
    out = np.empty(transform.domain.shape, dtype=source_data.dtype)
    if reader_name == "basic":
        source = BasicOnlySource(source_data)
        result = basic_reader.read_into(source, ReadContext(transform), out)
        assert len(source.keys) == 1
    else:
        result = numpy_reader.read_into(source_data, ReadContext(transform), out)
    assert result is None
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("reader_name", ["basic", "numpy"])
def test_builtin_readers_share_transform_affine_overflow(reader_name: str) -> None:
    transform = IndexTransform(
        domain=IndexDomain.from_shape((1,)),
        output=(ArrayMap(np.array([2**62], dtype=np.intp), stride=4),),
    )

    with pytest.raises(OverflowError, match="outside np.intp"):
        transform.apply_many(np.array([[0]], dtype=np.intp))

    out = np.empty(transform.domain.shape, dtype=np.intp)
    if reader_name == "basic":
        with pytest.raises(OverflowError, match="outside np.intp"):
            basic_reader.read_into(BasicOnlySource(np.arange(1)), ReadContext(transform), out)
    else:
        with pytest.raises(OverflowError, match="outside np.intp"):
            numpy_reader.read_into(np.arange(1), ReadContext(transform), out)


def test_numpy_reader_narrows_basic_slab_before_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(1_000_000).reshape(1000, 1000)
    transform = IndexTransform.from_shape(source.shape).oindex[:10, [0, 999]]
    seen: list[tuple[int, ...]] = []
    real_take = reader_module._take  # pyright: ignore[reportPrivateUsage]

    def recording_take(array: Any, indices: npt.NDArray[np.intp], axis: int) -> Any:
        seen.append(tuple(int(value) for value in array.shape))
        return real_take(array, indices, axis)

    monkeypatch.setattr(reader_module, "_take", recording_take)
    out = np.empty(transform.domain.shape, dtype=source.dtype)
    assert numpy_reader.read_into(source, ReadContext(transform), out) is None
    assert out.shape == (10, 2)
    np.testing.assert_array_equal(
        out,
        np.array(
            [
                [0, 999],
                [1000, 1999],
                [2000, 2999],
                [3000, 3999],
                [4000, 4999],
                [5000, 5999],
                [6000, 6999],
                [7000, 7999],
                [8000, 8999],
                [9000, 9999],
            ]
        ),
    )
    assert seen
    assert all(math.prod(shape) <= 20_000 for shape in seen), seen


def test_numpy_reader_preserves_a_mask_in_the_supplied_buffer() -> None:
    source = np.ma.masked_greater(np.arange(12).reshape(3, 4), 7)
    transform = IndexTransform.from_shape(source.shape)[:, 1:]
    out = np.ma.masked_all(transform.domain.shape, dtype=source.dtype)
    assert numpy_reader.read_into(source, ReadContext(transform), out) is None
    np.testing.assert_array_equal(np.ma.getmaskarray(out), np.ma.getmaskarray(source[:, 1:]))
    np.testing.assert_array_equal(np.ma.filled(out, 0), np.ma.filled(source[:, 1:], 0))
