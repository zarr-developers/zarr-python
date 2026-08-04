from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from zarr_indexing import IndexTransform
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


def expected_from_transform(
    source: np.ndarray[Any, Any], transform: IndexTransform
) -> np.ndarray[Any, Any]:
    shape = transform.domain.shape
    if np.prod(shape, dtype=np.intp) == 0:
        return np.empty(shape, dtype=source.dtype)
    if transform.input_rank == 0:
        domain_points = np.empty((1, 0), dtype=np.intp)
    else:
        domain_points = np.moveaxis(np.indices(shape, dtype=np.intp), 0, -1).reshape(
            -1, transform.input_rank
        )
        domain_points += np.asarray(transform.domain.inclusive_min, dtype=np.intp)
    source_points = transform.apply_many(domain_points)
    values = source[tuple(source_points.T)]
    return np.asarray(values).reshape(shape)


SOURCE = np.arange(6 * 7 * 8).reshape(6, 7, 8)
BASE = IndexTransform.from_shape(SOURCE.shape)


def zeroed(transform: IndexTransform) -> IndexTransform:
    return transform.translate_domain_to((0,) * transform.input_rank)


TRANSFORMS = (
    zeroed(BASE[1:6:2, 2, ::-2]),
    zeroed(BASE.oindex[[5, 1, 1], slice(1, 6), [7, 2]]),
    zeroed(BASE.vindex[np.array([[5], [1]]), np.array([[2, 4, 0]])]),
    zeroed(BASE.vindex[[5, 1], [2, 2]]),
    zeroed(BASE[:, 0:0, :]),
    zeroed(BASE[2, 3, 4]),
)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("reader_name", ["basic", "numpy"])
def test_builtin_readers_match_the_transform(transform: IndexTransform, reader_name: str) -> None:
    expected = expected_from_transform(SOURCE, transform)
    out = np.empty(transform.domain.shape, dtype=SOURCE.dtype)
    if reader_name == "basic":
        source = BasicOnlySource(SOURCE)
        result = basic_reader.read_into(source, transform, out)
        assert len(source.keys) == 1
    else:
        result = numpy_reader.read_into(SOURCE, transform, out)
    assert result is None
    np.testing.assert_array_equal(out, expected)


def test_numpy_reader_preserves_a_mask_in_the_supplied_buffer() -> None:
    source = np.ma.masked_greater(np.arange(12).reshape(3, 4), 7)
    transform = IndexTransform.from_shape(source.shape)[:, 1:]
    out = np.ma.masked_all(transform.domain.shape, dtype=source.dtype)
    assert numpy_reader.read_into(source, transform, out) is None
    np.testing.assert_array_equal(np.ma.getmaskarray(out), np.ma.getmaskarray(source[:, 1:]))
    np.testing.assert_array_equal(np.ma.filled(out, 0), np.ma.filled(source[:, 1:], 0))
