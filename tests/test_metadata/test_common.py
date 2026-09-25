"""Tests for metadata helpers shared by both Zarr formats."""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest

from zarr.core.metadata.common import parse_stored_regular_chunk_shape
from zarr.errors import ZarrUserWarning


@pytest.mark.parametrize(
    ("chunk_shape", "shape", "expected", "warning"),
    [
        ((4, 5), (10, 10), (4, 5), None),
        ((1, 1), (0, 0), (1, 1), None),
        ((0,), (0,), (1,), "of size 1"),
        ((False,), (0,), (1,), "of size 1"),
        ((np.int64(0),), (0,), (1,), "of size 1"),
        ((4, 0), (4, 0), (4, 1), "Dimension 1"),
        ((0, 0), (0, 0), (1, 1), "Dimension 0"),
        ((0,), (5,), (5,), "5 elements along this axis hold only the fill value"),
        ((4, 0), (4, 3), (4, 3), "3 elements along this axis hold only the fill value"),
    ],
    ids=[
        "valid",
        "valid-on-empty-axes",
        "legacy-zero",
        "legacy-json-false",
        "legacy-numpy-zero",
        "only-zero-size-corrected",
        "every-zero-size-corrected",
        "legacy-zero-on-grown-axis",
        "legacy-zero-on-grown-axis-2d",
    ],
)
def test_parse_stored_regular_chunk_shape(
    chunk_shape: tuple[Any, ...],
    shape: tuple[int, ...],
    expected: tuple[Any, ...],
    warning: str | None,
) -> None:
    """A valid chunk shape is returned as is; a chunk size of 0 is read as one
    chunk spanning the axis, with a warning saying how to re-save and, on an axis
    of positive length, that it holds only the fill value."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        parsed = parse_stored_regular_chunk_shape(chunk_shape, shape)
    assert parsed == expected
    messages = [str(w.message) for w in record if issubclass(w.category, ZarrUserWarning)]
    if warning is None:
        assert messages == []
    else:
        assert any(re.search(warning, message) for message in messages)
    for message in messages:
        assert "update_attributes({})" in message


def test_parse_stored_regular_chunk_shape_rejects_dimension_mismatch() -> None:
    """The chunk shape needs one entry per array axis."""
    with pytest.raises(ValueError, match="same number of dimensions"):
        parse_stored_regular_chunk_shape((4,), (10, 10))


def test_parse_stored_regular_chunk_shape_rejects_negative() -> None:
    """A negative chunk size is rejected even on a zero-length axis."""
    with pytest.raises(ValueError, match="chunk edge length must be >= 1, got -1"):
        parse_stored_regular_chunk_shape((-1,), (0,))
