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
        ((0,), (5,), (5,), "grown to 5.*was not saved"),
        ((4, 0), (4, 3), (4, 3), "grown to 3.*was not saved"),
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
    chunk spanning the axis, with a warning naming the writer and how to re-save,
    and, if the axis has grown, that data written to it was not saved."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        parsed = parse_stored_regular_chunk_shape(
            chunk_shape, shape, legacy_writers="an old writer"
        )
    assert parsed == expected
    messages = [str(w.message) for w in record if issubclass(w.category, ZarrUserWarning)]
    if warning is None:
        assert messages == []
    else:
        assert any(re.search(warning, message) for message in messages)
    for message in messages:
        assert "an old writer" in message
        assert "update_attributes({})" in message


def test_parse_stored_regular_chunk_shape_rejects_dimension_mismatch() -> None:
    """The chunk shape needs one entry per array axis."""
    with pytest.raises(ValueError, match="same number of dimensions"):
        parse_stored_regular_chunk_shape((4,), (10, 10), legacy_writers="an old writer")


def test_parse_stored_regular_chunk_shape_rejects_negative() -> None:
    """A negative chunk size is rejected even on a zero-length axis."""
    with pytest.raises(ValueError, match="chunk edge length must be >= 1, got -1"):
        parse_stored_regular_chunk_shape((-1,), (0,), legacy_writers="an old writer")
