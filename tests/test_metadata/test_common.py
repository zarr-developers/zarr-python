"""Tests for metadata helpers shared by both Zarr formats."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from zarr.core.metadata.common import parse_stored_regular_chunk_shape
from zarr.errors import ZarrUserWarning


@pytest.mark.parametrize(
    ("chunk_shape", "shape", "expected", "warns"),
    [
        ((4, 5), (10, 10), (4, 5), False),
        ((1, 1), (0, 0), (1, 1), False),
        ((0,), (0,), (1,), True),
        ((False,), (0,), (1,), True),
        ((np.int64(0),), (0,), (1,), True),
        ((4, 0), (4, 0), (4, 1), True),
        ((0, 0), (0, 0), (1, 1), True),
    ],
    ids=[
        "valid",
        "valid-on-empty-axes",
        "legacy-zero",
        "legacy-json-false",
        "legacy-numpy-zero",
        "only-empty-axis-corrected",
        "every-empty-axis-corrected",
    ],
)
def test_parse_stored_regular_chunk_shape(
    chunk_shape: tuple[Any, ...], shape: tuple[int, ...], expected: tuple[Any, ...], warns: bool
) -> None:
    """A valid chunk shape is returned as is; a chunk size of 0 on a zero-length
    axis is read as 1, with a warning naming the writer and how to re-save."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        parsed = parse_stored_regular_chunk_shape(
            chunk_shape, shape, legacy_writers="an old writer"
        )
    assert parsed == expected
    messages = [str(w.message) for w in record if issubclass(w.category, ZarrUserWarning)]
    assert bool(messages) is warns
    for message in messages:
        assert "an old writer" in message
        assert "update_attributes({})" in message


def test_parse_stored_regular_chunk_shape_rejects_dimension_mismatch() -> None:
    """The chunk shape needs one entry per array axis."""
    with pytest.raises(ValueError, match="same number of dimensions"):
        parse_stored_regular_chunk_shape((4,), (10, 10), legacy_writers="an old writer")


@pytest.mark.parametrize(("chunk_shape", "shape"), [((0,), (5,)), ((4, 0), (4, 3))])
def test_parse_stored_regular_chunk_shape_rejects_zero_on_nonempty_axis(
    chunk_shape: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    """A chunk size of 0 is only tolerated on a zero-length axis."""
    with pytest.raises(ValueError, match="chunk edge length must be >= 1, got 0"):
        parse_stored_regular_chunk_shape(chunk_shape, shape, legacy_writers="an old writer")


def test_parse_stored_regular_chunk_shape_rejects_negative() -> None:
    """A negative chunk size is rejected even on a zero-length axis."""
    with pytest.raises(ValueError, match="chunk edge length must be >= 1, got -1"):
        parse_stored_regular_chunk_shape((-1,), (0,), legacy_writers="an old writer")
