"""Tests for strict closed-set JSON dispatch."""

from __future__ import annotations

import pytest

from zarr_chunk_key_encoding import (
    ChunkKeyConfigurationError,
    ChunkKeyEncoding,
    DefaultChunkKeyEncoding,
    UnknownChunkKeyEncodingError,
    V2ChunkKeyEncoding,
    chunk_key_encoding_from_json,
)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("default", DefaultChunkKeyEncoding()),
        ("v2", V2ChunkKeyEncoding()),
        ({"name": "default"}, DefaultChunkKeyEncoding()),
        (
            {"name": "default", "configuration": {"separator": "."}},
            DefaultChunkKeyEncoding(separator="."),
        ),
        (
            {"name": "v2", "configuration": {"separator": "/"}},
            V2ChunkKeyEncoding(separator="/"),
        ),
    ],
)
def test_from_json_dispatch(data: object, expected: ChunkKeyEncoding) -> None:
    """JSON dispatch supports both core encodings and both metadata forms."""
    assert chunk_key_encoding_from_json(data) == expected  # type: ignore[arg-type]


def test_from_json_unknown_name() -> None:
    """Unknown extension names receive the dedicated dispatch error."""
    with pytest.raises(UnknownChunkKeyEncodingError, match="not_an_encoding"):
        chunk_key_encoding_from_json({"name": "not_an_encoding"})


def test_from_json_missing_name() -> None:
    """Object-form metadata requires a string name."""
    with pytest.raises(ChunkKeyConfigurationError, match="'name'"):
        chunk_key_encoding_from_json({"configuration": {}})


def test_from_json_invalid_type() -> None:
    """Non-string, non-object input receives a package configuration error."""
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key encoding metadata"):
        chunk_key_encoding_from_json(3)  # type: ignore[arg-type]
