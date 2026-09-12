"""Encoding contracts checked independently across ranks, bounds and key grammars."""

from __future__ import annotations

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zarr_chunk_key_encoding import (
    ChunkKeyDecodeError,
    DefaultChunkKeyEncoding,
    InvalidChunkCoordsError,
    V2ChunkKeyEncoding,
)

ENCODINGS = [
    DefaultChunkKeyEncoding(separator="/"),
    DefaultChunkKeyEncoding(separator="."),
    V2ChunkKeyEncoding(separator="/"),
    V2ChunkKeyEncoding(separator="."),
]


@pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
@given(coordinates=st.lists(st.integers(0, 2**64 - 1), max_size=5))
def test_bounded_roundtrip(
    encoding: DefaultChunkKeyEncoding | V2ChunkKeyEncoding, coordinates: list[int]
) -> None:
    """Binding resolves rank zero and preserves every positive unsigned-width index."""
    shape = [coordinate + 1 for coordinate in coordinates]
    bounded = encoding.to_bounded(shape)
    shape[:] = [0] * len(shape)
    expected = tuple(coordinates)
    key = bounded.encode(coordinates)
    assert key == encoding.encode(coordinates)
    assert bounded.decode(key) == expected
    assert all(type(index) is int for index in bounded.decode(key))


@pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
@given(key=st.text(max_size=40))
def test_decoding_matches_ascii_grammar(
    encoding: DefaultChunkKeyEncoding | V2ChunkKeyEncoding, key: str
) -> None:
    """Only canonical ASCII decimal components form a store key."""
    separator = re.escape(encoding.separator)
    component = r"(?:0|[1-9][0-9]*)"
    if isinstance(encoding, DefaultChunkKeyEncoding):
        pattern = rf"c(?:{separator}{component})*"
    else:
        pattern = rf"{component}(?:{separator}{component})*"
    if re.fullmatch(pattern, key):
        assert encoding.encode(encoding.decode(key)) == key
    else:
        with pytest.raises(ChunkKeyDecodeError):
            encoding.decode(key)


@pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
@given(shape=st.lists(st.integers(0, 2**64 - 1), min_size=1, max_size=5))
def test_bounded_rejects_extent(
    encoding: DefaultChunkKeyEncoding | V2ChunkKeyEncoding, shape: list[int]
) -> None:
    """The exclusive upper bound is invalid for both encoding and decoding."""
    bounded = encoding.to_bounded(shape)
    with pytest.raises(InvalidChunkCoordsError):
        bounded.encode(shape)
    with pytest.raises(ChunkKeyDecodeError):
        bounded.decode(encoding.encode(shape))


@pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
@given(coordinates=st.lists(st.integers(0, 2**64 - 1), min_size=1, max_size=5))
def test_bounded_rejects_rank_mismatch(
    encoding: DefaultChunkKeyEncoding | V2ChunkKeyEncoding, coordinates: list[int]
) -> None:
    """Canonical keys are rejected if their rank differs from the bound grid."""
    bounded = encoding.to_bounded([2**64] * (len(coordinates) + 1))
    with pytest.raises(InvalidChunkCoordsError):
        bounded.encode(coordinates)
    with pytest.raises(ChunkKeyDecodeError):
        bounded.decode(encoding.encode(coordinates))
