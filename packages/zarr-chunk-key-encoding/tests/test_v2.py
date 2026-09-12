"""Tests for the `v2` chunk key encoding."""

from __future__ import annotations

import pytest

from zarr_chunk_key_encoding import (
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    InvalidChunkCoordsError,
    Separator,
    V2ChunkKeyEncoding,
)


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize(
    ("chunk_coords", "expected_parts"),
    [
        ((0,), ["0"]),
        ((1, 23), ["1", "23"]),
        ((0, 0, 0), ["0", "0", "0"]),
        ((True, 5), ["1", "5"]),  # bools normalize via __index__
        ((10**20,), [str(10**20)]),
    ],
)
def test_encode_decode(
    separator: Separator, chunk_coords: tuple[int, ...], expected_parts: list[str]
) -> None:
    """For reasonable coordinates and both separators, `encode` produces the
    spec-defined key and `decode` inverts it."""
    encoding = V2ChunkKeyEncoding(separator=separator)
    expected_key = separator.join(expected_parts)
    key = encoding.encode(chunk_coords)
    assert type(key) is str
    assert key == expected_key
    expected_roundtrip = tuple(int(c) for c in chunk_coords)
    assert encoding.decode(expected_key) == expected_roundtrip


def test_encode_rank_zero() -> None:
    """The empty grid index encodes to "0", which decodes to (0,): the
    documented rank-zero ambiguity of this encoding."""
    encoding = V2ChunkKeyEncoding()
    assert encoding.encode(()) == "0"
    assert encoding.decode("0") == (0,)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("v2", V2ChunkKeyEncoding()),
        ({"name": "v2"}, V2ChunkKeyEncoding()),
        ({"name": "v2", "configuration": {}}, V2ChunkKeyEncoding()),
        (
            {"name": "v2", "configuration": {"separator": "/"}},
            V2ChunkKeyEncoding(separator="/"),
        ),
        (
            {"name": "v2", "configuration": {"separator": "."}},
            V2ChunkKeyEncoding(separator="."),
        ),
        ({"name": "v2", "must_understand": True}, V2ChunkKeyEncoding()),
    ],
)
def test_from_json(data: object, expected: V2ChunkKeyEncoding) -> None:
    """All permitted JSON shapes construct the expected encoding."""
    assert V2ChunkKeyEncoding.from_json(data) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("separator", [".", "/"])
def test_to_json_roundtrip(separator: Separator) -> None:
    """`to_json` emits the object form with an explicit separator, and
    `from_json` inverts it."""
    encoding = V2ChunkKeyEncoding(separator=separator)
    data = encoding.to_json()
    assert data == {"name": "v2", "configuration": {"separator": separator}}
    assert V2ChunkKeyEncoding.from_json(data) == encoding


def test_invalid_separator() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key separator"):
        V2ChunkKeyEncoding(separator="_")  # type: ignore[arg-type]


def test_encode_negative_coords() -> None:
    with pytest.raises(InvalidChunkCoordsError, match="non-negative"):
        V2ChunkKeyEncoding().encode((-1,))


def test_encode_non_integer_coords() -> None:
    with pytest.raises(InvalidChunkCoordsError, match="must be integers"):
        V2ChunkKeyEncoding().encode(("a",))  # type: ignore[arg-type]


def test_decode_empty_key() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="Invalid chunk key ''"):
        V2ChunkKeyEncoding().decode("")


def test_decode_non_integer_part() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        V2ChunkKeyEncoding().decode("0.x")


def test_decode_non_canonical_part() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        V2ChunkKeyEncoding().decode("00")


def test_decode_wrong_separator() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        V2ChunkKeyEncoding(separator=".").decode("0/1")


def test_from_json_wrong_name() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key encoding name"):
        V2ChunkKeyEncoding.from_json("default")
