"""Tests for the `default` chunk key encoding."""

from __future__ import annotations

import pytest

from zarr_chunk_key_encoding import (
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    DefaultChunkKeyEncoding,
    InvalidChunkCoordsError,
    Separator,
)


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize(
    ("chunk_coords", "expected_parts"),
    [
        ((), ["c"]),
        ((0,), ["c", "0"]),
        ((1, 23), ["c", "1", "23"]),
        ((0, 0, 0), ["c", "0", "0", "0"]),
        ((True, 5), ["c", "1", "5"]),  # bools normalize via __index__
        ((10**20,), ["c", str(10**20)]),
    ],
)
def test_encode_decode(
    separator: Separator, chunk_coords: tuple[int, ...], expected_parts: list[str]
) -> None:
    """For reasonable coordinates and both separators, `encode` produces the
    spec-defined key and `decode` inverts it."""
    encoding = DefaultChunkKeyEncoding(separator=separator)
    expected_key = separator.join(expected_parts)
    key = encoding.encode(chunk_coords)
    assert type(key) is str
    assert key == expected_key
    expected_roundtrip = tuple(int(c) for c in chunk_coords)
    assert encoding.decode(expected_key) == expected_roundtrip


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("default", DefaultChunkKeyEncoding()),
        ({"name": "default"}, DefaultChunkKeyEncoding()),
        ({"name": "default", "configuration": {}}, DefaultChunkKeyEncoding()),
        (
            {"name": "default", "configuration": {"separator": "."}},
            DefaultChunkKeyEncoding(separator="."),
        ),
        (
            {"name": "default", "configuration": {"separator": "/"}},
            DefaultChunkKeyEncoding(separator="/"),
        ),
        (
            {"name": "default", "must_understand": True},
            DefaultChunkKeyEncoding(),
        ),
    ],
)
def test_from_json(data: object, expected: DefaultChunkKeyEncoding) -> None:
    """All permitted JSON shapes construct the expected encoding."""
    assert DefaultChunkKeyEncoding.from_json(data) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("separator", [".", "/"])
def test_to_json_roundtrip(separator: Separator) -> None:
    """`to_json` emits the object form with an explicit separator, and
    `from_json` inverts it."""
    encoding = DefaultChunkKeyEncoding(separator=separator)
    data = encoding.to_json()
    assert data == {"name": "default", "configuration": {"separator": separator}}
    assert DefaultChunkKeyEncoding.from_json(data) == encoding


def test_invalid_separator() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key separator"):
        DefaultChunkKeyEncoding(separator="-")  # type: ignore[arg-type]


def test_encode_negative_coords() -> None:
    with pytest.raises(InvalidChunkCoordsError, match="non-negative"):
        DefaultChunkKeyEncoding().encode((0, -1))


def test_encode_non_integer_coords() -> None:
    with pytest.raises(InvalidChunkCoordsError, match="must be integers"):
        DefaultChunkKeyEncoding().encode((0.5,))  # type: ignore[arg-type]


def test_decode_missing_prefix() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="Invalid chunk key"):
        DefaultChunkKeyEncoding().decode("0/1")


def test_decode_wrong_separator() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="Invalid chunk key"):
        DefaultChunkKeyEncoding(separator="/").decode("c.0.1")


def test_decode_non_integer_part() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        DefaultChunkKeyEncoding().decode("c/x")


def test_decode_negative_part() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        DefaultChunkKeyEncoding().decode("c/-1")


def test_decode_non_canonical_part() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        DefaultChunkKeyEncoding().decode("c/01")


def test_decode_oversized_integer_part() -> None:
    """Integer conversion limits are reported through the package error hierarchy."""
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        DefaultChunkKeyEncoding().decode("c/" + "9" * 5_000)


def test_decode_empty_part() -> None:
    with pytest.raises(ChunkKeyDecodeError, match="not a canonical"):
        DefaultChunkKeyEncoding().decode("c/")


def test_from_json_wrong_name() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key encoding name"):
        DefaultChunkKeyEncoding.from_json({"name": "v2"})


def test_from_json_missing_name() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="missing required key 'name'"):
        DefaultChunkKeyEncoding.from_json({"configuration": {}})


def test_from_json_extra_envelope_key() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="unexpected keys"):
        DefaultChunkKeyEncoding.from_json({"name": "default", "separator": "/"})


def test_from_json_extra_configuration_key() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="unexpected keys"):
        DefaultChunkKeyEncoding.from_json(
            {"name": "default", "configuration": {"separator": "/", "suffix": ".bin"}}
        )


def test_from_json_configuration_not_a_mapping() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="'configuration' must be"):
        DefaultChunkKeyEncoding.from_json({"name": "default", "configuration": "/"})


def test_from_json_invalid_separator() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key separator"):
        DefaultChunkKeyEncoding.from_json({"name": "default", "configuration": {"separator": "-"}})


@pytest.mark.parametrize("value", ["yes", 1, None, False])
def test_from_json_must_understand_must_be_true(value: object) -> None:
    """Anything but the boolean `true` is rejected.

    `false` included: the v3 spec does not support it for chunk key
    encodings, since a reader that does not recognize one cannot skip it,
    so a document carrying it is malformed rather than merely redundant.
    """
    with pytest.raises(ChunkKeyConfigurationError, match="'must_understand'"):
        DefaultChunkKeyEncoding.from_json({"name": "default", "must_understand": value})


@pytest.mark.parametrize(
    "data",
    [
        {"name": "default", 1: "x", "zz": "y"},
        {"name": "default", "configuration": {1: "x", "zz": "y"}},
    ],
)
def test_from_json_mixed_type_keys(data: object) -> None:
    """Unexpected keys of mixed types are reported, not sorted into a TypeError.

    Every error this package raises derives from `ChunkKeyEncodingError`;
    sorting `1` against `"zz"` to build the message would break that.
    """
    with pytest.raises(ChunkKeyConfigurationError, match="unexpected keys"):
        DefaultChunkKeyEncoding.from_json(data)  # type: ignore[arg-type]


def test_from_json_invalid_type() -> None:
    with pytest.raises(ChunkKeyConfigurationError, match="Invalid chunk key encoding metadata"):
        DefaultChunkKeyEncoding.from_json(3)  # type: ignore[arg-type]
