from __future__ import annotations

import pytest

from zarr.core.chunk_key_encodings import (
    DefaultChunkKeyEncoding,
    OffsetChunkKeyEncoding,
    V2ChunkKeyEncoding,
    parse_chunk_key_encoding,
)


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize(
    "coords",
    [(), (0,), (1, 2), (10, 0, 3)],
)
def test_default_encoding_round_trips(separator: str, coords: tuple[int, ...]) -> None:
    """Encoding coordinates and decoding the result returns the coordinates."""
    encoding = DefaultChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]

    key = encoding.encode_chunk_key(coords)
    assert encoding.decode_chunk_key(key) == coords


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize("coords", [(0,), (1, 2), (10, 0, 3)])
def test_v2_encoding_round_trips(separator: str, coords: tuple[int, ...]) -> None:
    """The v2 encoding round-trips coordinates for either separator."""
    encoding = V2ChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]

    key = encoding.encode_chunk_key(coords)
    assert encoding.decode_chunk_key(key) == coords


@pytest.mark.parametrize("separator", ["/", "."])
def test_v2_zero_dimensional_key_is_ambiguous(separator: str) -> None:
    """A 0-d v2 array stores its sole chunk under `"0"`, the same key a 1-d
    array uses for chunk 0, so decoding cannot recover the empty tuple on its
    own -- the array's dimensionality is what disambiguates it."""
    encoding = V2ChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]

    assert encoding.encode_chunk_key(()) == "0"
    assert encoding.decode_chunk_key("0") == (0,)


@pytest.mark.parametrize(
    "chunk_key",
    [
        "0/1",  # no "c" prefix at all
        "c0/1",  # "c" not followed by the separator
        "x/0/1",  # wrong prefix character
        "",
    ],
)
def test_default_encoding_rejects_key_without_prefix(chunk_key: str) -> None:
    """A key that does not carry the `c<separator>` prefix is not a chunk key
    for this encoding, and must be rejected rather than silently decoded."""
    encoding = DefaultChunkKeyEncoding(separator="/")

    with pytest.raises(ValueError, match="Invalid chunk key"):
        encoding.decode_chunk_key(chunk_key)


def test_default_encoding_rejects_key_using_the_other_separator() -> None:
    """A key encoded with `.` is not valid for a `/`-separated encoding."""
    encoding = DefaultChunkKeyEncoding(separator="/")

    with pytest.raises(ValueError, match="Invalid chunk key"):
        encoding.decode_chunk_key("c.0.1")


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize("coords", [(0,), (1, 2), (10, 0, 3)])
def test_offset_encoding_with_zero_offset_is_the_default_encoding(
    separator: str, coords: tuple[int, ...]
) -> None:
    """An all-zero offset gives byte-for-byte the default encoding's keys, so an array can
    switch to this encoding without any chunk moving."""
    default = DefaultChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]
    offset = OffsetChunkKeyEncoding(separator=separator, offset=(0,) * len(coords))  # type: ignore[arg-type]

    assert offset.encode_chunk_key(coords) == default.encode_chunk_key(coords)
    assert offset.decode_chunk_key(default.encode_chunk_key(coords)) == coords


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize("coords", [(0, 0), (5, 4), (7, 0)])
def test_offset_encoding_shifts_keys_and_round_trips(
    separator: str, coords: tuple[int, ...]
) -> None:
    encoding = OffsetChunkKeyEncoding(separator=separator, offset=(-2, 1))  # type: ignore[arg-type]

    key = encoding.encode_chunk_key(coords)
    assert key == separator.join(["c", str(coords[0] - 2), str(coords[1] + 1)])
    assert encoding.decode_chunk_key(key) == coords


def test_offset_encoding_produces_negative_key_components() -> None:
    """The chunk that holds array index 0 after two chunks were added in front of it is
    stored under a negative key; the chunk that held index 0 before keeps `c/0/0`."""
    encoding = OffsetChunkKeyEncoding(offset=(-2, 0))

    assert encoding.encode_chunk_key((0, 0)) == "c/-2/0"
    assert encoding.encode_chunk_key((2, 0)) == "c/0/0"
    assert encoding.decode_chunk_key("c/-2/0") == (0, 0)


def test_offset_encoding_rejects_a_rank_mismatch() -> None:
    encoding = OffsetChunkKeyEncoding(offset=(0, 0))

    with pytest.raises(ValueError, match="2 dimension"):
        encoding.encode_chunk_key((1, 2, 3))
    with pytest.raises(ValueError, match="2 dimension"):
        encoding.decode_chunk_key("c/1/2/3")


@pytest.mark.parametrize("offset", [(1.5,), ("1",), (True,), "12", 3])
def test_offset_encoding_rejects_a_non_integer_offset(offset: object) -> None:
    with pytest.raises(TypeError, match="sequence of integers"):
        OffsetChunkKeyEncoding(offset=offset)  # type: ignore[arg-type]


def test_offset_encoding_metadata_round_trips() -> None:
    encoding = OffsetChunkKeyEncoding(offset=(-1, 0), separator=".")

    assert encoding.to_dict() == {
        "name": "offset",
        "configuration": {"separator": ".", "offset": [-1, 0]},
    }
    assert parse_chunk_key_encoding(encoding.to_dict()) == encoding
    assert (
        parse_chunk_key_encoding({"name": "offset", "offset": (-1, 0), "separator": "."})
        == encoding
    )
    assert parse_chunk_key_encoding({"name": "offset", "offset": (3,)}) == OffsetChunkKeyEncoding(
        offset=(3,)
    )


def test_offset_encoding_requires_an_offset() -> None:
    with pytest.raises(ValueError, match="requires an offset"):
        parse_chunk_key_encoding({"name": "offset"})
    with pytest.raises(ValueError, match="requires an offset"):
        parse_chunk_key_encoding({"name": "offset", "configuration": {"separator": "/"}})
