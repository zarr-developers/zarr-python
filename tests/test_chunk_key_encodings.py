from __future__ import annotations

import pytest

from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding, V2ChunkKeyEncoding


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
