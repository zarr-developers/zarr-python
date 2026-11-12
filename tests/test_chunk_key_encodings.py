from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import zarr
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding, V2ChunkKeyEncoding
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.storage import MemoryStore

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat


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


@pytest.mark.parametrize(
    ("name", "zarr_format", "chunk_keys"),
    [
        ("default", 3, ["c/0/0", "c/0/1"]),
        ("v2", 3, ["0.0", "0.1"]),
        ("v2", 2, ["0.0", "0.1"]),
    ],
)
async def test_create_array_short_hand_chunk_key_encoding(
    name: str, zarr_format: ZarrFormat, chunk_keys: list[str]
) -> None:
    """A bare chunk key encoding name given at array creation selects that encoding with
    its default separator, as it does when read from Zarr V3.1 short-hand metadata."""
    store = MemoryStore()
    array = zarr.create_array(
        store,
        shape=(1, 2),
        chunks=(1, 1),
        dtype="uint8",
        fill_value=0,
        chunk_key_encoding=name,
        zarr_format=zarr_format,
    )
    array[:] = 1
    metadata_keys = {"zarr.json", ".zarray", ".zattrs"}
    keys = sorted([key async for key in store.list() if key not in metadata_keys])
    assert keys == chunk_keys


@pytest.mark.parametrize("separator", ["/", "."])
def test_default_encoding_round_trips_negative_coordinates(separator: str) -> None:
    encoding = DefaultChunkKeyEncoding(separator=separator)  # type: ignore[arg-type]
    key = encoding.encode_chunk_key((-3, 0, 12))
    assert key == separator.join(["c", "-3", "0", "12"])
    assert encoding.decode_chunk_key(key) == (-3, 0, 12)


@pytest.mark.parametrize("chunk_key", ["c/-0", "c/1/-0", "c/-01"])
def test_default_encoding_rejects_negative_zero(chunk_key: str) -> None:
    """`-0` would name the same chunk as `0`, so it is not a valid key component."""
    with pytest.raises(ValueError, match="Invalid chunk key component"):
        DefaultChunkKeyEncoding(separator="/").decode_chunk_key(chunk_key)


def test_v2_encoding_rejects_negative_zero() -> None:
    with pytest.raises(ValueError, match="Invalid chunk key component"):
        V2ChunkKeyEncoding(separator=".").decode_chunk_key("1.-0")


@pytest.mark.parametrize(
    ("inclusive_min", "chunk_grid", "first_chunk"),
    [
        ((0, 0), {"name": "regular", "configuration": {"chunk_shape": (4, 5)}}, (0, 0)),
        ((-1, 3), {"name": "regular", "configuration": {"chunk_shape": (4, 5)}}, (-1, 0)),
        (
            (-1, 0),
            {
                "name": "offset",
                "configuration": {
                    "offset": (1, -1),
                    "base_chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": (4, 5)},
                    },
                },
            },
            (-1, 0),
        ),
        (
            (0, 0),
            {
                "name": "offset",
                "configuration": {
                    "offset": (6, -2),
                    "base_chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": (4, 5)},
                    },
                },
            },
            (-2, 0),
        ),
    ],
)
def test_array_metadata_keys_the_first_chunk_by_its_grid_index(
    inclusive_min: tuple[int, ...], chunk_grid: dict[str, Any], first_chunk: tuple[int, ...]
) -> None:
    """Chunk coordinates are zero-based on the array; the key names the grid index."""
    from tests.test_metadata.conftest import minimal_metadata_dict_v3

    document: dict[str, Any] = dict(minimal_metadata_dict_v3(chunk_grid=chunk_grid))
    del document["shape"]
    document["bounds"] = {
        "inclusive_min": inclusive_min,
        "exclusive_max": tuple(lo + 6 for lo in inclusive_min),
    }
    metadata = ArrayV3Metadata.from_dict(document)
    assert metadata.first_chunk == first_chunk
    assert metadata.encode_chunk_key((0, 0)) == "c/" + "/".join(map(str, first_chunk))
    assert metadata.encode_chunk_key((1, 1)) == "c/" + "/".join(str(f + 1) for f in first_chunk)
    assert metadata.decode_chunk_key(metadata.encode_chunk_key((1, 0))) == (1, 0)
