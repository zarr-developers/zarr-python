"""Tests for prepared bounded chunk key encodings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Self

import pytest

from zarr_chunk_key_encoding import (
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    ChunkKeyEncoding,
    ChunkKeyEncodingJSON,
    DefaultChunkKeyEncoding,
    InvalidChunkCoordsError,
    V2ChunkKeyEncoding,
)

if TYPE_CHECKING:
    from zarr_metadata import JSONValue

ENCODINGS = [
    DefaultChunkKeyEncoding(),
    DefaultChunkKeyEncoding(separator="."),
    V2ChunkKeyEncoding(),
    V2ChunkKeyEncoding(separator="/"),
]


@pytest.mark.parametrize("encoding", ENCODINGS, ids=repr)
@pytest.mark.parametrize(
    ("grid_shape", "chunk_coords"),
    [
        ((), ()),
        ((1,), (0,)),
        ((5,), (4,)),
        ((2, 3), (0, 0)),
        ((2, 3), (1, 2)),
        ((4, 1, 7), (3, 0, 6)),
    ],
)
def test_encode_decode(
    encoding: ChunkKeyEncoding,
    grid_shape: tuple[int, ...],
    chunk_coords: tuple[int, ...],
) -> None:
    """Bounded operations agree with the underlying encoding and invert valid keys."""
    bounded = encoding.to_bounded(grid_shape)
    key = bounded.encode(chunk_coords)
    assert bounded.encoding is encoding
    assert bounded.grid_shape == grid_shape
    assert key == encoding.encode(chunk_coords)
    assert bounded.decode(key) == chunk_coords


class _CountingShape(Sequence[int]):
    """A shape that records reads so operations can prove they reuse normalization."""

    values = (2, 3)

    def __init__(self) -> None:
        self.reads = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> int:
        self.reads += 1
        return self.values[index]


def test_grid_shape_is_normalized_once() -> None:
    """Repeated operations use the tuple prepared during construction."""
    grid_shape = _CountingShape()
    bounded = DefaultChunkKeyEncoding().to_bounded(grid_shape)
    reads_after_construction = grid_shape.reads

    assert bounded.grid_shape == (2, 3)
    assert bounded.encode((1, 2)) == "c/1/2"
    assert bounded.decode("c/1/2") == (1, 2)
    assert grid_shape.reads == reads_after_construction


def test_zero_extent_grid_has_no_valid_coordinates() -> None:
    """A zero extent is valid configuration but leaves the bounded domain empty."""
    bounded = DefaultChunkKeyEncoding().to_bounded((0, 3))
    with pytest.raises(InvalidChunkCoordsError, match="shape"):
        bounded.encode((0, 0))
    with pytest.raises(ChunkKeyDecodeError, match="outside"):
        bounded.decode("c/0/0")


def test_invalid_grid_shape() -> None:
    """Grid entries must be non-negative integers."""
    for grid_shape in ((-1,), (1.5,), ("a",)):
        with pytest.raises(ChunkKeyConfigurationError):
            DefaultChunkKeyEncoding().to_bounded(grid_shape)  # type: ignore[arg-type]


def test_encode_invalid_coords() -> None:
    """Coordinates invalid in isolation retain the ordinary encode error."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(InvalidChunkCoordsError, match="non-negative"):
        bounded.encode((-1, 0))


def test_encode_wrong_rank() -> None:
    """Coordinates with the wrong rank are invalid for the prepared grid."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(InvalidChunkCoordsError, match="shape"):
        bounded.encode((1,))


def test_encode_out_of_bounds() -> None:
    """Coordinates at an extent are invalid for the prepared grid."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(InvalidChunkCoordsError, match="shape"):
        bounded.encode((2, 0))


def test_decode_malformed_key() -> None:
    """Malformed keys retain the underlying decode error."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(ChunkKeyDecodeError, match="Invalid chunk key"):
        bounded.decode("c/01/0")


def test_decode_wrong_rank() -> None:
    """A decoded key with the wrong rank is invalid for the prepared grid."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(ChunkKeyDecodeError, match="outside"):
        bounded.decode("c/1")


def test_decode_out_of_bounds() -> None:
    """A decoded key beyond an extent is invalid for the prepared grid."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    with pytest.raises(ChunkKeyDecodeError, match="outside"):
        bounded.decode("c/2/0")


def test_decode_rank_zero_out_of_domain() -> None:
    """Only the rank-zero encoding is valid for a rank-zero grid."""
    bounded = V2ChunkKeyEncoding().to_bounded(())
    with pytest.raises(ChunkKeyDecodeError, match="outside"):
        bounded.decode("5")


def test_removed_bounded_protocols() -> None:
    """The prepared view has no collection or persistence surface."""
    bounded = DefaultChunkKeyEncoding().to_bounded((2, 3))
    for name in ("from_json", "from_unbounded", "to_json", "__contains__", "__iter__", "__len__"):
        assert not hasattr(bounded, name)


class _NoDecode(ChunkKeyEncoding):
    """A minimal encoding without `decode`, for testing propagation."""

    name = "no-decode"

    @classmethod
    def from_json(cls, data: ChunkKeyEncodingJSON) -> Self:
        """Construct without inspecting the metadata."""
        return cls()

    def to_json(self) -> Mapping[str, JSONValue]:
        """Return the name-only object form."""
        return {"name": self.name}

    def encode(self, chunk_coords: Sequence[int]) -> str:
        """Join coordinates with `/`, using `z` for the rank-zero key."""
        return "/".join(str(c) for c in chunk_coords) or "z"


def test_decode_without_underlying_decode() -> None:
    """Missing decode propagates except for the directly recognizable rank-zero key."""
    with pytest.raises(NotImplementedError):
        _NoDecode().to_bounded((2, 3)).decode("0/0")
    assert _NoDecode().to_bounded(()).decode("z") == ()
