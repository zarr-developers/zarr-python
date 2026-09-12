"""Chunk key encodings prepared with a known chunk grid shape."""

from collections.abc import Sequence
from dataclasses import dataclass

from zarr_chunk_key_encoding._abc import ChunkKeyEncoding
from zarr_chunk_key_encoding._errors import (
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    InvalidChunkCoordsError,
)
from zarr_chunk_key_encoding._parsing import normalize_chunk_coords


def _normalize_grid_shape(grid_shape: Sequence[int]) -> tuple[int, ...]:
    """Normalize and validate a chunk grid shape once."""
    try:
        return normalize_chunk_coords(grid_shape)
    except InvalidChunkCoordsError as exc:
        raise ChunkKeyConfigurationError(
            f"Invalid chunk grid shape {grid_shape!r}: entries must be non-negative integers."
        ) from exc


def _in_grid(coords: tuple[int, ...], grid_shape: tuple[int, ...]) -> bool:
    """Return whether non-negative coordinates name a cell of a grid."""
    return len(coords) == len(grid_shape) and all(
        coordinate < extent for coordinate, extent in zip(coords, grid_shape, strict=True)
    )


@dataclass(frozen=True)
class BoundedChunkKeyEncoding:
    """An encoding prepared for repeated use with one chunk grid.

    The grid shape is normalized and validated during construction. Each
    subsequent operation therefore checks only its coordinates or decoded key
    against the prepared tuple.

    Parameters
    ----------
    encoding : ChunkKeyEncoding
        The underlying chunk key encoding.
    grid_shape : sequence of int
        The number of stored chunks or shards along each dimension.
    """

    encoding: ChunkKeyEncoding
    grid_shape: tuple[int, ...]

    def __init__(self, encoding: ChunkKeyEncoding, grid_shape: Sequence[int]) -> None:
        object.__setattr__(self, "encoding", encoding)
        object.__setattr__(self, "grid_shape", _normalize_grid_shape(grid_shape))

    def encode(self, chunk_coords: Sequence[int]) -> str:
        """Encode coordinates after checking them against the prepared grid.

        Raises
        ------
        InvalidChunkCoordsError
            If the coordinates are invalid, have the wrong rank, or are
            outside the grid.
        """
        coords = normalize_chunk_coords(chunk_coords)
        if not _in_grid(coords, self.grid_shape):
            raise InvalidChunkCoordsError(
                f"Chunk coordinates {coords!r} do not name a cell of the "
                f"chunk grid with shape {self.grid_shape!r}."
            )
        return self.encoding.encode(coords)

    def decode(self, chunk_key: str) -> tuple[int, ...]:
        """Decode a key after checking it against the prepared grid.

        Raises
        ------
        ChunkKeyDecodeError
            If the key is malformed, has the wrong rank, or names a chunk
            outside the grid.
        NotImplementedError
            If the underlying encoding does not support decoding.
        """
        if self.grid_shape == () and chunk_key == self.encoding.encode(()):
            return ()
        coords = self.encoding.decode(chunk_key)
        if not _in_grid(coords, self.grid_shape):
            raise ChunkKeyDecodeError(
                f"Chunk key {chunk_key!r} names a chunk outside the "
                f"chunk grid with shape {self.grid_shape!r}."
            )
        return coords
