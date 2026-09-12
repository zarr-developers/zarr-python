"""Behavioral parity with `zarr.core.chunk_key_encodings`.

These tests only run when `zarr` is importable (e.g. via
`just test-parity`, which layers this package over the repo-root
environment); the package itself does not depend on `zarr`.

Skipping is the right default -- the package-isolated test runs have no
`zarr` and should not fail for its absence -- but it would also let this
suite silently never run anywhere. Setting `ZARR_PARITY_REQUIRED` turns the
skip into a hard import error, so the job that exists to run these tests
fails loudly if its environment ever stops providing `zarr`.
"""

from __future__ import annotations

import os

import pytest

from zarr_chunk_key_encoding import (
    DefaultChunkKeyEncoding,
    Separator,
    V2ChunkKeyEncoding,
)

if os.environ.get("ZARR_PARITY_REQUIRED"):
    import zarr.core.chunk_key_encodings as zarr_cke
else:
    zarr_cke = pytest.importorskip("zarr.core.chunk_key_encodings")

COORDS: tuple[tuple[int, ...], ...] = ((), (0,), (1, 23), (0, 0, 0), (7, 8, 9, 10))


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize("chunk_coords", COORDS)
def test_default_encode_parity(separator: Separator, chunk_coords: tuple[int, ...]) -> None:
    """`default` produces byte-identical keys to zarr-python."""
    theirs = zarr_cke.DefaultChunkKeyEncoding(separator=separator)
    ours = DefaultChunkKeyEncoding(separator=separator)
    assert ours.encode(chunk_coords) == theirs.encode_chunk_key(chunk_coords)


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize("chunk_coords", COORDS)
def test_v2_encode_parity(separator: Separator, chunk_coords: tuple[int, ...]) -> None:
    """`v2` produces byte-identical keys to zarr-python."""
    theirs = zarr_cke.V2ChunkKeyEncoding(separator=separator)
    ours = V2ChunkKeyEncoding(separator=separator)
    assert ours.encode(chunk_coords) == theirs.encode_chunk_key(chunk_coords)


@pytest.mark.parametrize("separator", [".", "/"])
@pytest.mark.parametrize("chunk_coords", COORDS)
def test_json_parity(separator: Separator, chunk_coords: tuple[int, ...]) -> None:
    """`to_json` matches zarr-python's `to_dict` for both encodings."""
    assert (
        DefaultChunkKeyEncoding(separator=separator).to_json()
        == zarr_cke.DefaultChunkKeyEncoding(separator=separator).to_dict()
    )
    assert (
        V2ChunkKeyEncoding(separator=separator).to_json()
        == zarr_cke.V2ChunkKeyEncoding(separator=separator).to_dict()
    )


@pytest.mark.parametrize("encoding_type", [DefaultChunkKeyEncoding, V2ChunkKeyEncoding])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64"])
def test_unsigned_coordinate_bounds(
    encoding_type: type[DefaultChunkKeyEncoding | V2ChunkKeyEncoding], dtype: str
) -> None:
    """NumPy unsigned coordinates are normalized without signed narrowing."""
    import numpy as np

    from zarr_chunk_key_encoding import ChunkKeyDecodeError, InvalidChunkCoordsError

    limit = int(np.iinfo(dtype).max)
    coordinates = np.array([limit], dtype=dtype)
    encoding = encoding_type()
    bounded = encoding.to_bounded((limit + 1,))
    assert bounded.decode(bounded.encode(coordinates)) == (limit,)
    too_small = encoding.to_bounded((limit,))
    with pytest.raises(InvalidChunkCoordsError):
        too_small.encode(coordinates)
    with pytest.raises(ChunkKeyDecodeError):
        too_small.decode(encoding.encode(coordinates))
