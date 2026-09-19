"""Tests for codec kind classification."""

from __future__ import annotations

import pytest

from zarr_metadata.v3.codec.kind import codec_kind_of_name

# (codec name, expected kind). Classification is by name alone.
CASES: dict[str, str | None] = {
    "transpose": "array_array",
    "cast_value": "array_array",
    "scale_offset": "array_array",
    "bytes": "array_bytes",
    "sharding_indexed": "array_bytes",
    "blosc": "bytes_bytes",
    "crc32c": "bytes_bytes",
    "gzip": "bytes_bytes",
    "zstd": "bytes_bytes",
    "lightspeed": None,
}


@pytest.mark.parametrize(("name", "kind"), CASES.items(), ids=list(CASES))
def test_kind_of_name(name: str, kind: str | None) -> None:
    assert codec_kind_of_name(name) == kind
