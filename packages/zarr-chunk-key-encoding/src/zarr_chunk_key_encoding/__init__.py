"""Chunk key encodings for Zarr version 3 arrays.

A chunk key encoding maps the grid index of a chunk — a tuple of
non-negative integers — to the string key under which that chunk is stored,
and (where well-defined) back again. This package provides:

- `ChunkKeyEncoding` — the abstract base class
- `DefaultChunkKeyEncoding`, `V2ChunkKeyEncoding` — the two encodings
  defined by the Zarr v3 core spec
- `BoundedChunkKeyEncoding` — an encoding bound to a known chunk grid
  (via `ChunkKeyEncoding.to_bounded`), whose finite key set supports membership
  testing, iteration, and `len`
- `chunk_key_encoding_from_json` / `parse_chunk_key_encoding` — construct
  encodings from JSON metadata or looser user input

JSON shapes are typed by `zarr-metadata`; this package supplies the runtime
behavior for those types.

Chunk key encoding is a Zarr v3 extension point, but this package covers the
closed set the core spec defines: there is no registration API and no entry
point group. The machinery for third-party encodings — registration,
discovery, provenance tiers — is common to every v3 extension point and
belongs in one shared package rather than reinvented here.

>>> from zarr_chunk_key_encoding import chunk_key_encoding_from_json
>>> encoding = chunk_key_encoding_from_json(
...     {"name": "default", "configuration": {"separator": "/"}}
... )
>>> encoding.encode((1, 23))
'c/1/23'
>>> encoding.decode("c/1/23")
(1, 23)
"""

from importlib.metadata import version

from zarr_chunk_key_encoding._abc import ChunkKey, ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding._bounded import BoundedChunkKeyEncoding
from zarr_chunk_key_encoding._default import DefaultChunkKeyEncoding
from zarr_chunk_key_encoding._errors import (
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    ChunkKeyEncodingError,
    InvalidChunkCoordsError,
    UnknownChunkKeyEncodingError,
)
from zarr_chunk_key_encoding._from_json import (
    CHUNK_KEY_ENCODINGS,
    ChunkKeyEncodingLike,
    ChunkKeyEncodingParams,
    chunk_key_encoding_from_json,
    get_chunk_key_encoding_class,
    parse_chunk_key_encoding,
)
from zarr_chunk_key_encoding._separator import SEPARATORS, Separator, parse_separator
from zarr_chunk_key_encoding._v2 import V2ChunkKeyEncoding

__version__ = version("zarr-chunk-key-encoding")

__all__ = [
    "CHUNK_KEY_ENCODINGS",
    "SEPARATORS",
    "BoundedChunkKeyEncoding",
    "ChunkKey",
    "ChunkKeyConfigurationError",
    "ChunkKeyDecodeError",
    "ChunkKeyEncoding",
    "ChunkKeyEncodingError",
    "ChunkKeyEncodingJSON",
    "ChunkKeyEncodingLike",
    "ChunkKeyEncodingParams",
    "DefaultChunkKeyEncoding",
    "InvalidChunkCoordsError",
    "Separator",
    "UnknownChunkKeyEncodingError",
    "V2ChunkKeyEncoding",
    "__version__",
    "chunk_key_encoding_from_json",
    "get_chunk_key_encoding_class",
    "parse_chunk_key_encoding",
    "parse_separator",
]
