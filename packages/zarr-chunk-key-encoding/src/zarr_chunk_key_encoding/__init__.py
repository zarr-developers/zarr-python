"""Chunk key encodings for Zarr version 3 arrays.

The package implements the two encodings defined by the Zarr v3 core
specification, strict decoding, JSON construction, and a prepared bounded view
for repeated operations against one grid shape.

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

from zarr_chunk_key_encoding._abc import ChunkKeyEncoding, ChunkKeyEncodingJSON
from zarr_chunk_key_encoding._bounded import BoundedChunkKeyEncoding
from zarr_chunk_key_encoding._default import DefaultChunkKeyEncoding
from zarr_chunk_key_encoding._errors import (
    ChunkKeyConfigurationError,
    ChunkKeyDecodeError,
    ChunkKeyEncodingError,
    InvalidChunkCoordsError,
    UnknownChunkKeyEncodingError,
)
from zarr_chunk_key_encoding._from_json import chunk_key_encoding_from_json
from zarr_chunk_key_encoding._parsing import Separator
from zarr_chunk_key_encoding._v2 import V2ChunkKeyEncoding

__version__ = version("zarr-chunk-key-encoding")

__all__ = [
    "BoundedChunkKeyEncoding",
    "ChunkKeyConfigurationError",
    "ChunkKeyDecodeError",
    "ChunkKeyEncoding",
    "ChunkKeyEncodingError",
    "ChunkKeyEncodingJSON",
    "DefaultChunkKeyEncoding",
    "InvalidChunkCoordsError",
    "Separator",
    "UnknownChunkKeyEncodingError",
    "V2ChunkKeyEncoding",
    "__version__",
    "chunk_key_encoding_from_json",
]
