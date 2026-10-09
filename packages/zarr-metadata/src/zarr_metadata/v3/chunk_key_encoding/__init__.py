"""
Zarr v3 chunk key encoding metadata types.

Each chunk key encoding lives in its own submodule:

- `default` -- v3 default encoding (`/`-separated)
- `v2`      -- v2-compatibility encoding (`.`-separated by default)

Both are defined by the v3 core spec:
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/default/index.rst
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/v2/index.rst

The `<X>ChunkKeyEncodingMetadata` aliases re-exported here are the canonical
type for each encoding's permitted JSON shapes. For the underlying
`<X>ChunkKeyEncodingObject`, `<X>ChunkKeyEncodingConfiguration`, etc., import
directly from the leaf submodule.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

from zarr_metadata.v3.chunk_key_encoding.default import DefaultChunkKeyEncodingMetadata
from zarr_metadata.v3.chunk_key_encoding.v2 import V2ChunkKeyEncodingMetadata

__all__ = [
    "DefaultChunkKeyEncodingMetadata",
    "V2ChunkKeyEncodingMetadata",
]
