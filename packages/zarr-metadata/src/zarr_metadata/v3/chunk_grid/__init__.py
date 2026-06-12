"""
Zarr v3 chunk grid metadata types.

Each chunk grid lives in its own submodule:

- `regular`     -- core v3 spec
- `rectilinear` -- zarr-extensions

The `<X>ChunkGridMetadata` aliases re-exported here are the canonical type
for each grid's permitted JSON shapes. For the underlying
`<X>ChunkGridObject`, `<X>ChunkGridConfiguration`, etc., import directly
from the leaf submodule.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-grids
"""

from zarr_metadata.v3.chunk_grid.rectilinear import RectilinearChunkGridMetadata
from zarr_metadata.v3.chunk_grid.regular import RegularChunkGridMetadata

__all__ = [
    "RectilinearChunkGridMetadata",
    "RegularChunkGridMetadata",
]
