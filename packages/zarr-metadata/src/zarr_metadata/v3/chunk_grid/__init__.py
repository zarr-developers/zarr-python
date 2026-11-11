"""Zarr v3 chunk grid metadata types.

Each chunk grid lives in its own submodule:

- `regular`     -- core v3 spec
- `rectilinear` -- zarr-extensions

Import a grid's types and its definition from its submodule, or from this
package, which re-exports every name its submodules export:
`<X>ChunkGridMetadata`, the canonical type for the grid's permitted JSON
shapes, `<X>ChunkGridObject`, `<X>ChunkGridConfiguration`, and
`<X>_CHUNK_GRID`.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-grids
"""

from zarr_metadata.v3.chunk_grid.rectilinear import (
    RECTILINEAR_CHUNK_GRID,
    RECTILINEAR_CHUNK_GRID_NAME,
    RectilinearChunkGridConfiguration,
    RectilinearChunkGridMetadata,
    RectilinearChunkGridName,
    RectilinearChunkGridObject,
    RectilinearDimSpec,
    canonical_chunk_shapes,
    canonical_dim_spec,
)
from zarr_metadata.v3.chunk_grid.regular import (
    REGULAR_CHUNK_GRID,
    REGULAR_CHUNK_GRID_NAME,
    RegularChunkGridConfiguration,
    RegularChunkGridMetadata,
    RegularChunkGridName,
    RegularChunkGridObject,
)

__all__ = [
    "RECTILINEAR_CHUNK_GRID",
    "RECTILINEAR_CHUNK_GRID_NAME",
    "REGULAR_CHUNK_GRID",
    "REGULAR_CHUNK_GRID_NAME",
    "RectilinearChunkGridConfiguration",
    "RectilinearChunkGridMetadata",
    "RectilinearChunkGridName",
    "RectilinearChunkGridObject",
    "RectilinearDimSpec",
    "RegularChunkGridConfiguration",
    "RegularChunkGridMetadata",
    "RegularChunkGridName",
    "RegularChunkGridObject",
    "canonical_chunk_shapes",
    "canonical_dim_spec",
]
