"""Zarr v3 chunk grid metadata types.

Each chunk grid lives in its own submodule:

- `regular`     -- core v3 spec
- `rectilinear` -- zarr-extensions

Import a grid's types and its definition from its submodule:
`<X>ChunkGridMetadata`, the canonical type for the grid's permitted JSON
shapes, `<X>ChunkGridObject`, `<X>ChunkGridConfiguration`, and
`<X>_CHUNK_GRID`.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-grids
"""

__all__: list[str] = []
