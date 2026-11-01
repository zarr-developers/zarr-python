"""Zarr v3 chunk key encoding metadata types.

Each chunk key encoding lives in its own submodule:

- `default` -- v3 default encoding (`/`-separated)
- `v2`      -- v2-compatibility encoding (`.`-separated by default)

Both are defined by the v3 core spec:
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/default/index.rst
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/chunk-key-encodings/v2/index.rst

Import an encoding's types and its definition from its submodule:
`<X>ChunkKeyEncodingMetadata`, the canonical type for the encoding's
permitted JSON shapes, `<X>ChunkKeyEncodingObject`,
`<X>ChunkKeyEncodingConfiguration`, and `<X>_CHUNK_KEY_ENCODING`.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding
"""

__all__: list[str] = []
