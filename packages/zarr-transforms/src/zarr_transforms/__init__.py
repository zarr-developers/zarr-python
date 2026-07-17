"""Composable, lazy coordinate transforms for zarr array indexing.

This package implements TensorStore-inspired index transforms. The core idea:
every indexing operation (slicing, fancy indexing, etc.) produces a coordinate
mapping from user space to storage space. These mappings compose lazily — no
I/O until you explicitly read or write.

Key types:

- `IndexDomain` — a rectangular region of integer coordinates
- `IndexTransform` — maps input coordinates to storage coordinates
- `ConstantMap`, `DimensionMap`, `ArrayMap` — the three ways a single
  output dimension can depend on the input (see `output_map.py`)
- `compose` — chain two transforms into one

The chunk-resolution helpers (`iter_chunk_transforms`,
`sub_transform_to_selections`) and `selection_to_transform` are also exported
here: they form the surface the zarr integration layer (array indexing) depends
on. The `*Like` grid Protocols describe the chunk-grid surface chunk resolution
consumes without importing zarr.
"""

from zarr_transforms.chunk_resolution import (
    iter_chunk_transforms,
    sub_transform_to_selections,
)
from zarr_transforms.composition import compose
from zarr_transforms.domain import IndexDomain
from zarr_transforms.grid import ChunkGridLike, DimensionGridLike
from zarr_transforms.json import (
    IndexDomainJSON,
    IndexTransformJSON,
    OutputIndexMapJSON,
    index_domain_from_json,
    index_domain_to_json,
    index_transform_from_json,
    index_transform_to_json,
)
from zarr_transforms.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_transforms.transform import IndexTransform, selection_to_transform

__all__ = [
    "ArrayMap",
    "ChunkGridLike",
    "ConstantMap",
    "DimensionGridLike",
    "DimensionMap",
    "IndexDomain",
    "IndexDomainJSON",
    "IndexTransform",
    "IndexTransformJSON",
    "OutputIndexMap",
    "OutputIndexMapJSON",
    "compose",
    "index_domain_from_json",
    "index_domain_to_json",
    "index_transform_from_json",
    "index_transform_to_json",
    "iter_chunk_transforms",
    "selection_to_transform",
    "sub_transform_to_selections",
]
