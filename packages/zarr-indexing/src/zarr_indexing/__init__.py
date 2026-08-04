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

`LazyArray` wraps an array-API-like array and gives it deferred indexing
through `.lazy[...]`, yielding its reads as `Partition`s.

`plan_chunks` projects a transform through a caller-selected chunk grid without
coupling the result to a storage backend or scheduler. `selection_to_transform`
is also exported for consumers starting with a NumPy-style selection. The
`DimensionGridLike` Protocol describes the narrow grid surface chunk resolution
consumes without importing zarr.
"""

from importlib.metadata import version

from zarr_indexing.chunk_resolution import (
    ChunkCoverage,
    ChunkPlan,
    ChunkProjection,
    plan_chunks,
)
from zarr_indexing.composition import compose
from zarr_indexing.domain import IndexDomain
from zarr_indexing.errors import BoundsCheckError, VindexInvalidSelectionError
from zarr_indexing.grid import (
    DimensionGridLike,
    EdgeDimensionGrid,
    dimension_grids_from_chunks,
)
from zarr_indexing.json import (
    IndexDomainJSON,
    IndexTransformJSON,
    OutputIndexMapJSON,
    index_domain_from_json,
    index_domain_to_json,
    index_transform_from_json,
    index_transform_to_json,
    transform_from_canonical,
    transform_to_canonical,
)
from zarr_indexing.lazy_array import LazyArray, Partition
from zarr_indexing.messages import NdselError, normalize_ndsel, parse_ndsel
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.reader import BasicReader, NumPyReader, Reader, basic_reader, numpy_reader
from zarr_indexing.transform import (
    IndexTransform,
    array_map_dependent_axis,
    selection_to_transform,
)

__version__ = version("zarr-indexing")

__all__ = [
    "ArrayMap",
    "BasicReader",
    "BoundsCheckError",
    "ChunkCoverage",
    "ChunkPlan",
    "ChunkProjection",
    "ConstantMap",
    "DimensionGridLike",
    "DimensionMap",
    "EdgeDimensionGrid",
    "IndexDomain",
    "IndexDomainJSON",
    "IndexTransform",
    "IndexTransformJSON",
    "LazyArray",
    "NdselError",
    "NumPyReader",
    "OutputIndexMap",
    "OutputIndexMapJSON",
    "Partition",
    "Reader",
    "VindexInvalidSelectionError",
    "__version__",
    "array_map_dependent_axis",
    "basic_reader",
    "compose",
    "dimension_grids_from_chunks",
    "index_domain_from_json",
    "index_domain_to_json",
    "index_transform_from_json",
    "index_transform_to_json",
    "normalize_ndsel",
    "numpy_reader",
    "parse_ndsel",
    "plan_chunks",
    "selection_to_transform",
    "transform_from_canonical",
    "transform_to_canonical",
]
