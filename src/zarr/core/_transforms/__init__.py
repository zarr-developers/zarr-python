"""Composable, lazy coordinate transforms for zarr array indexing.

This package implements TensorStore-inspired index transforms. The core idea:
every indexing operation (slicing, fancy indexing, etc.) produces a coordinate
mapping from user space to storage space. These mappings compose lazily - no
I/O until you explicitly read or write.

Private package: this module is not part of the public zarr API. The leading
underscore in the package name signals this. Importers outside this package
must be limited to other private zarr modules.

Key types:

- `IndexDomain` -- a rectangular region of integer coordinates
- `IndexTransform` -- maps input coordinates to storage coordinates
- `ConstantMap`, `DimensionMap`, `ArrayMap` -- the three ways a single
  output dimension can depend on the input (see `output_map.py`)
- `compose` -- chain two transforms into one
"""

from zarr.core._transforms.composition import compose
from zarr.core._transforms.domain import IndexDomain
from zarr.core._transforms.output_map import (
    ArrayMap,
    ConstantMap,
    DimensionMap,
    OutputIndexMap,
)
from zarr.core._transforms.transform import IndexTransform

__all__ = [
    "ArrayMap",
    "ConstantMap",
    "DimensionMap",
    "IndexDomain",
    "IndexTransform",
    "OutputIndexMap",
    "compose",
]
