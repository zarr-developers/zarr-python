"""Composable, lazy coordinate transforms for zarr array indexing.

This package implements TensorStore-inspired index transforms. The core idea:
every indexing operation (slicing, fancy indexing, etc.) produces a coordinate
mapping from user space to storage space. These mappings compose lazily — no
I/O until you explicitly read or write.

Key types:

- ``IndexDomain`` — a rectangular region of integer coordinates
- ``IndexTransform`` — maps input coordinates to storage coordinates
- ``ConstantMap``, ``DimensionMap``, ``ArrayMap`` — the three ways a single
  output dimension can depend on the input (see ``output_map.py``)
- ``compose`` — chain two transforms into one
"""

from zarr.core.transforms.composition import compose
from zarr.core.transforms.domain import IndexDomain
from zarr.core.transforms.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr.core.transforms.transform import IndexTransform

__all__ = [
    "ArrayMap",
    "ConstantMap",
    "DimensionMap",
    "IndexDomain",
    "IndexTransform",
    "OutputIndexMap",
    "compose",
]
