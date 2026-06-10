from typing import Any

from zarr.core.array import (
    Array,
    AsyncArray,
    CompressorLike,
    CompressorsLike,
    FiltersLike,
    SerializerLike,
    ShardsLike,
)
from zarr.core.array_spec import ArrayConfigLike
from zarr.core.buffer import NDArrayLikeOrScalar
from zarr.core.chunk_key_encodings import ChunkKeyEncodingLike
from zarr.core.common import (
    JSON,
    AccessModeLiteral,
    BytesLike,
    ChunksLike,
    DimensionNamesLike,
    MemoryOrder,
    NodeType,
    ShapeLike,
    ZarrFormat,
)
from zarr.core.dtype import ZDTypeLike
from zarr.core.indexing import (
    BasicSelection,
    CoordinateSelection,
    Fields,
    MaskSelection,
    OrthogonalSelection,
    Selection,
)
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

__all__ = [
    "JSON",
    "AccessModeLiteral",
    "AnyArray",
    "AnyAsyncArray",
    "ArrayConfigLike",
    "ArrayV2",
    "ArrayV3",
    "AsyncArrayV2",
    "AsyncArrayV3",
    "BasicSelection",
    "BytesLike",
    "ChunkKeyEncodingLike",
    "ChunksLike",
    "CompressorLike",
    "CompressorsLike",
    "CoordinateSelection",
    "DimensionNamesLike",
    "Fields",
    "FiltersLike",
    "MaskSelection",
    "MemoryOrder",
    "NDArrayLikeOrScalar",
    "NodeType",
    "OrthogonalSelection",
    "Selection",
    "SerializerLike",
    "ShapeLike",
    "ShardsLike",
    "ZDTypeLike",
    "ZarrFormat",
]

type AnyAsyncArray = AsyncArray[Any]
"""A Zarr format 2 or 3 `AsyncArray`"""

type AsyncArrayV2 = AsyncArray[ArrayV2Metadata]
"""A Zarr format 2 `AsyncArray`"""

type AsyncArrayV3 = AsyncArray[ArrayV3Metadata]
"""A Zarr format 3 `AsyncArray`"""

type AnyArray = Array[Any]
"""A Zarr format 2 or 3 `Array`"""

type ArrayV2 = Array[ArrayV2Metadata]
"""A Zarr format 2 `Array`"""

type ArrayV3 = Array[ArrayV3Metadata]
"""A Zarr format 3 `Array`"""
