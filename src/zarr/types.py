from typing import Any

from zarr.core.array import Array, AsyncArray
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

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
