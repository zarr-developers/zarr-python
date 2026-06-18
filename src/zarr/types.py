from typing import Any

from zarr.core.array import Array, AsyncArray
from zarr.core.common import JSON, ZarrFormat
from zarr.core.dtype.common import DTypeConfig_V2, DTypeJSON
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

__all__ = (
    "JSON",
    "AnyArray",
    "AnyAsyncArray",
    "ArrayV2",
    "ArrayV3",
    "AsyncArrayV2",
    "AsyncArrayV3",
    "DTypeConfig_V2",
    "DTypeJSON",
    "ZarrFormat",
)
