from typing import Any, TypeAlias

from zarr.core.array import Array
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

AnyArray: TypeAlias = Array[Any]
AnyArray.__doc__ = "A Zarr format 2 or 3 `Array`"

ArrayV2: TypeAlias = Array[ArrayV2Metadata]
ArrayV2.__doc__ = "A Zarr format 2 `Array`"

ArrayV3: TypeAlias = Array[ArrayV3Metadata]
ArrayV3.__doc__ = "A Zarr format 3 `Array`"
