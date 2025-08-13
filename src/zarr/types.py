"""Public type definitions and constants"""

from typing import Literal

from zarr.core.types import (
    ArrayMetadataJSON_V2,
    ArrayMetadataJSON_V3,
    GroupMetadataJSON_V2,
    GroupMetadataJSON_V3,
)

__all__ = [
    "ArrayMetadataJSON_V2",
    "ArrayMetadataJSON_V3",
    "GroupMetadataJSON_V2",
    "GroupMetadataJSON_V3",
    "ZarrFormat",
]
ZarrFormat = Literal[2, 3]
"""The versions of Zarr that are supported."""
