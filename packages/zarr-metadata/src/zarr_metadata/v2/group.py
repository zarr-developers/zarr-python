"""Zarr v2 group metadata types."""

from typing import Literal, TypedDict


class GroupMetadataV2(TypedDict):
    """
    Zarr v2 group metadata document (the ``.zgroup`` content).

    Attributes live in a sibling ``.zattrs`` file, so they are not part
    of this dict.
    """

    zarr_format: Literal[2]


__all__ = [
    "GroupMetadataV2",
]
