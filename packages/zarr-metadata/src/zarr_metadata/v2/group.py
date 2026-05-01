"""Zarr v2 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

from typing import Literal

from typing_extensions import TypedDict


class GroupMetadataV2(TypedDict):
    """
    Zarr v2 group metadata document (the `.zgroup` content).

    Attributes live in a sibling `.zattrs` file, so they are not part
    of this dict.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]


__all__ = [
    "GroupMetadataV2",
]
