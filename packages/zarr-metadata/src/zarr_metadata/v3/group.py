"""Zarr v3 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
"""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata.v3.array import ExtraField


class GroupMetadataV3(TypedDict, extra_items=ExtraField):  # type: ignore[call-arg]
    """
    Zarr v3 group metadata document (the `zarr.json` content for a group).

    Extra keys are permitted if they conform to `ExtraField`.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, object]]


__all__ = [
    "GroupMetadataV3",
]
