"""Zarr v3 group metadata types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata import JSON
    from zarr_metadata.v3.consolidated import ConsolidatedMetadataV3

from zarr_metadata.v3.array import AllowedExtraField


class GroupMetadataV3(TypedDict, extra_items=AllowedExtraField):  # type: ignore[call-arg]
    """
    Zarr v3 group metadata document (the ``zarr.json`` content for a group).

    Extra keys are permitted if they conform to ``AllowedExtraField``.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSON]]
    consolidated_metadata: NotRequired[ConsolidatedMetadataV3]


__all__ = [
    "GroupMetadataV3",
]
