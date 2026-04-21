"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under ``zarr_metadata.v2`` and ``zarr_metadata.v3``.
Codec and dtype spec types live under ``zarr_metadata.codec`` and
``zarr_metadata.dtype``.
"""

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

from typing_extensions import ReadOnly

JSON = str | int | float | bool | Mapping[str, "JSON"] | Sequence["JSON"] | None
"""Any valid JSON value."""


class NamedConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
    """
    Named-config envelope with optional configuration.

    Generic with two parameters: name literal and configuration mapping.
    """

    name: ReadOnly[TName]
    configuration: NotRequired[ReadOnly[TConfig]]


class NamedRequiredConfig[TName: str, TConfig: Mapping[str, object]](TypedDict):
    """
    Named-config envelope with required configuration.

    Generic with two parameters: name literal and configuration mapping.
    """

    name: ReadOnly[TName]
    configuration: ReadOnly[TConfig]


# Version-polymorphic unions -- imported after primitives to avoid circular import.
from zarr_metadata.v2.array import ArrayMetadataV2  # noqa: E402
from zarr_metadata.v2.group import GroupMetadataV2  # noqa: E402
from zarr_metadata.v3.array import ArrayMetadataV3  # noqa: E402
from zarr_metadata.v3.group import GroupMetadataV3  # noqa: E402

ArrayMetadata = ArrayMetadataV2 | ArrayMetadataV3
"""Any Zarr array metadata document (v2 or v3)."""

GroupMetadata = GroupMetadataV2 | GroupMetadataV3
"""Any Zarr group metadata document (v2 or v3)."""


__all__ = [
    "JSON",
    "ArrayMetadata",
    "ArrayMetadataV2",
    "ArrayMetadataV3",
    "GroupMetadata",
    "GroupMetadataV2",
    "GroupMetadataV3",
    "NamedConfig",
    "NamedRequiredConfig",
]
