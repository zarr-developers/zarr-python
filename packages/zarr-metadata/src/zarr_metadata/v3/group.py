"""Zarr v3 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
"""

from collections.abc import Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import ZarrV3ExtensionField


class ZarrV3GroupMetadataJSON(TypedDict, extra_items=ZarrV3ExtensionField):
    """
    Zarr v3 group metadata document (the `zarr.json` content for a group).

    Extra keys may contain arbitrary JSON values.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
    """

    zarr_format: Literal[3]
    """The Zarr format version, `3`."""
    node_type: Literal["group"]
    """The kind of node the document describes, `"group"`."""
    attributes: NotRequired[Mapping[str, JSONValue]]
    """User attributes: arbitrary JSON values by string key."""


ZarrV3GroupMetadataStoreKey = Literal["zarr.json"]
"""Literal type of the store key holding a v3 group's metadata document."""

ZARR_V3_GROUP_METADATA_STORE_KEY: Final[ZarrV3GroupMetadataStoreKey] = "zarr.json"
"""The store key a v3 group's metadata document is persisted under.

v3 uses one key for both node types; the document's `node_type` field
distinguishes a group from an array.
"""


__all__ = [
    "ZARR_V3_GROUP_METADATA_STORE_KEY",
    "ZarrV3GroupMetadataJSON",
    "ZarrV3GroupMetadataStoreKey",
]
