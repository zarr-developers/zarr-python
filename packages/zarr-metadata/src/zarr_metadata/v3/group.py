"""Zarr v3 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
"""

from collections.abc import Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3._common import ZarrV3ExtensionField


class ZarrV3ConsolidatedMetadataEnvelopeJSON(TypedDict, closed=True):
    """
    The `consolidated_metadata` member of a v3 group document, as the core spec fixes it.

    The spec allows this one additional field on a group, for historical
    reasons, and fixes its envelope: `kind` is `"inline"`,
    `must_understand` is false, and `metadata` is an object. The spec
    leaves the entries of `metadata` to the reference implementation, so
    here they are any JSON. `ZarrV3ConsolidatedMetadataJSON`, in
    `zarr_metadata.v3.consolidated`, types the entries as the reference
    implementation writes them: an array or group document at each path.

    See https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816
    """

    kind: Literal["inline"]
    must_understand: Literal[False]
    metadata: Mapping[str, JSONValue]


class ZarrV3GroupMetadataJSON(TypedDict, extra_items=ZarrV3ExtensionField):
    """
    Zarr v3 group metadata document (the `zarr.json` content for a group).

    Extra keys may contain arbitrary JSON values.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]
    consolidated_metadata: NotRequired[ZarrV3ConsolidatedMetadataEnvelopeJSON]


class ZarrV3GroupMetadataJSONPartial(TypedDict, total=False, extra_items=ZarrV3ExtensionField):
    """
    Partial form of `ZarrV3GroupMetadataJSON`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `ZarrV3GroupMetadataJSON` exactly.
    The only difference is `total=False`, which makes every key optional
    at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v3 group metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere.

    The `NotRequired[...]` wrappers are intentional: keeping them
    preserves byte-identical `__annotations__` with `ZarrV3GroupMetadataJSON` so the
    `==` check in `tests/test_partial_equivalence.py` passes without
    special-casing those fields (PEP 655 explicitly permits `NotRequired` inside
    `total=False`).

    Drift between this type and `ZarrV3GroupMetadataJSON` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]
    consolidated_metadata: NotRequired[ZarrV3ConsolidatedMetadataEnvelopeJSON]


ZarrV3GroupMetadataStoreKey = Literal["zarr.json"]
"""Literal type of the store key holding a v3 group's metadata document."""

ZARR_V3_GROUP_METADATA_STORE_KEY: Final[ZarrV3GroupMetadataStoreKey] = "zarr.json"
"""The store key a v3 group's metadata document is persisted under.

v3 uses one key for both node types; the document's `node_type` field
distinguishes a group from an array.
"""


__all__ = [
    "ZARR_V3_GROUP_METADATA_STORE_KEY",
    "ZarrV3ConsolidatedMetadataEnvelopeJSON",
    "ZarrV3GroupMetadataJSON",
    "ZarrV3GroupMetadataJSONPartial",
    "ZarrV3GroupMetadataStoreKey",
]
