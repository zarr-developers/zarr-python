"""Zarr v3 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
"""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3.array import ZarrV3ExtensionField


class ZarrV3GroupMetadataJSON(TypedDict, extra_items=ZarrV3ExtensionField):
    """
    Zarr v3 group metadata document (the `zarr.json` content for a group).

    Extra keys may contain arbitrary JSON values.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]


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

    The `NotRequired[...]` wrapper on `attributes` is intentional: keeping it
    preserves byte-identical `__annotations__` with `ZarrV3GroupMetadataJSON` so the
    `==` check in `tests/test_partial_equivalence.py` passes without
    special-casing that field (PEP 655 explicitly permits `NotRequired` inside
    `total=False`).

    Drift between this type and `ZarrV3GroupMetadataJSON` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]


__all__ = [
    "ZarrV3GroupMetadataJSON",
    "ZarrV3GroupMetadataJSONPartial",
]
