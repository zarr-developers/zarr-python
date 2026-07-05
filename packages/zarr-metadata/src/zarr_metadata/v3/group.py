"""Zarr v3 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
"""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata.v3.array import ExtensionFieldV3


class GroupMetadataV3(TypedDict, extra_items=ExtensionFieldV3):
    """
    Zarr v3 group metadata document (the `zarr.json` content for a group).

    Extra keys are permitted if they conform to `ExtensionFieldV3`.

    See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]


class GroupMetadataV3Partial(TypedDict, total=False, extra_items=ExtensionFieldV3):
    """
    Partial form of `GroupMetadataV3`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `GroupMetadataV3` exactly.
    The only difference is `total=False`, which makes every key optional
    at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v3 group metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere.

    The `NotRequired[...]` wrapper on `attributes` is intentional: keeping it
    preserves byte-identical `__annotations__` with `GroupMetadataV3` so the
    `==` check in `tests/test_partial_equivalence.py` passes without
    special-casing that field (PEP 655 explicitly permits `NotRequired` inside
    `total=False`).

    Drift between this type and `GroupMetadataV3` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, JSONValue]]


__all__ = [
    "GroupMetadataV3",
    "GroupMetadataV3Partial",
]
