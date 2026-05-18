"""Zarr v2 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

from collections.abc import Mapping
from typing import Literal, NotRequired

from typing_extensions import TypedDict


class ZGroupMetadata(TypedDict):
    """
    On-disk `.zgroup` file content.

    Strict shape of the JSON document persisted at `<path>/.zgroup` for
    a v2 group. The spec defines exactly one field. User attributes live
    in a sibling `.zattrs` file and are NOT part of this type; see
    `ZAttrsMetadata`.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]


class GroupMetadataV2(TypedDict):
    """
    Zarr v2 group metadata document, in-memory merged form.

    Models the union of `.zgroup` (the spec-defined `zarr_format` field)
    and `.zattrs` (user attributes). On disk these are persisted as two
    separate files; this type folds them so a single TypedDict represents
    the complete in-memory state of a v2 group node. Consumers that read
    or write the real on-disk files should use `ZGroupMetadata` (strict
    `.zgroup`) plus `ZAttrsMetadata` directly.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    attributes: NotRequired[Mapping[str, object]]


class GroupMetadataV2Partial(TypedDict, total=False):
    """
    Partial form of `GroupMetadataV2`: every field is `NotRequired`.

    Field annotations mirror `GroupMetadataV2` exactly. The only difference is
    `total=False`, which makes every key optional at the type level.

    Use this when typing dicts that intentionally hold a subset of a complete
    v2 group metadata document — e.g. test fixtures that override only a few
    fields of a base template, or callers that build a fragment to be merged
    into a complete document elsewhere. Provided for symmetry with the other
    `*Partial` types; the practical effect is that `zarr_format` becomes optional.

    The `NotRequired[...]` wrapper on `attributes` is intentional: keeping it
    preserves byte-identical `__annotations__` with `GroupMetadataV2` so the
    `==` check in `tests/test_partial_equivalence.py` passes without
    special-casing that field (PEP 655 explicitly permits `NotRequired` inside
    `total=False`).

    Note: v2 group metadata has no `extra_items` setting (the v2 spec has no
    extension-field concept), so this partial inherits the same closed shape.

    Drift between this type and `GroupMetadataV2` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[2]
    attributes: NotRequired[Mapping[str, object]]


__all__ = [
    "GroupMetadataV2",
    "GroupMetadataV2Partial",
    "ZGroupMetadata",
]
