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


__all__ = [
    "GroupMetadataV2",
    "ZGroupMetadata",
]
