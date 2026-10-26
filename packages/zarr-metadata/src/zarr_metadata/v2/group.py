"""Zarr v2 group metadata types.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

from collections.abc import Mapping
from typing import Final, Literal, NotRequired

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue


class ZarrV2ZGroupJSON(TypedDict, closed=True):
    """
    On-disk `.zgroup` file content.

    Strict shape of the JSON document persisted at `<path>/.zgroup` for
    a v2 group. The spec defines exactly one field and forbids others. User
    attributes live in a sibling `.zattrs` file and are NOT part of this
    type; see `ZarrV2ZAttrsJSON`.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L306-L313

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]


class ZarrV2GroupMetadataJSON(TypedDict, closed=True):
    """
    Zarr v2 group metadata document, in-memory merged form.

    Models the union of `.zgroup` (the spec-defined `zarr_format` field,
    https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L306-L313) and `.zattrs` (user attributes, https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L323-L330). On disk these are persisted as two
    separate files; this type folds them so a single TypedDict represents
    the complete in-memory state of a v2 group node. Consumers that read
    or write the real on-disk files should use `ZarrV2ZGroupJSON` (strict
    `.zgroup`) plus `ZarrV2ZAttrsJSON` directly.

    The type is closed: `.zgroup` holds no other keys ("Other keys MUST NOT be present",
    https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313),
    and `attributes` is the only member folded in.

    See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    zarr_format: Literal[2]
    attributes: NotRequired[Mapping[str, JSONValue]]


ZarrV2GroupMetadataStoreKey = Literal[".zgroup"]
"""Literal type of the store key holding a v2 group's metadata document."""

ZARR_V2_GROUP_METADATA_STORE_KEY: Final[ZarrV2GroupMetadataStoreKey] = ".zgroup"
"""The store key a v2 group's metadata document is persisted under."""


__all__ = [
    "ZARR_V2_GROUP_METADATA_STORE_KEY",
    "ZarrV2GroupMetadataJSON",
    "ZarrV2GroupMetadataStoreKey",
    "ZarrV2ZGroupJSON",
]
