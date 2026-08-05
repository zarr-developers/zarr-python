"""Zarr v2 user-attributes file content.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

from collections.abc import Mapping
from typing import Final, Literal

from zarr_metadata._common import JSONValue

ZarrV2ZAttrsJSON = Mapping[str, JSONValue]
"""On-disk `.zattrs` file content.

A JSON object holding user-defined attributes for a v2 array or group.
Spec-defined keys for arrays / groups live in sibling `.zarray` / `.zgroup`
files (modeled by `ZarrV2ZArrayJSON` / `ZarrV2ZGroupJSON`). This type does not
constrain the keys or values of the attributes mapping.
"""


ZarrV2AttributesStoreKey = Literal[".zattrs"]
"""Literal type of the store key holding a v2 node's user attributes."""

ZARR_V2_ATTRIBUTES_STORE_KEY: Final[ZarrV2AttributesStoreKey] = ".zattrs"
"""The store key a v2 node's user attributes are persisted under.

Shared by arrays and groups: both node types keep their attributes in a
sibling `.zattrs` file.
"""


__all__ = [
    "ZARR_V2_ATTRIBUTES_STORE_KEY",
    "ZarrV2AttributesStoreKey",
    "ZarrV2ZAttrsJSON",
]
