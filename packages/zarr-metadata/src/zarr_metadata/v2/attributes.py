"""Zarr v2 user-attributes file content.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

from collections.abc import Mapping

from zarr_metadata._common import JSONValue

ZarrV2ZAttrsJSON = Mapping[str, JSONValue]
"""On-disk `.zattrs` file content.

A JSON object holding user-defined attributes for a v2 array or group.
Spec-defined keys for arrays / groups live in sibling `.zarray` / `.zgroup`
files (modeled by `ZarrV2ZArrayJSON` / `ZarrV2ZGroupJSON`). This type does not
constrain the keys or values of the attributes mapping.
"""


__all__ = [
    "ZarrV2ZAttrsJSON",
]
