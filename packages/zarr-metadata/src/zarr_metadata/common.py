"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under `zarr_metadata.v2` and `zarr_metadata.v3`.
Codec and dtype spec types live under `zarr_metadata.v3.codec` and
`zarr_metadata.v3.data_type`.
"""

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

JSON = str | int | float | bool | Mapping[str, "JSON"] | Sequence["JSON"] | None
"""Any valid JSON value."""


class NamedConfig(TypedDict):
    """
    Externally-tagged union member for a metadata field.

    Generic with two parameters: name literal and configuration mapping.
    """

    name: str
    configuration: NotRequired[Mapping[str, JSON]]
