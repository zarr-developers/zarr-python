"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under `zarr_metadata.v2` and `zarr_metadata.v3`.
Codec and dtype spec types live under `zarr_metadata.v3.codec` and
`zarr_metadata.v3.data_type`.
"""

from collections.abc import Mapping
from typing import NotRequired

from typing_extensions import TypedDict


class NamedConfig(TypedDict):
    """
    Externally-tagged union member for a metadata field.

    The `configuration` mapping holds arbitrary JSON-encodable values;
    it is typed as `Mapping[str, object]` because the type system cannot
    express or verify JSON-encodability.
    """

    name: str
    configuration: NotRequired[Mapping[str, object]]
