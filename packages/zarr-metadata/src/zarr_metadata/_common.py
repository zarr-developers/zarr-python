"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under `zarr_metadata.v2` and `zarr_metadata.v3`.
Codec and dtype spec types live under `zarr_metadata.v3.codec` and
`zarr_metadata.v3.data_type`.
"""

from collections.abc import Mapping
from typing import NotRequired, TypeAlias

from typing_extensions import TypedDict

JSONValue: TypeAlias = (
    int | float | bool | None | str | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]
)
"""A recursive type alias for JSON-encodable values."""


class NamedConfig(TypedDict):
    """
    Externally-tagged union member for a metadata field.

    The `configuration` mapping holds arbitrary JSON-encodable values;
    it is typed as `Mapping[str, JSONValue]`.
    """

    name: str
    configuration: NotRequired[Mapping[str, JSONValue]]
