"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under `zarr_metadata.v2` and `zarr_metadata.v3`.
Codec and dtype spec types live under `zarr_metadata.v3.codec` and
`zarr_metadata.v3.data_type`.
"""

from collections.abc import Mapping
from typing import NotRequired

from typing_extensions import TypeAliasType, TypedDict

JSONValue = TypeAliasType(
    "JSONValue",
    "int | float | bool | None | str | list[JSONValue] | tuple[JSONValue, ...] | Mapping[str, JSONValue]",
)
"""A recursive type alias for JSON-encodable values.

Defined via `TypeAliasType` (rather than a plain `TypeAlias`) so the
self-reference is a named recursion point that pydantic can resolve when
building a `TypeAdapter`; a bare recursive `TypeAlias` raises
`PydanticUserError`/`RecursionError` at validation time.
"""


class ZarrV3NamedConfigJSON(TypedDict):
    """
    Externally-tagged union member for a metadata field.

    The optional `configuration` mapping holds arbitrary JSON-encodable
    values. `must_understand` is implicitly true when absent.
    """

    name: str
    configuration: NotRequired[Mapping[str, JSONValue]]
    must_understand: NotRequired[bool]
