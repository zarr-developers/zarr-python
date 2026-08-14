"""
Top-level cross-version primitives for Zarr metadata.

Version-specific types live under `zarr_metadata.v2` and `zarr_metadata.v3`.
Codec and dtype spec types live under `zarr_metadata.v3.codec` and
`zarr_metadata.v3.data_type`.
"""

from collections.abc import Mapping, Sequence
from typing import NotRequired

from typing_extensions import TypeAliasType, TypedDict

JSONValue = TypeAliasType(
    "JSONValue",
    int | float | bool | str | Sequence["JSONValue"] | Mapping[str, "JSONValue"] | None,
)
"""A recursive type alias for JSON-encodable values.

Defined via `TypeAliasType` (rather than a plain `TypeAlias`) so the
self-reference is a named recursion point that pydantic can resolve when
building a `TypeAdapter`; a bare recursive `TypeAlias` raises
`PydanticUserError`/`RecursionError` at validation time.

The array arm is the covariant `Sequence` rather than the invariant
`list["JSONValue"] | tuple["JSONValue", ...]`, so values typed with a
*narrower* element type still count as JSON values: a `list[str]` field on a
TypedDict is assignable to `JSONValue` under `Sequence` but not under
`list[JSONValue]` (`list` is invariant in its element type, and pyright's
diagnostic for that failure suggests exactly this change). This is what lets
downstream TypedDicts give their fields precise types (`Sequence[str]`,
`list[int]`, ...) while remaining assignable to `Mapping[str, JSONValue]`.
The type-level cost, accepted deliberately: `Sequence` says nothing about the
concrete container, and it admits `str`/`bytes` (`str` was already a union
arm); runtime code narrowing a JSON array must exclude `str`/`bytes`/
`bytearray` regardless of how this alias is spelled.
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
