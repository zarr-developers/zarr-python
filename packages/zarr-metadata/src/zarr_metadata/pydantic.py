"""Optional pydantic (v2) integration: field types over the core models.

Importing this module requires pydantic; the core package deliberately does
not depend on it, so this module is never imported by `zarr_metadata` itself.

Each exported name is an `Annotated` field type over the corresponding core
model class — the instances ARE the core classes, so values interoperate
freely with non-pydantic code (equality, isinstance, nesting). Validation
delegates to the library: a raw document routes through `from_json` (the
single source of truth for structural validation and normalization, so
pydantic's field-level coercion can never bypass it), an existing model
instance passes through unchanged, and serialization emits the canonical
document via `to_json`. `MetadataValidationError` subclasses `ValueError`,
so a failed parse surfaces as a pydantic `ValidationError` carrying the
loc-annotated problem messages.

Usage:

    import zarr_metadata.pydantic as zmp

    class ArrayManifest(BaseModel):
        path: str
        metadata: zmp.ArrayMetadataV3

Static type checkers see each field type as its core model class, so
`manifest.metadata` is an `ArrayMetadataModelV3`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypeVar

from pydantic import BeforeValidator, InstanceOf, PlainSerializer, WithJsonSchema

from zarr_metadata.model import (
    ArrayMetadataModelV2,
    ArrayMetadataModelV3,
    ConsolidatedMetadataModelV2,
    ConsolidatedMetadataModelV3,
    GroupMetadataModelV2,
    GroupMetadataModelV3,
    NamedConfigModelV3,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_M = TypeVar("_M")


def _coerce_to(cls: type[_M], parse: Callable[[object], _M]) -> Callable[[object], _M]:
    """A validator that passes instances of `cls` through and parses anything else."""

    def coerce(value: object) -> _M:
        if isinstance(value, cls):
            return value
        return parse(value)

    return coerce


# JSON schemas describe the DOCUMENT form each field accepts (the validation
# input), not the in-memory model shape.
_DOCUMENT_SCHEMA = {"type": "object"}
_FIELD_SCHEMA = {"anyOf": [{"type": "string"}, {"type": "object"}]}

ArrayMetadataV3 = Annotated[
    InstanceOf[ArrayMetadataModelV3],
    BeforeValidator(_coerce_to(ArrayMetadataModelV3, ArrayMetadataModelV3.from_json)),
    PlainSerializer(ArrayMetadataModelV3.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ArrayMetadataV3"}),
]
"""Field type for a v3 array metadata document (`zarr.json` content)."""

ArrayMetadataV2 = Annotated[
    InstanceOf[ArrayMetadataModelV2],
    BeforeValidator(_coerce_to(ArrayMetadataModelV2, ArrayMetadataModelV2.from_json)),
    PlainSerializer(ArrayMetadataModelV2.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ArrayMetadataV2"}),
]
"""Field type for a v2 array metadata document (merged `.zarray` + `.zattrs` form)."""

GroupMetadataV3 = Annotated[
    InstanceOf[GroupMetadataModelV3],
    BeforeValidator(_coerce_to(GroupMetadataModelV3, GroupMetadataModelV3.from_json)),
    PlainSerializer(GroupMetadataModelV3.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "GroupMetadataV3"}),
]
"""Field type for a v3 group metadata document (`zarr.json` content)."""

GroupMetadataV2 = Annotated[
    InstanceOf[GroupMetadataModelV2],
    BeforeValidator(_coerce_to(GroupMetadataModelV2, GroupMetadataModelV2.from_json)),
    PlainSerializer(GroupMetadataModelV2.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "GroupMetadataV2"}),
]
"""Field type for a v2 group metadata document (merged `.zgroup` + `.zattrs` form)."""

ConsolidatedMetadataV3 = Annotated[
    InstanceOf[ConsolidatedMetadataModelV3],
    BeforeValidator(_coerce_to(ConsolidatedMetadataModelV3, ConsolidatedMetadataModelV3.from_json)),
    PlainSerializer(ConsolidatedMetadataModelV3.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ConsolidatedMetadataV3"}),
]
"""Field type for v3 inline consolidated metadata."""

ConsolidatedMetadataV2 = Annotated[
    InstanceOf[ConsolidatedMetadataModelV2],
    BeforeValidator(_coerce_to(ConsolidatedMetadataModelV2, ConsolidatedMetadataModelV2.from_json)),
    PlainSerializer(ConsolidatedMetadataModelV2.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ConsolidatedMetadataV2"}),
]
"""Field type for a v2 `.zmetadata` document."""

MetadataFieldV3 = Annotated[
    InstanceOf[NamedConfigModelV3],
    BeforeValidator(_coerce_to(NamedConfigModelV3, NamedConfigModelV3.from_json)),
    PlainSerializer(NamedConfigModelV3.to_json, return_type=str | dict),
    WithJsonSchema(_FIELD_SCHEMA | {"title": "MetadataFieldV3"}),
]
"""Field type for one v3 metadata field (bare name string or name + configuration)."""

__all__ = [
    "ArrayMetadataV2",
    "ArrayMetadataV3",
    "ConsolidatedMetadataV2",
    "ConsolidatedMetadataV3",
    "GroupMetadataV2",
    "GroupMetadataV3",
    "MetadataFieldV3",
]
