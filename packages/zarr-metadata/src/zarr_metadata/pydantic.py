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
        metadata: zmp.ZarrV3ArrayMetadata

Static type checkers see each field type as its core model class, so
`manifest.metadata` is a `zarr_metadata.model.ZarrV3ArrayMetadata`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypeVar

from pydantic import BeforeValidator, InstanceOf, PlainSerializer, WithJsonSchema

from zarr_metadata import model as _model

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

ZarrV3ArrayMetadata = Annotated[
    InstanceOf[_model.ZarrV3ArrayMetadata],
    BeforeValidator(_coerce_to(_model.ZarrV3ArrayMetadata, _model.ZarrV3ArrayMetadata.from_json)),
    PlainSerializer(_model.ZarrV3ArrayMetadata.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ZarrV3ArrayMetadata"}),
]
"""Field type for a v3 array metadata document (`zarr.json` content)."""

ZarrV2ArrayMetadata = Annotated[
    InstanceOf[_model.ZarrV2ArrayMetadata],
    BeforeValidator(_coerce_to(_model.ZarrV2ArrayMetadata, _model.ZarrV2ArrayMetadata.from_json)),
    PlainSerializer(_model.ZarrV2ArrayMetadata.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ZarrV2ArrayMetadata"}),
]
"""Field type for a v2 array metadata document (merged `.zarray` + `.zattrs` form)."""

ZarrV3GroupMetadata = Annotated[
    InstanceOf[_model.ZarrV3GroupMetadata],
    BeforeValidator(_coerce_to(_model.ZarrV3GroupMetadata, _model.ZarrV3GroupMetadata.from_json)),
    PlainSerializer(_model.ZarrV3GroupMetadata.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ZarrV3GroupMetadata"}),
]
"""Field type for a v3 group metadata document (`zarr.json` content)."""

ZarrV2GroupMetadata = Annotated[
    InstanceOf[_model.ZarrV2GroupMetadata],
    BeforeValidator(_coerce_to(_model.ZarrV2GroupMetadata, _model.ZarrV2GroupMetadata.from_json)),
    PlainSerializer(_model.ZarrV2GroupMetadata.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ZarrV2GroupMetadata"}),
]
"""Field type for a v2 group metadata document (merged `.zgroup` + `.zattrs` form)."""

ZarrV3ConsolidatedMetadata = Annotated[
    InstanceOf[_model.ZarrV3ConsolidatedMetadata],
    BeforeValidator(
        _coerce_to(_model.ZarrV3ConsolidatedMetadata, _model.ZarrV3ConsolidatedMetadata.from_json)
    ),
    PlainSerializer(_model.ZarrV3ConsolidatedMetadata.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ZarrV3ConsolidatedMetadata"}),
]
"""Field type for v3 inline consolidated metadata."""

ZarrV2ConsolidatedMetadata = Annotated[
    InstanceOf[_model.ZarrV2ConsolidatedMetadata],
    BeforeValidator(
        _coerce_to(_model.ZarrV2ConsolidatedMetadata, _model.ZarrV2ConsolidatedMetadata.from_json)
    ),
    PlainSerializer(_model.ZarrV2ConsolidatedMetadata.to_json, return_type=dict),
    WithJsonSchema(_DOCUMENT_SCHEMA | {"title": "ZarrV2ConsolidatedMetadata"}),
]
"""Field type for a v2 `.zmetadata` document."""

ZarrV3MetadataField = Annotated[
    InstanceOf[_model.ZarrV3NamedConfig],
    BeforeValidator(_coerce_to(_model.ZarrV3NamedConfig, _model.ZarrV3NamedConfig.from_json)),
    PlainSerializer(_model.ZarrV3NamedConfig.to_json, return_type=dict),
    WithJsonSchema(_FIELD_SCHEMA | {"title": "ZarrV3MetadataField"}),
]
"""Field type for one v3 metadata field (bare name string or name + configuration)."""

__all__ = [
    "ZarrV2ArrayMetadata",
    "ZarrV2ConsolidatedMetadata",
    "ZarrV2GroupMetadata",
    "ZarrV3ArrayMetadata",
    "ZarrV3ConsolidatedMetadata",
    "ZarrV3GroupMetadata",
    "ZarrV3MetadataField",
]
