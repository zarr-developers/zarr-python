"""The JSON Schema of a v3 `zarr.json`, as a scope reads it."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from zarr_metadata._typed_json import Schemas
from zarr_metadata.v3._definition import DataTypeDefinition, field_schemas, written_name
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context, ZarrV3Context, scoped
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.consolidated import (
    ZARR_V3_CONSOLIDATED_METADATA_KEY,
    ZarrV3ConsolidatedMetadataJSON,
)
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata._typed_json import JSONSchema, SchemaLeaf


def node_metadata_json_schema_v3(*, context: ZarrV3Context | None = None) -> JSONSchema:
    """The JSON Schema of a v3 `zarr.json` read in `context`: an array document or a group document, as `validate_node_metadata_v3` reads it, except for the rules.

    Use it in an editor that validates a `zarr.json` while it is
    written, or in a validator written in another language. The schema
    is JSON Schema draft 2020-12, in the form `json_schema` writes. Each
    extension point is a field as the scope `context` reads it: either a
    name that a definition in scope claims, with the configuration that
    definition declares, or a name that no definition claims. When the
    document names a data type in scope, the fill value must have the
    JSON shape that the data type's definition declares (for `int8`, an
    integer in [-128, 127]). A group's `consolidated_metadata` holds
    array and group documents, by path. A `null` value, which
    zarr-python 3.0 and 3.1 wrote, is refused, because the validator
    refuses it too. Each document is in `$defs` under the name of its
    TypedDict. `ZarrV3ArrayMetadataJSON` describes only an array.

    A JSON Schema describes each member on its own. It does not contain
    the rules that relate members to each other: one dimension name per
    dimension of the shape, a chunk grid that fits the shape, codecs in
    pipeline order with each codec checked against the chunk it
    receives, the hierarchy that the consolidated documents form below
    their group, and the checks in a definition's `rules`. So a document
    the schema accepts may still have a problem. Every JSON document
    that `validate_node_metadata_v3` finds no problem with is accepted
    by the schema. A validator reads JSON as a parser returns it, with
    arrays as lists. A model's `to_json` returns tuples, and a Python
    validator does not accept tuples as arrays.
    """
    scope = scoped(context, CORE_AND_EXTENSIONS)
    schemas = Schemas(_documents(scope))
    array = schemas.of(ZarrV3ArrayMetadataJSON)
    group = schemas.of(ZarrV3GroupMetadataJSON)
    return schemas.document({"anyOf": [array, group]})


def _documents(context: Context) -> SchemaLeaf:
    """The schema leaf for documents read in `context`: an array's fill value constrained by its data type, a group's consolidated metadata, and each field as `context` reads it."""
    fields = field_schemas(context)

    def leaf(annotation: object, schemas: Schemas) -> JSONSchema | None:
        if annotation is ZarrV3ArrayMetadataJSON:
            return schemas.defined(
                annotation, annotation.__name__, lambda: _array(context, schemas)
            )
        if annotation is ZarrV3GroupMetadataJSON:
            return schemas.defined(annotation, annotation.__name__, lambda: _group(schemas))
        return fields(annotation, schemas)

    return leaf


def _array(context: Context, schemas: Schemas) -> JSONSchema:
    """An array document: its TypedDict, plus a constraint on its fill value for each data type in scope, which applies when the document names that data type."""
    schema = schemas.object_of(ZarrV3ArrayMetadataJSON)
    held: list[JSONValue] = []
    for definition in context.definitions():
        if not isinstance(definition, DataTypeDefinition):
            continue
        fill_value = schemas.of(definition.fill_value)
        if len(fill_value) == 0:
            continue  # a data type that declares no fill value shape accepts any JSON
        name = written_name(definition)
        named: JSONSchema = {
            "anyOf": [name, {"type": "object", "properties": {"name": name}, "required": ["name"]}]
        }
        condition: JSONSchema = {
            "if": {"properties": {"data_type": named}, "required": ["data_type"]},
            "then": {"properties": {"fill_value": fill_value}},
        }
        held.append(condition)
    return schema if len(held) == 0 else {**schema, "allOf": held}


def _group(schemas: Schemas) -> JSONSchema:
    """A group document: its TypedDict, plus the consolidated metadata member that the model reads."""
    schema = schemas.object_of(ZarrV3GroupMetadataJSON)
    properties = cast("dict[str, JSONValue]", schema.get("properties", {}))
    consolidated: JSONSchema = schemas.of(ZarrV3ConsolidatedMetadataJSON)
    return {**schema, "properties": {**properties, ZARR_V3_CONSOLIDATED_METADATA_KEY: consolidated}}


__all__ = ["node_metadata_json_schema_v3"]
