"""The JSON Schema of a v3 `zarr.json`, as a scope reads one."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from zarr_metadata._typed_json import Schemas
from zarr_metadata.v3._definition import DataTypeDefinition, field_schemas, written_name
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.consolidated import (
    ZARR_V3_CONSOLIDATED_METADATA_KEY,
    ZarrV3ConsolidatedMetadataJSON,
)
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata._typed_json import JSONSchema, SchemaLeaf


def node_metadata_json_schema_v3(*, context: Context | None = None) -> JSONSchema:
    """The JSON Schema of a v3 `zarr.json` read in `context`: an array document or a group document, as `validate_node_metadata_v3` reads one, but for the rules.

    For an editor that validates a `zarr.json` as it is written, or a
    validator in another language. JSON Schema draft 2020-12, as
    `json_schema` writes one. Each extension point is a field as
    `field_json_schema` writes one in `context`: one a definition in scope
    reads, or a name none of them claims. The fill value is the JSON shape
    the data type's definition declares for one -- an `int8`'s an integer
    in [-128, 127] -- when the document names a data type in scope. A
    group's `consolidated_metadata` holds array and group documents, by
    path; a `null` one, which a zarr-python 3.0.x bug wrote, is refused, as
    the validator refuses it. Each document is in `$defs` under the name of its
    TypedDict: `ZarrV3ArrayMetadataJSON` is an array's alone.

    A JSON Schema says what each member is, and what the rules say of
    members read together is not in it: one dimension name per dimension
    of the shape, a chunk grid that fits the shape, codecs in the order a
    pipeline takes them, each against the chunk it is handed, the
    hierarchy the documents of consolidated metadata make below their
    group, and what a definition's `rules` say. So a document it accepts may still have a
    problem, and a JSON document `validate_node_metadata_v3` finds none
    with, it accepts. A validator reads JSON as a parser gives it, arrays
    as lists: a model's `to_json` writes tuples, which a Python validator
    does not take for arrays.
    """
    scope = CORE_AND_EXTENSIONS if context is None else context
    schemas = Schemas(_documents(scope))
    array = schemas.of(ZarrV3ArrayMetadataJSON)
    group = schemas.of(ZarrV3GroupMetadataJSON)
    return schemas.document({"anyOf": [array, group]})


def _documents(context: Context) -> SchemaLeaf:
    """The schema leaf of the documents read in `context`: an array's fill value held to its data type, and a group's consolidated metadata; each field as `context` reads one."""
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
    """An array document: its TypedDict, and its fill value held to the data type it names, for each data type in scope."""
    schema = schemas.object_of(ZarrV3ArrayMetadataJSON)
    held: list[JSONValue] = []
    for definition in context.tables.get(DataTypeDefinition, {}).values():
        fill_value = schemas.of(cast("DataTypeDefinition[Any]", definition).fill_value)
        if len(fill_value) == 0:
            continue  # a data type that says nothing of its fill value takes any JSON
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
    """A group document: its TypedDict, and the consolidated metadata the model reads."""
    schema = schemas.object_of(ZarrV3GroupMetadataJSON)
    properties = cast("dict[str, JSONValue]", schema.get("properties", {}))
    consolidated: JSONSchema = schemas.of(ZarrV3ConsolidatedMetadataJSON)
    return {**schema, "properties": {**properties, ZARR_V3_CONSOLIDATED_METADATA_KEY: consolidated}}


__all__ = ["node_metadata_json_schema_v3"]
