"""The JSON Schema of a Zarr v3 field and of a whole v3 `zarr.json`, as a scope reads them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from zarr_metadata._typed_json import Schemas, is_alias
from zarr_metadata.definition._context import scoped
from zarr_metadata.definition._definition import (
    Definition,
    EmptyConfiguration,
    as_kind,
    field_kind,
)
from zarr_metadata.definition._envelope import EXTENSION_NAME_SCHEMA_PATTERN
from zarr_metadata.v3._context import ZarrV3Context, ZarrV3Kind, is_v3_kind
from zarr_metadata.v3._core import CORE_AND_EXTENSIONS
from zarr_metadata.v3._definition import (
    RAW_BYTES_NAME,
    RAW_BYTES_NAME_PATTERN,
    STATIC_SIZE_ALIASES,
    ArrayExtraFieldDefinition,
    CodecDefinition,
    DataTypeDefinition,
    ExtraFieldDefinition,
    GroupExtraFieldDefinition,
)
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

if TYPE_CHECKING:
    from zarr_metadata._common import JSONValue
    from zarr_metadata._typed_json import JSONSchema, SchemaLeaf


def kind_field(kind: type[Definition[Any]]) -> object:
    """The field alias for a member that holds any field of `kind`: the first alias the kind declares that does not restrict the field. For example, `ZarrV3CodecField` instead of `ZarrV3StaticCodecField`."""
    return next(alias for alias in kind.field_aliases if alias not in STATIC_SIZE_ALIASES)


_RAW_BYTES_SCHEMA_PATTERN: Final = f"^{RAW_BYTES_NAME_PATTERN.pattern}(?![\\s\\S])"
"""`RAW_BYTES_NAME_PATTERN`, matched against the whole name, as a JSON Schema pattern.

A lookahead for "no character at all" anchors the match to the end of
the name. A `$` would also match before a final newline in a validator
that matches patterns the way Python does, so `"r16\\n"`, which is not a
valid name, would be read as raw bits.
"""


def field_json_schema_v3(
    kind: type[ZarrV3Kind], *, context: ZarrV3Context | None = None
) -> JSONSchema:
    """The JSON Schema of one metadata field read as `kind` in `context` (`CORE_AND_EXTENSIONS` when none is given): everything `resolve` checks, except the rules. A scope of another format raises `TypeError`, as it does for every v3 reader. An extra field is written as the member alone, `{key: value}`.

    The schema accepts a field that one of the definitions in scope reads
    (its name, its configuration as the TypedDict declares it, an optional
    `must_understand` of `true`, and its bare name when it needs no
    configuration). It also accepts a name that no definition claims, with
    any configuration, which is what keeps the format open. A field inside
    a configuration is written the same way, in the same scope, and a
    member that accepts only codecs of static size accepts only those. The
    output is JSON Schema draft 2020-12, as `json_schema` writes it. The
    nested fields and the configuration of each definition are in
    `$defs`, under the name of the field alias or TypedDict. Constraints
    that only a rule enforces (a blosc `typesize` checked against its
    `shuffle`) are not in the schema, so a field it accepts may still have
    a problem.
    """
    asked = as_kind(kind)
    if not is_v3_kind(asked):
        msg = (
            f"{asked.__name__} is a kind of another format; the JSON Schema writer writes "
            "Zarr v3 fields only"
        )
        raise TypeError(msg)
    context = scoped(context, CORE_AND_EXTENSIONS)
    schemas = Schemas(field_schemas(context))
    if issubclass(asked, (ArrayExtraFieldDefinition, GroupExtraFieldDefinition)):
        return schemas.document(_extra_field_schema(asked, context, schemas))
    return schemas.document(schemas.of(kind_field(asked)))


def field_schemas(context: ZarrV3Context) -> SchemaLeaf:
    """The schema leaf that `field_json_schema_v3` uses. It writes each field alias as a field of its kind, as `context` reads it."""

    def leaf(annotation: object, schemas: Schemas) -> JSONSchema | None:
        kind = field_kind(annotation)
        if kind is None or not is_alias(annotation):
            return None
        alias = annotation
        static = annotation in STATIC_SIZE_ALIASES
        return schemas.defined(
            alias, alias.__name__, lambda: _field_schema(kind, static, context, schemas)
        )

    return leaf


def _field_schema(
    kind: type[Definition[Any]], static: bool, context: ZarrV3Context, schemas: Schemas
) -> JSONSchema:
    """The schema of a field of `kind` as `context` reads it: a field that a definition in scope reads, or a field that none of them claims.

    `kind` is always a v3 kind. The v3 document TypedDicts hold v3 fields
    only, a v3 definition whose configuration holds a field of the other
    format is refused when it is built, and `field_json_schema_v3` refuses
    a kind of the other format before it writes anything. The check below
    only narrows `kind` for the type checker.
    """
    if not is_v3_kind(kind):
        msg = f"{kind.__name__} is a kind of another format, which a v3 schema does not hold"
        raise TypeError(msg)
    table = context.definitions_of(kind)
    branches: list[JSONValue] = []
    for definition in table.values():
        if static and isinstance(definition, CodecDefinition) and definition.size != "static":
            continue
        branches.extend(_read_by(definition, schemas))
    branches.extend(_unclaimed(table))
    return {"anyOf": branches}


def _extra_field_schema(
    kind: type[ArrayExtraFieldDefinition[Any] | GroupExtraFieldDefinition[Any]],
    context: ZarrV3Context,
    schemas: Schemas,
) -> JSONSchema:
    """The schema of an extra field of `kind` as `context` reads it, written as the member alone, `{key: value}`: a member whose key a definition in scope claims, with the value its configuration declares, or a member with any other key that the spec does not define, and any JSON value."""
    claimed = list(context.definitions_of(kind).values())
    branches: list[JSONValue] = [
        {
            "type": "object",
            "properties": {definition.name: schemas.of(definition.configuration)},
            "required": [definition.name],
            "additionalProperties": False,
        }
        for definition in claimed
    ]
    unclaimed: JSONSchema = {"type": "object", "minProperties": 1, "maxProperties": 1}
    # Neither a claimed key nor a member the spec defines is an unclaimed member.
    others = sorted({definition.name for definition in claimed} | kind.spec_members)
    unclaimed["propertyNames"] = {"not": {"enum": [*others]}}
    branches.append(unclaimed)
    return {"anyOf": branches}


def written_name(definition: Definition[Any]) -> JSONSchema:
    """The JSON Schema of the names a document may use for `definition`: its name, or `r` followed by a size for raw bits."""
    if isinstance(definition, DataTypeDefinition) and definition.name == RAW_BYTES_NAME:
        return {"type": "string", "pattern": _RAW_BYTES_SCHEMA_PATTERN}
    return {"const": definition.name}


def _read_by(definition: Definition[Any], schemas: Schemas) -> list[JSONValue]:
    """The schemas of the fields `definition` reads: an object with its name and configuration, and its bare name when it needs no configuration.

    The raw bits name encodes the configuration, so the configuration
    written next to it must be empty.
    """
    carried = isinstance(definition, DataTypeDefinition) and definition.name == RAW_BYTES_NAME
    name = written_name(definition)
    bare = carried or not definition.requires_configuration
    envelope: JSONSchema = {
        "type": "object",
        "properties": {
            "name": name,
            "configuration": schemas.of(
                EmptyConfiguration if carried else definition.configuration
            ),
            "must_understand": {"const": True},
        },
        "required": ["name"] if bare else ["name", "configuration"],
        "additionalProperties": False,
    }
    return [name, envelope] if bare else [envelope]


def _unclaimed(table: Mapping[str, Definition[Any]]) -> list[JSONValue]:
    """The schemas of the fields that no definition in `table` claims: a name that none of them uses, either bare or with any configuration."""
    claimed: list[JSONValue] = [written_name(definition) for definition in table.values()]
    name: JSONSchema = {"type": "string", "pattern": EXTENSION_NAME_SCHEMA_PATTERN}
    if len(claimed) != 0:
        name["not"] = {"anyOf": claimed}
    envelope: JSONSchema = {
        "type": "object",
        "properties": {
            "name": name,
            "configuration": {"type": "object"},
            "must_understand": {"const": True},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    return [name, envelope]


def node_metadata_json_schema_v3(*, context: ZarrV3Context | None = None) -> JSONSchema:
    """The JSON Schema of a v3 `zarr.json` read in `context`: an array document or a group document, as `read_node_metadata_v3` reads it, except for the rules.

    Use it in an editor that validates a `zarr.json` while it is
    written, or in a validator written in another language. The schema
    is JSON Schema draft 2020-12, in the form `json_schema` writes. Each
    extension point is a field as the scope `context` reads it: either a
    name that a definition in scope claims, with the configuration that
    definition declares, or a name that no definition claims. When the
    document names a data type in scope, the fill value must have the
    JSON shape that the data type's definition declares (for `int8`, an
    integer in [-128, 127]). In a scope that claims it, as `CORE` does,
    a group's `consolidated_metadata` is the envelope the core spec
    fixes, with any JSON as the documents it lists, because the reader
    does not read them, and a `null` value, which zarr-python 3.0 and 3.1
    wrote, is refused, because the reader refuses it too. Each document
    is in `$defs` under the name of its TypedDict.
    `ZarrV3ArrayMetadataJSON` describes only an array.

    A JSON Schema describes each member on its own. It does not contain
    the rules that relate members to each other: one dimension name per
    dimension of the shape, a chunk grid that fits the shape, codecs in
    pipeline order with each codec checked against the chunk it
    receives, and the checks in a definition's `rules`. So a document
    the schema accepts may still have a problem. Every JSON document
    that `read_node_metadata_v3` reads without a problem is accepted
    by the schema. A validator reads JSON as a parser returns it, with
    arrays as lists. A model's `to_json` returns tuples, and a Python
    validator does not accept tuples as arrays.
    """
    context = scoped(context, CORE_AND_EXTENSIONS)
    schemas = Schemas(_documents(context))
    array = schemas.of(ZarrV3ArrayMetadataJSON)
    group = schemas.of(ZarrV3GroupMetadataJSON)
    return schemas.document({"anyOf": [array, group]})


def _documents(context: ZarrV3Context) -> SchemaLeaf:
    """The schema leaf for documents read in `context`: an array's fill value constrained by its data type, each field as `context` reads it, and each extra field that a definition in `context` claims."""
    fields = field_schemas(context)

    def leaf(annotation: object, schemas: Schemas) -> JSONSchema | None:
        if annotation is ZarrV3ArrayMetadataJSON:
            return schemas.defined(
                annotation, annotation.__name__, lambda: _array(context, schemas)
            )
        if annotation is ZarrV3GroupMetadataJSON:
            return schemas.defined(
                annotation,
                annotation.__name__,
                lambda: _with_extra_fields(
                    schemas.object_of(ZarrV3GroupMetadataJSON), context.group_extra_fields, schemas
                ),
            )
        return fields(annotation, schemas)

    return leaf


def _with_extra_fields(
    schema: JSONSchema, definitions: Mapping[str, ExtraFieldDefinition[Any]], schemas: Schemas
) -> JSONSchema:
    """`schema`, a node document's schema, with a property for each extra field in `definitions`: the member's value, as the configuration TypedDict of the definition that claims its key declares it."""
    if len(definitions) == 0:
        return schema
    properties = _member_object(schema, "properties")
    extra: dict[str, JSONValue] = {
        key: schemas.of(definition.configuration) for key, definition in definitions.items()
    }
    return {**schema, "properties": {**properties, **extra}}


def _member_object(schema: Mapping[str, JSONValue], key: str) -> Mapping[str, JSONValue]:
    """The object at member `key` of `schema`, or an empty one when the member is absent. Raises `TypeError` for a member that is not an object, which is a bug in the schema this module built."""
    found = schema.get(key, {})
    if not isinstance(found, Mapping):
        msg = f"the schema member {key!r} is an object, got {found!r}"
        raise TypeError(msg)
    return found


def _array(context: ZarrV3Context, schemas: Schemas) -> JSONSchema:
    """An array document: its TypedDict, at least one codec, and a constraint on its fill value for each data type in scope, which applies when the document names that data type.

    The TypedDict cannot declare a minimum length, so the schema adds it:
    a pipeline needs its array -> bytes codec, and the reader rejects an
    empty `codecs`.
    """
    schema = schemas.object_of(ZarrV3ArrayMetadataJSON)
    properties = _member_object(schema, "properties")
    codecs = _member_object(properties, "codecs")
    schema = {**schema, "properties": {**properties, "codecs": {**codecs, "minItems": 1}}}
    schema = _with_extra_fields(schema, context.array_extra_fields, schemas)
    held: list[JSONValue] = []
    for definition in context.data_types.values():
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


__all__ = [
    "field_json_schema_v3",
    "field_schemas",
    "node_metadata_json_schema_v3",
    "written_name",
]
