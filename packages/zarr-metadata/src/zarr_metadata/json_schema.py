"""JSON Schema output: a TypedDict, one Zarr v3 field, or a whole v3 `zarr.json`.

Each function writes JSON Schema draft 2020-12, for a validator in
another language or for an editor that checks a document while it is
written. A schema describes each member on its own, so it does not
contain the rules that relate members to each other, and a document it
accepts may still have a problem.

**A TypedDict.** `json_schema(typed_dict)` writes the rules that
`zarr_metadata.typed_json.check` applies. The schema accepts the
values that `check` finds no problems with. A TypedDict becomes an object with its keys, closed or open as the
TypedDict declares. A bound becomes the matching JSON Schema keyword:
`Interval(ge=0, le=9)` becomes `minimum` and `maximum`. Each TypedDict
or type alias is written once in `$defs`, under its name. JSON Schema
accepts a number with no fractional part, such as `1.0`, as an integer,
while `check` requires `1`:

    from zarr_metadata.json_schema import json_schema

    json_schema(GzipCodecConfiguration)
    # {'$schema': 'https://json-schema.org/draft/2020-12/schema',
    #  'type': 'object',
    #  'properties': {'level': {'type': 'integer', 'minimum': 0, 'maximum': 9}},
    #  'required': ['level'], 'additionalProperties': False}

**One v3 field.** `field_json_schema_v3(kind, context=None)` writes one field
read as `kind` in the scope `context`: a field that a definition in
scope reads, or a name that no definition claims, with any
configuration.

**A whole `zarr.json`.** `node_metadata_json_schema_v3(context=None)`
writes a whole v3 `zarr.json`, array or group, as the scope reads it
(`CORE_AND_EXTENSIONS` when none is given). Its fields are written
as the scope reads them. For each definition, the schema accepts its
field (its name, its configuration as its TypedDict declares it,
including bounds, a `must_understand` of `true`, and its bare name when
it needs no configuration). It also accepts a name that nothing in scope
claims, with any configuration. A nested field is written in the same
scope, and a member that accepts only codecs of static size accepts
only those. The rules are not in the schema, so a field it accepts may
still have a problem. Any field that `resolve` reads without a problem
is accepted by the schema, as JSON (arrays as lists, as a parser
returns them). Each configuration TypedDict and each field alias is
written once, in `$defs`, under its name. The fill value must match its
data type's fill value schema.
"""

from zarr_metadata._typed_json import JSONSchema, json_schema
from zarr_metadata.v3._json_schema import field_json_schema_v3, node_metadata_json_schema_v3

__all__ = [
    "JSONSchema",
    "field_json_schema_v3",
    "json_schema",
    "node_metadata_json_schema_v3",
]
