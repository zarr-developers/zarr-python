# JSON Schema

`zarr_metadata.json_schema` writes the rules the readers apply as JSON
Schema (draft 2020-12). Use it in two ways:

- in an editor that checks a `zarr.json` while you write it,
- with a validator in another language.

The module has three functions:

- `node_metadata_json_schema_v3(context=None)` writes the schema of a whole
  v3 `zarr.json`, array or group.
- `declaration_json_schema_v3(kind, *, context=None)` writes the schema of
  one declaration read as `kind`.
- `json_schema(typed_dict)` writes the schema of any of the package's
  `TypedDict`s, as `check` from `zarr_metadata.typed_json` reads it.

The schema describes each extension point as its context reads it:

- A claimed configuration follows its definition's TypedDict, including
  bounds.
- A name that nothing in the context claims may have any configuration.
- The fill value must be valid for the data type the document names.

## Limits

- A schema describes each member on its own. It does not describe the rules
  that relate members to each other. So a document the schema accepts may
  still have a problem.
- The reverse holds: every JSON document the readers accept, the schema
  also accepts.
- A schema validator reads JSON as a parser produces it, with arrays as
  lists. A model's `to_json` returns tuples, and a Python validator does
  not accept a tuple as an array.

```python test="true" session="json-schema"
import json
from zarr_metadata.json_schema import node_metadata_json_schema_v3

with open("zarr.schema.json", "w") as f:
    json.dump(node_metadata_json_schema_v3(), f, indent=2)
```
