# Pydantic

`zarr_metadata.pydantic` is an optional integration. It requires Pydantic
2.13 or newer, and the core package does not depend on it. Each exported
name is a Pydantic field type over one of the models: a raw document given
to the field goes through the same strict reader as `from_json`, and the
value of the field is the model class itself.

```python test="true" session="pydantic"
import json
from pydantic import BaseModel, TypeAdapter
import zarr_metadata.pydantic as zmp

with open("zarr.json", "rb") as f:
    raw = json.load(f)

adapter: TypeAdapter[zmp.ZarrV3ArrayMetadataField] = TypeAdapter(zmp.ZarrV3ArrayMetadataField)
metadata = adapter.validate_python(raw)
encoded = metadata.to_key_value()["zarr.json"]


class ArrayManifest(BaseModel):
    path: str
    metadata: zmp.ZarrV3ArrayMetadataField


manifest = ArrayManifest(path="a/b", metadata=raw)
```

A static type checker sees `manifest.metadata` as a
`zarr_metadata.model.ZarrV3ArrayMetadata`. A failed read raises a Pydantic
`ValidationError` with one line error per problem, in the format Pydantic
uses for its own errors.

## Choosing the context

A field type reads its document in the context given by Pydantic's
validation context:

- If the validation context is a `Context`, every field type of that
  format reads in it.
- If it is a mapping, the v3 field types read in its
  `"zarr_metadata_context"` item and the v2 field types in its
  `"zarr_metadata_context_v2"` item.
- If no context is given for a format, its field types read in that
  format's default, `CORE_AND_EXTENSIONS_CONTEXT` or `CORE_V2_CONTEXT`.

A field annotated `Annotated[ZarrV3ArrayMetadata, zmp.InScope(CONTEXT)]`
is read in `CONTEXT`, whatever the validation context holds, and a model
instance given to it is read again in `CONTEXT`.

## A bare `TypeAdapter` is not a validator

A `TypeAdapter` over one of the public `TypedDict`s is a **coercive**
shape adapter, not a Zarr conformance validator; it may coerce values or
discard fields that the strict reader rejects. It suits input built from
Python objects, such as user-authored dicts, better than JSON read from a
store. For JSON, use the field types above or the readers directly.
