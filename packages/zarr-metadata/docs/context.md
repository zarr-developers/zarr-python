# Context

Resolving metadata documents requires a context, a `Context`, that defines which metadata definitions apply to a JSON metadata document. This library models resolved metadata as a `(Document, Context)` pair, which might be a bit unintuitive. We explain these design decisions here.

## Why do we need a context

The Zarr V3 spec defines the JSON structure of `zarr.json` for arrays and groups exactly, by defining the fields that must be present and the JSON types they must have. But the __interpretation__ of those fields as actual data types, codecs, chunk grids, etc is not strictly defined by the spec. Relating a JSON value like `"bytes"` to the `bytes` codec requires a mapping from names to codec types.

`zarr-metadata` makes this explicit by validating JSON documents with a __context object__. That context object contains definitions of codecs, data types, etc, which can be matched to fields in the JSON document.

### Context example

We will walk through an example that shows how `zarr-metadata` uses context to parse the same metadata in different ways.

```python test="true" session="context"
from zarr_metadata import (
    MetadataValidationError,
    ZarrV3ArrayMetadata,
    ZarrV3ArrayMetadataJSON,
    ZarrV3MetadataFieldJSON,
)
from zarr_metadata.model import read_array_metadata_v3
from zarr_metadata.v3 import (
    BYTES_CODEC,
    GZIP_CODEC,
    DEFAULT_CHUNK_KEY_ENCODING,
    FLOAT64_DATA_TYPE,
    REGULAR_CHUNK_GRID,
    ZarrV3Context,
)

codecs: tuple[ZarrV3MetadataFieldJSON, ...] = (
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "gzip", "configuration": {"level": 99}},
    {"name": "unknown"},
)

document: ZarrV3ArrayMetadataJSON = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (100,),
    "data_type": "float64",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (10,)}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0,
    "codecs": codecs,
}
```

In the above code snippet we define a `zarr.json` document with three codec declarations (we will just call them codecs for brevity): one named `bytes`, another named `gzip`, and a third named `unknown`. The `bytes` codec here is correctly configured according to the `bytes` codec spec. The `gzip` declaration here violates the `gzip` codec spec because the `level` configuration field is too high. And the `unknown` codec is unknown, it has no specification and cannot be resolved.

These three codecs represent three different scenarios that this library handles, two with distinct failure modes. We demonstrate how this works in the next code snippet.

```python test="true" session="context"
# A context with one data type, one chunk grid, one chunk key encoding
# and one codec. There's no gzip definition, and no unknown codec definition.
context = ZarrV3Context(
    data_types=[FLOAT64_DATA_TYPE],
    chunk_grids=[REGULAR_CHUNK_GRID],
    chunk_key_encodings=[DEFAULT_CHUNK_KEY_ENCODING],
    codecs=[BYTES_CODEC],
)

# The gzip codec is unclaimed, and its configuration is not checked. The
# document is valid in this context.
small = ZarrV3ArrayMetadata(document, context=context)
print(small.codecs[1])
# UnclaimedField(json=mappingproxy({'name': 'gzip', 'configuration': mappingproxy({'level': 99})}), name='gzip', read_as=<class 'zarr_metadata.v3._definition.CodecDefinition'>, configuration=mappingproxy({'level': 99}))
```

In this example, our context object only contained one codec definition, for the `bytes` codec. But the array metadata document contained three codecs, named `"bytes"`, `"gzip"` and `"unknown"`.

After resolution the `gzip` codec was explicitly flagged as unrecognized. But we can actually check that codec declaration against a real codec definition by expanding our context to include the `GZIP_CODEC` definition:

```python test="true" session="context"
new_context = context.update(codecs={**context.codecs, "gzip": GZIP_CODEC})
```

`update` returns a new context with the `codecs` slot replaced. We pass the codecs the old context already has, plus `gzip`, so the new context claims both `bytes` and `gzip`. The old context is unchanged.

Now read the same document in the new context. `read_array_metadata_v3` returns everything the read found, instead of raising, so we can look at each codec:

```python test="true" session="context"
reading = read_array_metadata_v3(document, context=new_context)
for loc, field in reading.fields():
    if loc[0] == "codecs":
        print(loc, type(field).__name__)
# ('codecs', 0) AcceptedField
# ('codecs', 1) RefusedField
# ('codecs', 2) UnclaimedField

for problem in reading.problems:
    print(problem.loc, problem.kind, problem.message)
# ('codecs', 1, 'configuration', 'level') invalid_value expected an integer in [0, 9], got 99

print(reading.metadata)
# None
```

These are the three scenarios:

- `bytes` is parsed to an `AcceptedField`. A definition in the context claims the name `bytes`, and the configuration passes its checks.
- The malformed `gzip` is parsed to a `RefusedField`. A definition claims the name `gzip`, but the configuration fails one of its checks: a `level` of 99 is outside the range gzip allows. The read reports the problem, with its location in the document, and builds no model, so `reading.metadata` is `None`. Building the model directly raises the same problem as a `MetadataValidationError`.
- The `unknown` codec is parsed to an `UnclaimedField`. Nothing in the context claims the name, so it is kept as written and not checked. This is not a problem: the spec lets a document name codecs a reader does not know.

So one JSON document can be valid in one context and invalid in another. This is why a model is a document _and_ the context it was read in: the context defines the interpretation of the document. A model keeps its context, and moving it to another context reads the document again. Our `small` model was valid in its original context, but it is not valid in the new one:

```python test="true" session="context"
try:
    small.refined_in(new_context)
except MetadataValidationError as error:
    print([problem.loc for problem in error.problems])
# [('codecs', 1, 'configuration', 'level')]
```

An unclaimed field is never an error, but a reader that does not understand a codec cannot decode the array's chunks. `unclaimed_fields` lists every field that nothing in the context claims, so a reader can check that it understands everything before it opens the array. With a valid `gzip` level, the document reads in the new context, and only `unknown` is left:

```python test="true" session="context"
fixed_codecs = (codecs[0], {"name": "gzip", "configuration": {"level": 1}}, codecs[2])
fixed = ZarrV3ArrayMetadata({**document, "codecs": fixed_codecs}, context=new_context)
print(list(fixed.unclaimed_fields))
# [('codecs', 2)]

# In the small context, gzip is unclaimed too.
print(list(small.unclaimed_fields))
# [('codecs', 1), ('codecs', 2)]
```

A reader that can decode `bytes` and `gzip` but not `unknown` should refuse to open this array, even though its metadata is valid. Or at least, it's not __invalid__ in its context.
