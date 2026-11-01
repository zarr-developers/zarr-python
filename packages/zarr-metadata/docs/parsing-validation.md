# Parsing and Validation

## The problem statement

Suppose your program opens a Zarr array written by a stranger. You want to decode some of the chunks of
the array. For this to work, your program must transform its representation of the array metadata document from a raw blob of UTF-8-encoded bytes to a schema for an array and its chunks.

We can break this process down into stages, where each stage introduces more elaborate **context** for parsing
the metadata, and consequently introduces more elaborate problems that can be discovered. We will walk through
these stages here, using Zarr V3 array metadata as the example. Group metadata and Zarr V2 metadata are not covered here.

!!! note
    These stages are presented in a logical, narrative order. Even though all Zarr implementations have to validate JSON metadata, they won't necessarily use code that reflects the explicit stages defined here.

### JSON

At this stage we check that the document is valid JSON.

We read the `zarr.json` object from storage as plain bytes. We need to ensure that those bytes are UTF-8 text that encodes valid JSON.

This procedure yields one of two values:

1. A JSON value
2. A problem with the entire document: the bytes are not UTF-8, or they are not JSON

### JSON document shape

At this stage we check that the document is a JSON object with per-field structure consistent with rules in the Zarr V3 spec.

After confirming that the object named `zarr.json` is indeed valid JSON, we now need to check that the structure of this object is consistent with the rules in the Zarr V3 spec, namely the rules about the [structure of array metadata](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#array-metadata).

At this stage we need to ensure that:

- every **required** field is present
- every **required** and **optional** field has the correct shape

The correct shape varies per field. The Zarr V3 spec defines a **generic metadata element type** that is reused for data types, codecs, etc. Inhabitants of that type are:

- strings
- JSON objects with a `"name"` field and optionally `"configuration"` and `"must_understand"` fields

The `data_type` field must comply with this shape. The `codecs` field must be a non-empty JSON array of values that comply with this shape.

The spec does not define a shape for **extra** fields. Instead, the [relevant section of the spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#unknown) says that a reader must refuse to open an array with an extra field, unless that field is a JSON object with `"must_understand": false`. Like an unknown name, which we will see later, an extra field is not a problem with the document. It's a field that some readers can't use. So at this stage we don't check extra fields; we just note which of them a reader must understand.

!!! note
    At this stage we don't need to know **which** data type, or chunk grid, or codecs, are declared in the metadata document. We are only looking at **how** those fields are declared. We will resolve the raw JSON to a specific data type, chunk grid, etc. in a later validation pass.

This procedure yields one of two values:

1. A JSON object that adheres to the required shape
2. An enumeration of localized problems, like "`/data_type`: invalid type" or "`/codecs`: missing key", or "`/codecs/0`: invalid type". The latter example refers to the first element of the `codecs` array using [JSON pointer syntax](https://www.rfc-editor.org/info/rfc6901/). Localizing problems to a **position** in the document is important to ensure that an interested user or a repair routine can diagnose and fix that problem.

For a program that repairs broken `zarr.json` documents, the list of problems is a useful return value. It informs a repair procedure (if one is possible). But a program that ultimately decodes chunks of the array must only accept the shape-compliant JSON object, and subject it to the next stage of validation.

### Context-free consistency checks

At this stage we check if correlated fields in the document are consistent.

Some of the fields in the array metadata document are semantically coupled in a way that can be directly checked at the structural level. As of this writing, the only such fields are the `shape` field (a JSON array of non-negative integers), and the `dimension_names` field (an optional JSON array of strings or `null`s).

The length of the `shape` field must match the length of the `dimension_names` field, if that field is declared.

This procedure yields one of two values:

1. A Zarr-metadata-shaped JSON object with consistent `shape` and `dimension_names` fields
2. A problem localized to the `dimension_names` field. Strictly speaking, we can't say which of the two fields is wrong. But we treat `shape` as the reference for the number of axes in the array, so the problem goes on the field that disagrees with it. `shape` is required, and other fields like `chunk_grid` depend on it; `dimension_names` is optional, and only names the axes. This metadata has to be repaired; there's no other way to handle it.

### Resolving and validating each field

At this stage we check that each metadata object names a known type, given a registry of data types, codec types, etc., and that its configuration is valid for that type.

We take the JSON given for the `data_type`, elements of `codecs`, `chunk_grid`, `chunk_key_encoding`, and `storage_transformers` fields and attempt to **resolve** that JSON into a concrete data type, chunk grid, etc.

In the previous stages, we just had two elements to work with:

- the `zarr.json` metadata document
- the rules defined in the Zarr V3 spec.

This stage introduces a third element to our process: a registry of data types, codecs, chunk grids, etc. We "resolve" anonymous JSON metadata into a data type or codec by finding a type matching that JSON in the registry.

That registry is keyed by string identifiers, which almost all[^1] data types, codecs, etc. present. And the result of this stage is no longer context-free: it depends on the contents of the registry, i.e. which codec, data type, etc. definitions are in scope.

For example, the string `"float64"` should resolve to the 64-bit floating point data type with the same name defined in the [Zarr V3 spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/data-types/index.html#core-data-types).

We say "should" because that's what the Zarr V3 spec says about this. There is not a hard requirement that all implementations resolve the string `"float64"` to the 64-bit floating point data type. Someone could write their own (deviant) Zarr implementation that gives the name `"float64"` to a boolean data type. The mapping from JSON identifier (a string) to metadata type (a definition) is held by the registry that entered scope at this stage of validation.

For each field, we do two steps:

1. **Resolve** the name. We look up the name of the field in the registry.
2. **Validate** the configuration. If we found a match, we must then ensure that the configuration of the JSON value works for the type we resolved.

What if a name isn't in the registry? This is **not** a problem with the document. The document might be perfectly valid, and written by a program with a bigger registry than ours. An unknown name tells us something about our registry, not about the document. So we don't report it as a problem. Instead we mark the field as **unknown**, skip the second step, and leave it to the program that reads the document to decide what to do with it. A program that only needs to list the arrays in a hierarchy might be fine with an unknown codec. A program that decodes chunks is not, and it can try again with a more comprehensive registry.

If the name is in the registry, we validate the configuration. Suppose our metadata JSON had the following `chunk_grid` field: `"chunk_grid": {"name": "regular", "configuration": "error"}`. If we resolved the name of the chunk grid (`"regular"`) to the [regular chunk grid](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/chunk-grids/regular-grid/index.html) defined in the core spec, then we must do additional checks on the structure of the `configuration` field before we can interpret this chunk grid. In this case, the spec requires that the `configuration` field be a JSON object with a `chunk_shape` field, which is a JSON array of positive integers. Our `"configuration": "error"` field has the wrong type, so this is a problem localized to `/chunk_grid/configuration`.

Validating a configuration might require resolving more fields. Consider this example:

```json
{
    "codecs": [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [10, 10],
                "codecs": ["not_a_codec"],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
            }
        }
    ]
}
```

The name `"sharding_indexed"` should resolve to the [codec with the same name](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/sharding-indexed/index.html), and the definition of that codec states that its `codecs` field contains more codec metadata. These JSON array elements go through the same two steps, recursively: `"not_a_codec"` is not in the registry, so it is marked as unknown, not reported as a problem.

Repeating these steps, recursively if necessary, for all metadata objects, we yield one of two values:

1. A Zarr metadata object where each field is either unknown, or resolved to a type and internally consistent with the definition of that type
2. An enumeration of localized validation problems.

### Consistency checks between fields

At this stage we check that our resolved metadata is consistent across the entire metadata document.

Here we are in the space of interactions between metadata objects. It's a big space. One key invariant that cuts across many fields is dimensionality, i.e. the number of array axes. The `chunk_grid` and `shape` fields must agree on the dimensionality of the array. That is,

```json
{
    "shape": [1, 2, 3],
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}}
}
```

is a snippet of an invalid metadata document because the `shape` field applies to an array with three axes while the `chunk_grid` field applies to an array with just one. As with `dimension_names`, `shape` is the reference, so the problem is localized to `/chunk_grid/configuration/chunk_shape`. Dimensionality shows up in codecs too: the `order` of a [`transpose`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/transpose/index.html) codec must name each array axis exactly once.

The `fill_value` field is another example. The spec does not define a single shape for `fill_value`. Instead, each data type defines which JSON values it accepts as a fill value. So `"fill_value": "hello"` might be fine for a string data type, but it's a problem for `"float64"`. We can only check `fill_value` after we have resolved `data_type`.

The `codecs` field has the most interactions of all, because a list of codecs is a **pipeline**. The spec divides codecs into three kinds:

- array → array codecs, like `transpose`, which take an array and return an array
- array → bytes codecs, like `bytes`, which take an array and return bytes
- bytes → bytes codecs, like `gzip`, which take bytes and return bytes

A valid pipeline is zero or more array → array codecs, then exactly one array → bytes codec, then zero or more bytes → bytes codecs. We can't check this ordering until we know the kind of each codec, and we only know that after resolution.

Each codec in the pipeline is also handed a value produced by everything that came before it. The first codec is handed a chunk with the data type of the array and the shape of the chunk grid's chunks. An array → array codec may change that shape or data type before handing it on. Some codecs have rules about what they are handed. For example:

- The `bytes` codec requires an `endian` field in its configuration when the data type it is handed takes more than one byte per element. `{"name": "bytes"}` is valid for a `uint8` array, but a problem for a `float64` array.
- The `sharding_indexed` codec requires that its inner `chunk_shape` evenly divides the shape of the chunk it is handed, which is the outer chunk (or shard) shape.

To check these rules, we walk the pipeline in order, tracking the chunk that each codec is handed. For codecs that hold pipelines of their own, like `sharding_indexed`, we walk those inner pipelines too.

What about fields marked as unknown? We know nothing about them, and so we can't check anything that depends on them. If the data type is unknown, we can't check the fill value, or whether the `bytes` codec needs an `endian`. If a codec is unknown, we can't say what kind of codec it is, or what it hands on to the next codec. In these cases we skip the check rather than guess. An unknown field limits what we can say about the rest of the document.

This procedure yields one of two values:

1. A Zarr metadata object where every known field is valid on its own and consistent with every other known field. If no field is unknown, this is the metadata that a program can use to decode chunks. If some field is unknown, the program needs a registry that knows it before it can decode chunks.
2. An enumeration of localized consistency problems, e.g. "`/chunk_grid/configuration/chunk_shape`: expected one chunk length per dimension of `shape`".

## How `zarr-metadata` implements these stages

`zarr-metadata` does not run each stage as a separate function. Instead, a single read of the document collects the problems from every stage, so a caller gets the full list of problems at once.

The **registry** from the resolve-and-validate stage is called a **scope** in this library. For Zarr V3 it is a [`ZarrV3Context`](api/v3/definition.md) object, with one slot for each kind of field (`data_types`, `codecs`, `chunk_grids`, ...), and every reader takes a `context` argument. The default scope is `CORE_AND_EXTENSIONS_CONTEXT`, which holds the data types, codecs, chunk grids, etc. defined in the Zarr V3 spec and in [zarr-extensions](https://github.com/zarr-developers/zarr-extensions/). `CORE_CONTEXT` holds only the definitions from the Zarr V3 spec. `ZARR_PYTHON_CONTEXT` adds to `CORE_AND_EXTENSIONS_CONTEXT` the data types that zarr-python writes but no specification registers (`null_terminated_bytes` for NumPy's `S`, `raw_bytes` for NumPy's `V`, and `variable_length_bytes`), so a strict check of what zarr-python writes passes it.

`read_array_metadata_v3` reads a JSON document and returns a "reading". `reading.problems` is a tuple of the problems, and `reading.metadata` is the validated model, or `None` if there are problems. Each problem has a `loc`, which is its position in the document as a tuple, a `kind`, like `missing_key` or `invalid_value`, and a human-readable `message`:

```python test="true" session="parsing-validation"
from zarr_metadata.model import read_array_metadata_v3

raw = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [1, 2, 3],
    "data_type": "float64",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": "hello",
    "codecs": [{"name": "bytes"}],
}

reading = read_array_metadata_v3(raw)
for problem in reading.problems:
    print(problem.loc, problem.kind)
# ('fill_value',) invalid_value
# ('chunk_grid', 'configuration', 'chunk_shape') invalid_value
# ('codecs', 0, 'configuration', 'endian') missing_key
```

If you only want the model, `ZarrV3ArrayMetadata.from_json(raw)` returns it, or raises `MetadataValidationError` with every problem in `.problems`.

To read one field on its own, such as a codec, use `resolve` from `zarr_metadata.definition`. It takes the field, the kind of field, and the scope, and returns the field as read and its problems:

```python test="true" session="parsing-validation"
from zarr_metadata.definition import resolve
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CodecDefinition

field, problems = resolve(
    {"name": "gzip", "configuration": {"level": 12}}, CodecDefinition, CORE_AND_EXTENSIONS_CONTEXT
)
for problem in problems:
    print(problem.loc, problem.kind)
# ('configuration', 'level') invalid_value
```

A field with an unknown name is called **unclaimed**, because no definition in the scope claims that name. It is not reported as a problem: it is read as an `UnclaimedField`, and the checks that depend on it are skipped, as described above. So an empty `reading.problems` does not mean that every field is known. A program that decodes chunks should also check that no field is unclaimed. A model's `unclaimed_fields` lists each one, at any depth, by its location, and leaves out an extra member that says `"must_understand": false`, with everything in it, since a reader may ignore it. A reading with problems has no model, and a node with problems cannot be opened either:

```python test="true" session="parsing-validation"
unclaimed = None if reading.metadata is None else reading.metadata.unclaimed_fields
can_open = unclaimed is not None and len(unclaimed) == 0
```

To make an unknown data type or codec known, add its definition to the scope's slot and read the document again:

```python test="true" session="parsing-validation"
from zarr_metadata.definition import EmptyConfiguration
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CodecDefinition

# The definition of the codec that was unknown: here, a codec named
# `acme.codec` that takes no configuration.
MY_CODEC = CodecDefinition(
    name="acme.codec", configuration=EmptyConfiguration, kind="bytes_bytes", size="dynamic"
)
context = CORE_AND_EXTENSIONS_CONTEXT.update(codecs=[*CORE_AND_EXTENSIONS_CONTEXT.codecs.values(), MY_CODEC])
reading = read_array_metadata_v3(raw, context=context)
```

`update` replaces each slot it is given and keeps the others. A slot can also be given as a mapping from each definition's name to it, so `CORE_AND_EXTENSIONS_CONTEXT.update(codecs={**CORE_AND_EXTENSIONS_CONTEXT.codecs, "gzip": MY_GZIP})` replaces the gzip definition with your own.

[^1]: The one exception is the `r*` data type, which is identified by a templated string with no `configuration` object. This mixes the identifier with the configuration in the same JSON value, which is not good design. So this library normalizes `r*` into the same form as all the other metadata objects: the field `"r16"` keeps its name, and is read by the `r*` definition with the configuration `{"bits": 16}`.
