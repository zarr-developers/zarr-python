# Parsing and Validation

## The problem statement

Suppose your program opens a Zarr array written by a stranger. You want to decode some of the chunks of
the array. For this to work, your program must transform its representation of the array metadata document from a raw blob of UTF8-encoded bytes to a schema for an array and its chunks.

We can break this process down into stages, where each stage introduces more elaborate **context** for parsing
the metadata, and consequently introduces more elaborate problems that can be discovered. We will walk through
these stages here.

> [!NOTE]
> These stages are presented in a logical, narrative order. Even though all Zarr implementations have to validate JSON metadata, they won't necessarily use code that reflects the explicit stages defined here.

### JSON

At this stage we check that the document is valid JSON.

We read the `zarr.json` object from storage as plain bytes. We need to ensure that those bytes encode valid JSON. There are two outcomes: JSON decoding succeeds, or it fails, in which case the input very likely wasn't JSON.

### JSON document shape

At this stage we check that the document is a JSON object with per-field structure consistent with rules in the Zarr V3 spec.

After confirming that the object named `zarr.json` is indeed valid JSON (e.g., because `json.loads` didn't fail), we now need to check that the structure of this object is consistent with the rules in the Zarr V3 spec, namely the rules about the [structure of array metadata](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#array-metadata).

At this stage we need to ensure that:
- every **required** field is present
- every **required** and **optional** field has the correct shape
- if any **extra** fields are present, they have the correct shape, which is defined in the [relevant section of te spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#unknown).

The correct shape varies per field. The Zarr V3 spec defines a __generic metadata element type__ that is reused for data types, codecs, etc. Inhabitants of that type are:
- strings
- JSON objects with a "name" field and optionally "configuration" and "must_understand" fields

The `data_type` field must comply with this shape. The `codecs` field must be a non-empty JSON array of values that comply with this shape.

> [!NOTE]
> At this stage we don't need to know **which** data type, or chunk grid, or codecs, are declared in the metadata document. We are only looking at **how** those fields are declared. We will resolve the raw JSON to a specific data type, chunk grid, etc in a later validation pass.

This procedure yields one of two values:
1. A JSON object that adheres to the required shape
2. An enumeration of localized problems, like "`data_type`: wrong type" or "`codecs`: missing field", or "`codecs/0`": wrong type". The latter example refers to the first element of the `codecs` array using [JSON pointer syntax](https://www.rfc-editor.org/info/rfc6901/). Localizing problems to a __position__ in the document is important to ensure that an interested user or a repair routine can diagnose and fix that problem.

For a program that repairs broken `zarr.json` documents, the list of problems is a useful return value. It informs a repair procedure (if one is possible). But a program that ultimate decodes chunks of the array must only accept the shape-compliant JSON object, and subject it to the next stage of validation.

### Context-free consistency checks

At this stage we check if orrelated fields in the document are consistent.

Some of the fields in the array metadata document are semantically coupled in a way that can be directly checked at the structural level. As of this writing, the only such fields are the `shape` field (a JSON array of non-negative integers), and the `dimension_names` field (an optional JSON array of strings or `null`s).

The length of the `shape` field must match the length of the `dimension_names` field, if that field is declared.


This procedure yields one of two values:
1. A Zarr-metadata-shaped JSON object with consistent `shape` and `dimension_names` fields
2. A problem localized to the (`shape`, `dimension_names`) pair. We can't say one of the two is correct, so we localize the problem to them both. This metadata has to be repaired, there's no other way to handle it.


### Resolving JSON metadata

At this stage we check if each metadata object resolves to a known type, given a registry of data types, codec types, etc.

At this stage we take the JSON given for the `data_type`, elements of `codecs`, `chunk_grid`, `chunk_key_encoding`, and `storage_transformers` fields and attempt to __resolve__ that JSON into a concrete data type, chunk grid, etc.

In the previous stages, we just had two elements to work with:
- the `zarr.json` metadata document
- the rules defined in the Zarr V3 spec.

This stage introduces a third element to our process: a registry of data types, codecs, chunks grids, etc. We "resolve" anonymous JSON metadata into a data type or codec by finding a type matching that JSON in the registry.

That registry is keyed by string identifiers, which almost all[^1] data types, codecs, etc present. And the result of this stage is no longer context-free: it depends on the contents of the registry, i.e. which codec, data type, etc definitions are in-scope.

For example, the string `"float64"` should resolve to the 64-bit floating point data type with the same name defined in the [Zarr V3 spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/data-types/index.html#core-data-types).

We say "should" because that's what the Zarr V3 spec says about this. There is not a hard requirement that all implementations resolve the string `"float64"` to the 64-bit floating point data type. Someone could write their own (deviant) Zarr implementation that gives the name `"float64"` to a boolean data type. The mapping from JSON identifier (a string) to metadata type (a definition) is held by the registry that entered scope at this stage of validation.

This procedure yields one of two values:
1. A Zarr-metadata-shaped JSON object with per-field metadata resolved to concrete types.
2. An enumeration of localized resolution failures, e.g. "`data_type`: No data type found with name <name>." This can be handled by attempting to resolve again with a more comprehensive registry.

### Validating resolved JSON metadata

At this stage we check that each resolved metadata object is consistent with the type we resolved it to in the previous stage.

In the previous stage we used the identifier presented by a JSON value to match that JSON value to a type of metadata object. After finding a match, we must then ensure that the configuration of the JSON value works for the type we resolved.

Suppose our metadata JSON had the following `chunk_grid` field: `"chunk_grid": {"name": "regular", "configuration": "error"}`. If in the previous stage we resolved the name of the chunk grid ("regular") to the [regular chunk grid](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/chunk-grids/regular-grid/index.html) defined in the core spec, then we must do additional checks on the structure of the `configuration` field before we can interpret this chunk grid. In this case, the spec requires that the `configuration` field be a JSON object with a `chunk_shape` field, which is a JSON array of positive integers. Our `"configuration": "error"` field has the wrong type, so this is a problem localized to `chunk_grid/configuration`.

Although we did type resolution in the previous stage, we might have to do it again. Consider this example:
```json
codecs: [
    {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": [10, 10],
            "codecs": ["not_a_codec"]
            }
        }
    ]
```

The name `"sharding_indexed"` should resolve to the [codec with the same name](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/sharding-indexed/index.html), and the definition of that codec states that its `codecs` field contains more codec metadata. These JSON array elements must be resolved to codec types to complete the validation of the `configuration` field of the `sharding_indexed` codec. So a registry of types must be in scope for this stage.

Repeating validation, recursively if necessary, for all resolved metadata objects, we yield one of values:
- A Zarr metadata object where each field is internally consistent with the resolved type definition for that field
- An enumeration of localized validation failures (and potentially resolution failures, if validation required resolution).


### Consistency checks between resolved metadata

At this stage we check that our resolved metadata is consistent across the entire metadata document.

Here we are in the space of interactions between metadata objects. It's a big space. One key invariant that cuts across many fields is dimensionality, i.e. the number of array axes. The `chunk_grid` and `shape` fields must agree on the dimensionality of the array. That is,

```json
{
    "shape": [1,2,3],
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
}
```
is a snippet of an invalid metadata document because the `shape` field applies to an array with three axes while the `chunk_grid` field applies to an array with just one.

[^1] The one exception is the `r*` data type, which is identified by a templated string with no `configuration` object. This mixes the identifier with the configuration in the same JSON value, which is not good design. Thus we want to normalize `r*` into the same form as all the other metadata objects.