# Parsing and Validation

## The problem statement

Suppose your program opens a Zarr array written by a stranger. You want to decode some of the chunks of
the array. For this to work, your program must transform the array metadata document. It starts as a raw blob of UTF-8-encoded bytes. It must end as a schema for an array and its chunks.

We can break this process down into stages. Each stage brings in more **context** for parsing the metadata. So each stage can discover more kinds of problems. We will walk through
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

At this stage we check that the document is a JSON object with per-member structure consistent with rules in the Zarr V3 spec.

We have confirmed that the object named `zarr.json` is valid JSON. Now we check that its structure follows the Zarr V3 spec's rules about the [structure of array metadata](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#array-metadata).

At this stage we need to ensure that:

- every **required** member is present
- every **required** and **optional** member has the correct shape

The correct shape varies per member. The Zarr V3 spec defines a [**generic metadata element type**](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#extension-definition) that is reused for data types, codecs, etc. Inhabitants of that type are:

- strings
- JSON objects with a `"name"` member and optionally `"configuration"` and `"must_understand"` members

The `data_type` member must comply with this shape. The `codecs` member must be a non-empty JSON array of values that comply with this shape.

The spec does not define a shape for **extra** fields. Instead, the [relevant section of the spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#unknown) says that a reader must refuse to open an array with an extra field. The exception is a field that is a JSON object with [`"must_understand": false`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#must-understand). Like an unknown name, which we will see later, an extra field is not a problem with the document. It's a field that some readers can't use. So at this stage we don't check extra fields; we just note which of them a reader must understand.

!!! note
    At this stage we don't need to know **which** data type, or chunk grid, or codecs, are declared in the metadata document. We are only looking at **how** those members are declared. We will resolve the raw JSON to a specific data type, chunk grid, etc. in a later validation pass.

This procedure yields one of two values:

1. A JSON object that adheres to the required shape
2. An enumeration of localized problems, like "`/data_type`: invalid type" or "`/codecs`: missing key", or "`/codecs/0`: invalid type". The latter example refers to the first element of the `codecs` array using [JSON pointer syntax](https://www.rfc-editor.org/info/rfc6901/). Each problem is localized to a **position** in the document. That lets a user or a repair routine diagnose and fix it.

For a program that repairs broken `zarr.json` documents, the list of problems is a useful return value. It informs a repair procedure (if one is possible). But a program that decodes chunks of the array must accept only the shape-compliant JSON object. It then passes that object to the next stage of validation.

### Context-free consistency checks

At this stage we check if correlated members in the document are consistent.

Some of the members in the array metadata document are semantically coupled in a way that can be directly checked at the structural level. As of this writing, there are only two such members:

- `shape`, a JSON array of non-negative integers;
- `dimension_names`, an optional JSON array of strings or `null`s.

The length of the `shape` member must match the length of the `dimension_names` member, if that member is declared.

This procedure yields one of two values:

1. A Zarr-metadata-shaped JSON object with consistent `shape` and `dimension_names` members
2. A problem localized to the `dimension_names` member. Strictly speaking, we can't say which of the two members is wrong. But we treat `shape` as the reference for the number of axes in the array. So the problem goes on the member that disagrees with it. `shape` is required, and other members like `chunk_grid` depend on it; `dimension_names` is optional, and only names the axes. This metadata has to be repaired; there's no other way to handle it.

### Resolving and validating each declaration

At this stage we use a registry of data types, codec types, etc. We check two things for each metadata object:

1. It names a type in the registry.
2. Its configuration is valid for that type.

We take the JSON given for these members:

- `data_type`,
- each element of `codecs`,
- `chunk_grid`,
- `chunk_key_encoding`,
- each element of `storage_transformers`.

We attempt to **resolve** that JSON into a concrete data type, chunk grid, etc.

In the previous stages, we just had two elements to work with:

- the `zarr.json` metadata document
- the rules defined in the Zarr V3 spec.

This stage introduces a third element to our process: a registry of data types, codecs, chunk grids, etc. We "resolve" anonymous JSON metadata into a data type or codec by finding a type matching that JSON in the registry.

That registry is keyed by string identifiers, which almost all[^1] data types, codecs, etc. present. The result of this stage is no longer context-free. It depends on the contents of the registry: which codec, data type, etc. definitions are in the context.

For example, the string `"float64"` should resolve to the 64-bit floating point data type with the same name defined in the [Zarr V3 spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/data-types/index.html#core-data-types).

We say "should" because that's what the Zarr V3 spec says about this. There is not a hard requirement that all implementations resolve the string `"float64"` to the 64-bit floating point data type. Someone could write their own (deviant) Zarr implementation that gives the name `"float64"` to a boolean data type. The registry holds the mapping from JSON identifier (a string) to metadata type (a definition). This stage of validation is where the registry comes in.

For each declaration, we do two steps:

1. **Resolve** the name. We look up the name of the declaration in the registry.
2. **Validate** the configuration. If we found a match, we must then ensure that the configuration of the JSON value works for the type we resolved.

What if a name isn't in the registry? This is **not** a problem with the document. The document might be perfectly valid, and written by a program with a bigger registry than ours. An unknown name tells us something about our registry, not about the document. So we don't report it as a problem. Instead we mark the declaration as **unknown** and skip the second step. The program that reads the document decides what to do with it. A program that only needs to list the arrays in a hierarchy might be fine with an unknown codec. A program that decodes chunks is not, and it can try again with a more comprehensive registry.

If the name is in the registry, we validate the configuration. Suppose our metadata JSON had the following `chunk_grid` member: `"chunk_grid": {"name": "regular", "configuration": "error"}`. Say we resolved the name `"regular"` to the [regular chunk grid](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/chunk-grids/regular-grid/index.html) defined in the core spec. Before we can interpret this chunk grid, we must check the structure of the `configuration` member. The spec requires `configuration` to be a JSON object with a `chunk_shape` member. `chunk_shape` must be a JSON array of positive integers. Our `"configuration": "error"` member has the wrong type, so this is a problem localized to `/chunk_grid/configuration`.

Validating a configuration might require resolving more declarations. Consider this example:

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

The name `"sharding_indexed"` should resolve to the [codec with the same name](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/sharding-indexed/index.html). That codec's definition says its `codecs` member contains more codec metadata. These JSON array elements go through the same two steps, recursively: `"not_a_codec"` is not in the registry, so it is marked as unknown, not reported as a problem.

Repeating these steps, recursively if necessary, for all metadata objects, we yield one of two values:

1. A Zarr metadata object where each declaration is either unknown, or resolved to a type and internally consistent with the definition of that type
2. An enumeration of localized validation problems.

### Consistency checks between members

At this stage we check that our resolved metadata is consistent across the entire metadata document.

Here we are in the space of interactions between metadata objects. It's a big space. One key invariant that cuts across many members is dimensionality, i.e. the number of array axes. The `chunk_grid` and `shape` members must agree on the dimensionality of the array. That is,

```json
{
    "shape": [1, 2, 3],
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}}
}
```

is a snippet of an invalid metadata document. The `shape` member describes an array with three axes. The `chunk_grid` member describes an array with just one. As with `dimension_names`, `shape` is the reference, so the problem is localized to `/chunk_grid/configuration/chunk_shape`. Dimensionality shows up in codecs too: the `order` of a [`transpose`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/transpose/index.html) codec must name each array axis exactly once.

The `fill_value` member is another example. The spec does not define a single shape for `fill_value`. Instead, each data type defines which JSON values it accepts as a fill value. So `"fill_value": "hello"` might be fine for a string data type, but it's a problem for `"float64"`. We can only check `fill_value` after we have resolved `data_type`.

The `codecs` member has the most interactions of all, because a list of codecs is a **pipeline**. [The spec](https://github.com/zarr-developers/zarr-specs/blob/v3.1.0/docs/v3/core/index.rst#L931-L1024) divides codecs into three kinds:

- array → array codecs, like `transpose`, which take an array and return an array
- array → bytes codecs, like `bytes`, which take an array and return bytes
- bytes → bytes codecs, like `gzip`, which take bytes and return bytes

A valid pipeline has, in order:

1. zero or more array → array codecs,
2. exactly one array → bytes codec,
3. zero or more bytes → bytes codecs.

We can't check this ordering until we know the kind of each codec, and we only know that after resolution.

Each codec in the pipeline is also handed a value produced by everything that came before it. The first codec is handed a chunk with the data type of the array and the shape of the chunk grid's chunks. An array → array codec may change that shape or data type before handing it on. Some codecs have rules about what they are handed. For example:

- The [`bytes`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/bytes/index.html#configuration-parameters) codec requires an `endian` member in its configuration when the data type it is handed takes more than one byte per element. `{"name": "bytes"}` is valid for a `uint8` array, but a problem for a `float64` array.
- The [`sharding_indexed`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/sharding-indexed/index.html#configuration-parameters) codec requires that its inner `chunk_shape` evenly divides the shape of the chunk it is handed. That chunk is the outer chunk, or shard.

To check these rules, we walk the pipeline in order, tracking the chunk that each codec is handed. For codecs that hold pipelines of their own, like `sharding_indexed`, we walk those inner pipelines too.

What about declarations marked as unknown? We know nothing about them, and so we can't check anything that depends on them. If the data type is unknown, we can't check the fill value, or whether the `bytes` codec needs an `endian`. If a codec is unknown, we can't say what kind of codec it is, or what it hands on to the next codec. In these cases we skip the check rather than guess. An unknown declaration limits what we can say about the rest of the document.

This procedure yields one of two values:

1. A Zarr metadata object where every known declaration is valid on its own and consistent with every other known declaration. If no declaration is unknown, this is the metadata that a program can use to decode chunks. If some declaration is unknown, the program needs a registry that knows it before it can decode chunks.
2. An enumeration of localized consistency problems, e.g. "`/chunk_grid/configuration/chunk_shape`: expected one chunk length per dimension of `shape`".

## How `zarr-metadata` implements these stages

`zarr-metadata` does not run each stage as a separate function. Instead, a single read of the document collects the problems from every stage, so a caller gets the full list of problems at once.

The **registry** from the resolve-and-validate stage is called a **context** in this library. For Zarr V3 it is a [`ZarrV3Context`](api/v3/definition.md) object. It has one slot for each kind of declaration (`data_types`, `codecs`, `chunk_grids`, ...). Every reader takes a `context` argument. The default context is `CORE_AND_EXTENSIONS_CONTEXT`, which holds the data types, codecs, chunk grids, etc. defined in the Zarr V3 spec and in [zarr-extensions](https://github.com/zarr-developers/zarr-extensions/). `CORE_CONTEXT` holds only the definitions from the Zarr V3 spec. `ZARR_PYTHON_CONTEXT` adds the data types that zarr-python writes but no specification registers.

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

To read one declaration on its own, such as a codec, use `resolve` from `zarr_metadata.definition`. It takes the declaration, the kind of declaration, and the context, and returns the declaration as read and its problems:

```python test="true" session="parsing-validation"
from zarr_metadata.definition import resolve
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CodecDefinition

declaration, problems = resolve(
    {"name": "gzip", "configuration": {"level": 12}}, CodecDefinition, CORE_AND_EXTENSIONS_CONTEXT
)
for problem in problems:
    print(problem.loc, problem.kind)
# ('configuration', 'level') invalid_value
```

A declaration with an unknown name is called **unclaimed**, because no definition in the context claims that name. It is not reported as a problem: it is read as an `UnclaimedDeclaration`, and the checks that depend on it are skipped, as described above. So an empty `reading.problems` does not mean that every declaration is known. A program that decodes chunks should also check that no declaration is unclaimed. A model's `unclaimed_declarations` lists each one, at any depth, by its location. It leaves out an extra field that says `"must_understand": false`, with everything in it, since a reader may ignore it.

To report each of these as a problem too, read the document with `read_openable_node_metadata_v3`. Its problems are:

1. the problems of `read_node_metadata_v3`,
2. then one problem of kind `unclaimed_declaration` for each declaration that a reader must understand and that nothing in the context claims.

So its reading has no problems exactly when a reader that understands the context can open the node:

```python test="true" session="parsing-validation"
from zarr_metadata.model import read_openable_node_metadata_v3

openable = read_openable_node_metadata_v3(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": 0,
        "codecs": [{"name": "bytes"}, {"name": "my_codec"}],
        "my_member": {"name": "my_member"},
        "ignorable": {"must_understand": False},
    }
)
for problem in openable.problems:
    print(problem.loc, problem.kind)
# ('codecs', 1) unclaimed_declaration
# ('my_member',) unclaimed_declaration
```

To make an unknown data type or codec known, add its definition to the context's slot and read the document again:

```python test="true" session="parsing-validation"
from zarr_metadata.definition import EmptyConfiguration
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS_CONTEXT, CodecDefinition, ZarrV3Context

# The definition of the codec that was unknown: here, a codec named
# `acme.codec` that takes no configuration.
MY_CODEC = CodecDefinition(
    name="acme.codec", configuration=EmptyConfiguration, kind="bytes_bytes", size="dynamic"
)
context = CORE_AND_EXTENSIONS_CONTEXT.union(ZarrV3Context(codecs=[MY_CODEC]))
reading = read_array_metadata_v3(raw, context=context)
```

`union` keeps every definition the context has, and raises `ScopeConflictError` if `MY_CODEC` has a name the context already files with another definition. To replace a definition on purpose, use `copy`, which, like pydantic's `model_copy(update=...)`, replaces whole each slot it is given and keeps the others. A slot can be given as a mapping from each definition's name to it, so `CORE_AND_EXTENSIONS_CONTEXT.copy(update={"codecs": {**CORE_AND_EXTENSIONS_CONTEXT.codecs, "gzip": MY_GZIP}})` replaces the gzip definition with your own. Passing `copy(update={"codecs": [MY_CODEC]})` instead would leave `MY_CODEC` as the only codec.

A data type's definition also says what its fill value may be. `fill_value` is the fill value's JSON shape, as a type annotation. `fill_value_rules` is a function of the configuration, the nested declarations and the fill value, that yields each problem the shape cannot express. For example, `Base64Bytes` is a string, so an annotation accepts any string, and the rule `base64_fill_value_rules` rejects one that is not base64. A rule of your own can yield from it, then check what depends on the configuration:

```python test="true" session="parsing-validation"
import base64

from typing_extensions import TypedDict

from zarr_metadata.definition import fill_value_problems
from zarr_metadata.problems import ValidationProblem
from zarr_metadata.v3.data_type.bytes import Base64Bytes, base64_fill_value_rules
from zarr_metadata.v3.definition import DataTypeDefinition


class PackedConfiguration(TypedDict, closed=True):
    length_bytes: int


def packed_fill_value_rules(configuration, nested, value):
    # The fill value is base64, of at most `length_bytes` bytes.
    problems = list(base64_fill_value_rules(configuration, nested, value))
    yield from problems
    if not problems and len(base64.b64decode(value)) > configuration["length_bytes"]:
        yield ValidationProblem((), "expected fewer bytes than the type holds", "invalid_value")


PACKED = DataTypeDefinition(
    name="acme.packed",
    configuration=PackedConfiguration,
    fill_value=Base64Bytes,
    fill_value_rules=packed_fill_value_rules,
)
context = CORE_AND_EXTENSIONS_CONTEXT.union(ZarrV3Context(data_types=[PACKED]))
data_type, _ = resolve(
    {"name": "acme.packed", "configuration": {"length_bytes": 2}}, DataTypeDefinition, context
)
for fill_value in ("YWI=", "not base64!", "YWJj"):
    print(fill_value, [problem.message for problem in fill_value_problems(data_type, fill_value)])
# YWI= []
# not base64! ['expected standard-alphabet base64, got "not base64!"']
# YWJj ['expected fewer bytes than the type holds']
```

A reader of a document in this context checks its fill value with these rules, and so does a pydantic field. A JSON Schema written for the context does not contain the rules, so it accepts any string here.

[^1]: The one exception is the [`r*` data type](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#data-types), which is identified by a templated string with no `configuration` object. This mixes the identifier with the configuration in the same JSON value, which is not good design. So this library normalizes `r*` into the same form as all the other metadata objects. The declaration `"r16"` keeps its name. The `r*` definition reads it with the configuration `{"bits": 16}`.
