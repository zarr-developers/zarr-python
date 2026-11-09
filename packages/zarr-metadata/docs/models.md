# Models and readers

A model is a metadata document plus the context it was read in. This page
covers:

- how a document becomes a model,
- what a model exposes,
- how to change a model,
- when two models are equal.

The [Context](context.md) page explains why the context is part of the
model. This page is about using it.

## Readers and readings

Every document type has a reader in `zarr_metadata.model`. A reader does
one read and returns a **reading**. A reading holds everything the read
found, as values. Nothing is raised.

`read_array_metadata_v3` returns:

- Each declaration as the context read it, with its location and the kind
  it was read as. The declaration is one of:
    - an `AcceptedDeclaration`, when a definition claims its name;
    - an `UnclaimedDeclaration`, when no definition does;
    - a `RefusedDeclaration`, when a definition claims it but finds problems.
- Each codec, with the chunk it receives.
- Every problem.
- The model, when there are no problems.

The model's `from_json` returns that model, or raises
`MetadataValidationError` with the same problems.

A consumer can walk the declarations to apply its own policy. Nothing is
read twice. Two helpers support this:

- `with_problems` pairs each declaration with its problems. These are the
  problems located in the declaration and in the declarations it contains.
  This matches how zod's `flattenError` groups issues.
- `canonical_of` returns the simplest spelling of a declaration that has no
  problems. It does not read the declaration again.

The code below finds the declarations that go beyond the core spec. (A
declaration with no name is already a problem.) It also finds the simplest
spelling of each declaration.

```python test="true" session="models"
import json
from zarr_metadata.model import read_array_metadata_v3
from zarr_metadata.definition import canonical_of, with_problems
from zarr_metadata.v3.definition import CORE_CONTEXT

with open("zarr.json", "rb") as f:
    raw = json.load(f)

reading = read_array_metadata_v3(raw)
beyond_core = [
    loc
    for loc, declaration in reading.declarations()
    if declaration.name is not None and CORE_CONTEXT.claimant(declaration.read_as, declaration.name) is None
]
simplest = {
    loc: canonical_of(declaration, problems)
    for loc, declaration, problems in with_problems(reading.declarations(), reading.problems)
}
metadata = reading.metadata  # None when reading.problems is not empty
```

The other v3 readers work the same way:

- `read_group_metadata_v3` reads a group. It checks the envelope of the
  group's `consolidated_metadata`. It does not read the documents that
  member lists. See [Consolidated metadata](consolidated-metadata.md).
- `read_node_metadata_v3` reads a `zarr.json` of either kind. It picks the
  kind from the document's
  [`node_type`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#node-type),
  the way a discriminated union uses its tag. If `node_type` names neither
  kind, the result is a `ZarrV3UnknownNodeMetadataReading` with the
  problems. Only `zarr_format` is read in that case, so a document of
  another format is reported as not v3.
- `node_metadata_from_json_v3` and `node_metadata_from_key_value_v3` build
  the model of either kind. They work like the models' own `from_json` and
  `from_key_value`.

Every reader takes `context=None`. The default context is
`CORE_AND_EXTENSIONS_CONTEXT`.

## A model is a document read in a context

`ZarrV3ArrayMetadata(document, context=None)` reads the document in the
context. With no context, it uses `CORE_AND_EXTENSIONS_CONTEXT`. If the
read finds problems, it raises `MetadataValidationError` with all of them.
An invalid model is never built.

- `to_json` returns the document as it was written.
- `to_key_value` writes that same document to store bytes.

Every typed attribute is a view of the read:

- Each declaration appears as the context read it: an
  `AcceptedDeclaration` or an `UnclaimedDeclaration`.
- `shape`, `attributes` and the other members that hold no declaration
  appear as the read refined them.

These views are read-only at every level. A list in the document appears
as a tuple. An object appears as a read-only mapping. Nothing in a model
can be changed in place.

`to_json` returns a copy that the caller may change. Each object in it is
a new `dict`, and each array is a new tuple, because the package's JSON
types spell a JSON array as a tuple. `json.dumps` writes the copy, but it
does not compare equal to what `json.loads` returns for the same
document, where arrays are lists. To ask whether a model holds a
document, compare models: `ZarrV3ArrayMetadata.from_json(document) == model`.
This also treats a default that the document leaves out as equal to the
default written out.

To change a model, use one of these methods:

- `update` replaces members of the document and reads the result in the
  model's own context. You do not pass the context again.
    - Give a member the spec defines by its name.
    - Give any other member in `extra_fields`. This is a mapping from the
      member's key to its value, or to `UNSET` to remove it.
    - A name `update` does not take raises `TypeError`, so a misspelled
      member is never written.
- `with_context` reads the document in another context.
- `refined_in` also reads the document in another context. The new context
  must claim what this context left unclaimed and contradict nothing.
  Otherwise it raises `ScopeConflictError`.

## Writing store bytes

`to_key_value` writes a model's document as the JSON bytes a store holds.
A `JSONWriter` decides how those bytes are spelled: `indent`, `sort_keys`,
`ensure_ascii` and `trailing_newline`. Its defaults are those of
`json.dumps`. `to_key_value` takes a `JSONWriter` or a mapping of its
settings, so you do not need to import the class to change one:

```python test="true" session="models"
from zarr_metadata.model import JSONWriter, ZarrV3GroupMetadata

group = ZarrV3GroupMetadata.create_default(attributes={"b": 1, "a": 2})
print(group.to_key_value({"indent": 2, "sort_keys": True})["zarr.json"].decode())
# {
#   "attributes": {
#     "a": 2,
#     "b": 1
#   },
#   "node_type": "group",
#   "zarr_format": 3
# }
assert group.to_key_value({"indent": 2}) == group.to_key_value(JSONWriter(indent=2))
```

A mapping with a key that is not a setting raises `TypeError`, so a
misspelled setting is not ignored. `JSONWriter.from_dict` turns a mapping
into a writer, and `to_dict` turns a writer back into a mapping.

## Extra fields

An extra field is a top-level member that the spec does not define. The
core spec hands such members to its
[extension rules](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#unknown).

A context can hold a definition for an extra field, filed by its key. The
definition goes in the `array_extra_fields` or `group_extra_fields` slot of
`ZarrV3Context`.

- If a definition claims the key, it reads the member. The declaration is
  the member alone, `{key: value}`.
- Otherwise the member is kept as written, as an `UnclaimedDeclaration`.

The spec calls these members
[additional fields](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#additional-fields).
The names `extra_fields`, `ExtraFieldDefinition` and
`must_understand_fields` follow it. That heading comes from
[PR 373](https://github.com/zarr-developers/zarr-specs/pull/373). It
renamed the section that v3.1.0 calls
[Unknown](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#unknown).
As of v3.1.0, no tagged release has the new heading. So the link points at
the latest spec. Pin it once a release has the heading.

Elsewhere, this package uses these words:

- *member*: a name/value pair of a JSON object.
- *declaration*: the value at an extension point.
- *field*: a field of a Python class, such as a dataclass or a pydantic
  model. Following the spec and NumPy, it also means a part of a structured
  data type (`StructField`).

A model has two lists of what its context does not recognize:

- `must_understand_fields` lists each unclaimed extra field that does not
  say `"must_understand": false`. The spec
  [requires](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#must-understand)
  a reader to fail to open a node with any such member. This list covers
  extra fields only.
- `unclaimed_declarations` lists every declaration that nothing in the
  context claims, by its location. It covers every kind: data type, chunk
  grid, chunk key encoding, codec, storage transformer and extra field. It
  looks at any depth, such as a codec inside a shard. It leaves out the
  members that waive `must_understand`. When it is empty, a reader can open
  the node.

## Equality

Two models are equal when they represent the same document, even if they
spell it differently.

Values the package interprets are compared by their canonical spelling.
These are each declaration, and the fill value read against its data type.
`canonical_of` and `canonical_fill_value` return that spelling. For example:

- `"NaN"` and `"0x7fc00000"` are the
  [same `float32` fill value](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/data-types/index.html#permitted-fill-values).
- `0.0` and `-0.0` are two different fill values.
- A v3 blosc codec with `noshuffle`
  [ignores `typesize`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/blosc/index.html#configuration-parameters).
  So it is the same codec with or without one.

A v2 document's
[dtype](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding),
codecs and fill value are compared the same way:

- `<b1` and `|b1` are one dtype. This holds in `dtype` and in a codec's
  dtype parameter.
- A v2 codec that leaves out a parameter equals one that writes the
  numcodecs default. So `{"id": "zstd", "level": 0}` equals
  `{"id": "zstd", "level": 0, "checksum": false}`.
- A v2 blosc `typesize` is always kept, whatever the `shuffle`. This
  differs from v3. Numcodecs defines the v2 codecs, and it writes
  `typesize` into each chunk's header. So two v2 blosc codecs that differ
  only in `typesize` encode different bytes, though they decode the same
  data. In v3, the
  [blosc spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/blosc/index.html#configuration-parameters)
  says `noshuffle` ignores `typesize`.
- A v2 node with no `.zattrs` equals one with an empty `.zattrs`. The
  [v2 spec](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#attributes)
  says the key "does not have to be present, and if it is absent the
  attributes should be treated as empty".

In the same way, a v3 document with no `attributes` equals one whose
`attributes` is empty.

Values the package does not interpret are compared as JSON text. These
are:

- attributes,
- extra fields,
- the configuration of a declaration that nothing in the context claims.

JSON text tells `true` from `1`, and `-0.0` from `0.0`. It treats `NaN` as
equal to itself.

Equal models have equal hashes. They may still write different documents,
because `to_json` writes each document as it was given.
