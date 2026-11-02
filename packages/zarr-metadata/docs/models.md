# Models and readers

A model is a metadata document and the context it was read in. This page
covers how a document becomes a model, what a model exposes, how to change
one, and when two models are equal. The [Context](context.md) page explains
why the context is part of the model; this page is about using it.

## Readers and readings

Every document type has a reader in `zarr_metadata.model`. A reader does
one read and returns a **reading**: everything the read found, as values,
with nothing raised. `read_array_metadata_v3` returns:

- each declaration as the context read it: an `AcceptedDeclaration` when a
  definition claims its name, an `UnclaimedDeclaration` when none does, or
  a `RefusedDeclaration`, together with the declaration's location and the
  kind it was read as;
- each codec, with the chunk it receives;
- every problem;
- the model, when there are no problems.

The model's `from_json` returns that model, or raises
`MetadataValidationError` with the same problems. A consumer can apply its
own policy by walking the declarations, without reading anything twice.
`with_problems` pairs each declaration with its problems: those located in
the declaration and in the declarations it contains. This matches how zod's
`flattenError` groups issues. `canonical_of` returns the simplest spelling
of a declaration that has no problems, without reading it again. For
example, the code below finds the declarations that go beyond the core spec
(a declaration with no name is already a problem), and the simplest
spelling of each declaration:

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

`read_group_metadata_v3` reads a group the same way. It checks the envelope
of the group's `consolidated_metadata`, and does not read the documents the
member lists (see [Consolidated metadata](consolidated-metadata.md)).
`read_node_metadata_v3` reads a `zarr.json` of either kind as the node type
its [`node_type`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#node-type) names, the same way a discriminated union uses its tag. A
document whose `node_type` is neither reads as
`ZarrV3UnknownNodeMetadataReading`, with the problems. Only its
`zarr_format` is read, so a document of another format is reported as not
v3. `node_metadata_from_json_v3` and `node_metadata_from_key_value_v3`
build the model of either kind, the same way the models' own `from_json`
and `from_key_value` build one.

Every reader takes `context=None` for the default context,
`CORE_AND_EXTENSIONS_CONTEXT`.

## A model is a document read in a context

`ZarrV3ArrayMetadata(document, context=None)` reads the document in the
context (`CORE_AND_EXTENSIONS_CONTEXT` when none is given) and raises
`MetadataValidationError` with every problem, so an invalid model is never
built. `to_json` returns the document as it was written, and `to_key_value`
writes that same document to store bytes.

Every typed attribute is a view of that read: each declaration as the
context read it (an `AcceptedDeclaration` or an `UnclaimedDeclaration`),
and `shape`, `attributes` and the members that hold no declaration as the
read refined them. These views are read-only at every level: a list in the
document is shown as a tuple, and an object as a read-only mapping.
`to_json` returns plain containers. A model holds nothing that can be
changed in place.

To change a model, call `update`. It replaces members of the document with
the given JSON members and reads the result in the model's own context, so
you do not pass the context again. A member the spec defines is given by its
name, and any other field in `extra_fields`, a mapping from its key to its
value, or to `UNSET` to remove it. A name `update` does not take raises
`TypeError`, so a misspelled member is never written. `with_context` reads
the document in another context. `refined_in` also reads the document in
another context, but only one that claims what this context left unclaimed
and contradicts nothing; otherwise it raises `ScopeConflictError`.

## Extra fields

A top-level member that the spec does not define is an extra field. The
core spec hands such a member to its [extension rules](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#unknown). A
context can hold a definition for one, by its key, in the
`array_extra_fields` or `group_extra_fields` slot of `ZarrV3Context`. A
member whose key a definition claims is read by that definition: an extra
field is a declaration written as the member alone, `{key: value}`. Any
other extra field is kept as written, as an `UnclaimedDeclaration`.

The spec calls these members [additional fields](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#additional-fields),
and the names `extra_fields`, `ExtraFieldDefinition` and
`must_understand_fields` follow it. Everywhere else this package says
*member* for a name/value pair of a JSON object and *declaration* for the
value at an extension point. It keeps *field* for the fields of a Python
class, such as a dataclass or a pydantic model, and, after the spec and
NumPy, for the parts of a structured data type (`StructField`).

The context is what a reader recognizes, so the model's
`must_understand_fields` lists each unclaimed member that does not say
`"must_understand": false`: the spec [requires](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#must-understand) a reader to fail to open a
node with any such member. It covers extra fields only. A data type, chunk
grid, chunk key encoding, codec or storage transformer that nothing in the
context claims is an `UnclaimedDeclaration` too, also when it sits inside
another declaration, such as a codec inside a shard. A model's
`unclaimed_declarations` lists every declaration nothing in the context
claims, at any depth, by its location, and leaves out the members that waive
`must_understand`: a reader can open the node when it is empty.

## Equality

Two models are equal when they represent the same document, even if each
is spelled differently. Values the package interprets (each declaration,
and the fill value against its data type) are compared by their canonical
spelling, as returned by `canonical_of` and `canonical_fill_value`. For
example, `"NaN"` and `"0x7fc00000"` are the [same `float32` fill value](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/data-types/index.html#permitted-fill-values),
`0.0` and `-0.0` are two different fill values, and a v3 blosc codec under
`noshuffle`, which [ignores `typesize`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/blosc/index.html#configuration-parameters), is the same codec with or without
one.

A v2 document's [dtype](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding), codecs and fill value are compared the same way:
`<b1` and `|b1` are one dtype, in `dtype` and in a codec's dtype parameter
alike, and a v2 codec that leaves out a parameter is the same as one that
writes the numcodecs default, so `{"id": "zstd", "level": 0}` equals
`{"id": "zstd", "level": 0, "checksum": false}`. A v2 blosc `typesize` is
kept whatever the `shuffle`, unlike in v3: v2 codecs are defined by
numcodecs, which writes it into each chunk's header, so two v2 blosc codecs
that differ only in it encode different bytes, though they decode the same
data. The [v3 blosc spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/blosc/index.html#configuration-parameters) says `noshuffle` ignores `typesize`. A v2 node
with no `.zattrs` equals one with an empty `.zattrs`, as the
[v2 spec](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#attributes) reads a missing `.zattrs`,
as a v3 document with and without an empty `attributes` are equal.

Values the package does not interpret (attributes, extra fields, and the
configuration of a declaration that nothing in the context claims) are
compared as JSON text. JSON text distinguishes `true` from `1` and `-0.0`
from `0.0`, and treats `NaN` as equal to itself. Equal models have equal
hashes, but may write different documents, because `to_json` writes each
document as it was given.
