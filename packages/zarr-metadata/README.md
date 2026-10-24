# zarr-metadata

Documentation: <https://zarr-metadata.readthedocs.io/>

Basic tools for modelling Zarr metadata, with minimal dependencies.

`zarr-metadata` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata)
and released independently of `zarr` itself. Install it with:

```
pip install zarr-metadata
```

## Who needs this

This library supports software that interacts with Zarr metadata documents.

## What this is

This is *not* a full Zarr implementation. It's a collection of data structures and routines that
closely model the content of the Zarr specifications, such as:

- **Typed JSON shapes** ([`zarr_metadata.v2`](https://zarr-metadata.readthedocs.io/en/latest/api/v2/) and
  [`zarr_metadata.v3`](https://zarr-metadata.readthedocs.io/en/latest/api/v3/)): `TypedDict` definitions and
  `Literal` aliases for the JSON documents specified by the
  [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html) and
  [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
  specifications, plus types for
  [zarr-extensions](https://github.com/zarr-developers/zarr-extensions/) and a
  few widely-used-but-unspecified entities (e.g. consolidated metadata).
- **Document models** ([`zarr_metadata.model`](https://zarr-metadata.readthedocs.io/en/latest/api/model/)): immutable
  models of whole metadata documents, each one a document and the scope
  it was read in, with validators, loc-aware parsers, and store-key
  (de)serialization. A document
  produced by `to_json` shares no mutable state with the model that produced
  it.
- **Definitions and scopes** ([`zarr_metadata.definition`](https://zarr-metadata.readthedocs.io/en/latest/api/definition/),
  with each format's kinds in `zarr_metadata.v3.definition` and
  `zarr_metadata.v2.definition`): each extension's configuration as a
  `TypedDict` with its rules, and the scope that decides which definitions
  read a document. `zarr_metadata.v3` and `zarr_metadata.v2` export every
  definition and scope of their format, such as `BYTES_CODEC` and
  `CORE_AND_EXTENSIONS`.
- **JSON type checking** ([`zarr_metadata.typed_json`](https://zarr-metadata.readthedocs.io/en/latest/api/typed_json/)):
  `check` checks a JSON value against any `TypedDict` and reports every
  problem with its location.
- **Problems** ([`zarr_metadata.problems`](https://zarr-metadata.readthedocs.io/en/latest/api/problems/)): the problem
  values every reader returns, and the error that carries them.
- **JSON Schema** ([`zarr_metadata.json_schema`](https://zarr-metadata.readthedocs.io/en/latest/api/json_schema/)): the
  JSON Schema of a `TypedDict`, a v3 field, or a whole `zarr.json`.
- **Optional Pydantic integration** ([`zarr_metadata.pydantic`](https://zarr-metadata.readthedocs.io/en/latest/api/pydantic/),
  requires Pydantic 2.13 or newer): each model as a Pydantic field type that
  validates raw documents through the same strict parser.

## What this is for

- **Static Type Checking** The public `TypedDict` definitions describe the static JSON shape of Zarr
  metadata.

- **Dynamic JSON validation** For strict, loc-aware validation of JSON loaded from disk, use the
  model parser:

  ```python
  import json
  from zarr_metadata.model import ZarrV3ArrayMetadata

  with open("zarr.json", "rb") as f:
      raw = json.load(f)

  metadata = ZarrV3ArrayMetadata.from_json(raw)
  ```

  The optional Pydantic integration delegates raw input to the same strict
  parser and returns the same normalized model class:

  ```python
  from pydantic import TypeAdapter
  import zarr_metadata.pydantic as zmp

  adapter: TypeAdapter[zmp.ZarrV3ArrayMetadataField] = TypeAdapter(zmp.ZarrV3ArrayMetadataField)
  metadata = adapter.validate_python(raw)
  encoded = metadata.to_key_value()["zarr.json"]
  ```

  Note that a bare `TypeAdapter` over a public `TypedDict` is a **coercive** shape
  adapter, not a Zarr conformance validator; it may coerce values or discard
  members that the strict model parser rejects. This is better suited for handling input provided
  from Python objects (e.g., user-authored dicts) than JSON.

## Validation boundary

The model validators enforce the declared document structure and a small set
of context-free consistency rules, e.g.:
- fixed format literals,
- finite JSON numbers outside `attributes` and the members nothing in scope reads
- non-negative dimensions
- one `dimension_names` entry per array dimension.

In a v3 document, validation also checks each extension point: the data
type, the chunk grid, the chunk key encoding, each codec, and each storage
transformer. Each one is checked by the definition that claims its name in
a scope. The scope is `CORE_AND_EXTENSIONS` unless a `context` is passed.
If the definition rejects a configuration, the validator rejects it too,
and a configuration key the definition does not declare is reported as
`unknown_key`. If no definition in the scope claims a name, the validator
does not check that extension point, and the consumer decides whether to
support it.

Some checks relate one part of a v3 document to another:

- The fill value is checked against the data type the document names, by
  that data type's definition.
- The chunk grid is checked against the shape, by the grid's definition.
- The codecs are checked as a pipeline. In order, each codec's definition
  checks the codec against the chunk it receives. This includes a shard's
  inner codecs and index codecs.

The validators do no arithmetic on values. For example, they do not check
whether a fill value survives a `cast_value` round trip.

Some choices that the spec text leaves open, or answers in two ways:

- **`attributes` may hold `NaN`, `Infinity` and `-Infinity`.** The spec
  does not interpret attributes, and zarr-python and xarray write these
  numbers there (for example, a CF `_FillValue`). The models read them,
  and `to_key_value` writes them back as bare tokens, which a strict JSON
  parser rejects. So may a member the spec does not define when no
  definition in scope claims it: nothing reads it, so it is kept as
  written. `check`, from `zarr_metadata.typed_json`, rejects a non-finite
  number anywhere, including in attributes.
- **A reader walks at most 256 levels of nesting.** A value nested deeper
  is an `invalid_value` problem at the first level past the limit,
  wherever it is. Every reader, writer and comparison uses one stack
  frame per level, and `copy.deepcopy` (and `pickle` before Python 3.12)
  uses two. A document at the limit therefore uses about half of the
  interpreter's default recursion limit, and the rest is left for the
  caller.
- **`consolidated_metadata: null` is a problem.** A zarr-python 3.0.x bug
  wrote it. The spec requires an object, and the package accepts only an
  object. A reader of those stores removes the key before reading.
- **An extension name must follow the spec's naming rule**,
  `^[a-z][a-z0-9-_.]+$`, or be a URI, which earlier versions of the spec
  required. Any other name is rejected before any definition is
  consulted, so `""` and `"foo/bar"` are problems, not unknown extensions.
- **`must_understand: false` is rejected at every extension point**,
  including codecs and storage transformers, although the core spec names
  only the data type, chunk grid and chunk key encoding. A reader that
  skips a codec reads wrong bytes, just as a reader that skips a data type
  reads wrong values. On an unknown top-level member, `must_understand`
  keeps its meaning: a reader can skip that member.
- **A v2 datetime or timedelta dtype must have a unit**, as in `<M8[ns]`,
  because the v2 spec requires the unit in brackets. zarr-python 3 writes
  `<M8` or `<m8` for a NumPy `datetime64` or `timedelta64` with no unit,
  and the package refuses those dtypes.

A regular grid's chunk lengths must be at least 1, even along a dimension
of length 0. The regular grid spec says "Chunk sizes must be greater than
zero". The core spec's wording, "non-zero when the corresponding dimensions
of the arrays have non-zero length", requires less, but does not explicitly
allow anything more. So a document with a chunk length of 0, which
zarr-python 3.0 and 3.1 wrote for an empty dimension, is rejected.

`read_array_metadata_v3` reads a document once and returns everything the
read found:

- each field as the scope read it: an `AcceptedField` when a definition
  claims its name, an `UnclaimedField` when none does, or a
  `RefusedField`, together with the field's location and the kind it was
  read as;
- each codec, with the chunk it receives;
- every problem;
- the model, when there are no problems.

`from_json` returns that model, or raises the problems. A consumer can
apply its own policy by walking the fields, without reading anything
twice. `with_problems` pairs each field with its problems: those located
in the field and in the fields it contains. This matches how zod's
`flattenError` groups issues. `canonical_of` returns the simplest
spelling of a field that has no problems, without reading it again. For
example, the code below finds the fields that go beyond the core spec (a
field with no name is already a problem), and the simplest spelling of
each field:

```python
from zarr_metadata.model import read_array_metadata_v3
from zarr_metadata.definition import canonical_of, with_problems
from zarr_metadata.v3.definition import CORE

reading = read_array_metadata_v3(raw)
beyond_core = [
    loc
    for loc, field in reading.fields()
    if field.name is not None and CORE.claimant(field.read_as, field.name) is None
]
simplest = {
    loc: canonical_of(field, problems)
    for loc, field, problems in with_problems(reading.fields(), reading.problems)
}
metadata = reading.metadata  # None when reading.problems is not empty
```

`read_group_metadata_v3` reads a group the same way. It checks the
envelope of the group's `consolidated_metadata`, and does not read the
documents the member lists (see below). `read_node_metadata_v3` reads
a `zarr.json` of either kind as the node type its `node_type` names, the
same way a discriminated union uses its tag. A document whose `node_type`
is neither reads as `ZarrV3UnknownNodeMetadataReading`, with the problems.
Only its `zarr_format` is read, so a document of another format is
reported as not v3. `node_metadata_from_json_v3` and
`node_metadata_from_key_value_v3` build the model of either kind, the same
way the models' own `from_json` and `from_key_value` build one.

Consolidated metadata holds the hierarchy below its group, with the group
as the root. The document of the node at `/a/b` is stored under the key
`a/b`. The documents and the group must form a tree: only groups have
child nodes, and the parent of every node is present.
`ZarrV3ConsolidatedMetadata` checks this when it reads the documents. `NodeName` and
`NodePath`, in `zarr_metadata.v3`, are the types of strings that satisfy
the spec's rules for node names and paths. They are modeled on the zarrs
types with the same names. `validate_node_name_v3`, `is_node_name_v3` and
`parse_node_name_v3`, and the matching `node_path` functions, check a
string against those rules.

A top-level member that the spec does not define is an extra field. A
scope can hold a definition for one, by its key, in the
`array_extra_fields` or `group_extra_fields` slot of `ZarrV3Context`. A
member whose key a definition claims is read by that definition, like any
other field. Any other member is kept as written, as an `UnclaimedField`.
The scope is what a reader recognizes, so the model's
`must_understand_fields` lists each unclaimed member that does not say
`"must_understand": false`: the spec requires a reader to fail to open a
node with any such member. It covers members only. A data type, chunk
grid, chunk key encoding, codec or storage transformer that nothing in
scope claims is an `UnclaimedField` too, also when it sits inside another
field, such as a codec inside a shard. `model.reading.fields()` yields
every field with its location, nested ones included, so a reader finds
each one it does not understand there. `CORE` holds
`CONSOLIDATED_METADATA_EXTRA_FIELD`, which reads a group's
`consolidated_metadata` as the envelope the spec fixes.

A v3 model consists of its document and the scope it was read in.
`ZarrV3ArrayMetadata(document, context=None)` reads the document in the
scope (`CORE_AND_EXTENSIONS` when none is given) and raises
`MetadataValidationError` with every problem, so an invalid model is never
built. `to_json` returns the document as it was written, and
`to_key_value` writes that same document to store bytes.

Every typed member is a view of that read: each field as the scope read
it (an `AcceptedField` or an `UnclaimedField`), and `shape`, `attributes`
and the other members as the read refined them. These views are read-only
at every level: a list in the document is shown as a tuple, and an object
as a read-only mapping. `to_json` returns plain containers.

To change a model, call `update`. It replaces members of the document with
the given JSON members and reads the result in the model's own scope, so
you do not pass the scope again. A member the spec defines is given by its
name, and any other member in `extra_fields`, a mapping from its key to its
value, or to `UNSET` to remove it. A name `update` does not take raises
`TypeError`, so a misspelled member is never written. `with_context` reads the document in
another scope. `refined_in` also reads the document in another scope, but
only one that claims what this scope left unclaimed and contradicts
nothing; otherwise it raises `ScopeConflictError`.

A group does not read the documents its `consolidated_metadata` lists.
Consolidated metadata is a convention of the reference implementation,
not core metadata, so a group holds the member as written, after it
checks the envelope that the core spec fixes. A reader can use the
member as a source of child documents in place of the store: read one
document with `read_node_metadata_v3` when that node is opened, or read
all of them with `ZarrV3ConsolidatedMetadata`, or with
`read_consolidated_metadata_v3`, which returns problems instead of
raising them. A document with a problem then fails only when it is read.
Pass `context=group.context` to read the documents in the group's own
scope; with no `context`, they are read in `CORE_AND_EXTENSIONS`.

To write consolidated metadata, build a `ZarrV3ConsolidatedMetadata` and
pass it to the group's `update`. The group reads the listed models in its
own scope, refuses one that its scope reads differently, and writes the
document. The model
reads every document strictly, and also accepts node models as entries.
Such an entry is accepted when its claims refine into the model's scope,
and refused at its path otherwise. `ZarrV3Context.joined` returns a scope in
which to consolidate children that were read in several scopes. Every
reader takes `context=None` for the default scope, `CORE_AND_EXTENSIONS`.

Two models are equal when they represent the same document, even if each
is spelled differently. Values the package interprets (each field, and the
fill value against its data type) are compared by their canonical
spelling, as returned by `canonical_of` and `canonical_fill_value`. For
example, `"NaN"` and `"0x7fc00000"` are the same `float32` fill value,
`0.0` and `-0.0` are two different fill values, and a blosc codec with and
without a `typesize` (which `noshuffle` ignores) is the same codec. A v2
document's dtype, codecs and fill value are compared the same way: `<b1`
and `|b1` are one dtype, and a v2 codec that leaves out a parameter is
the same as one that writes the numcodecs default, so
`{"id": "zstd", "level": 0}` equals
`{"id": "zstd", "level": 0, "checksum": false}`. Values the package does
not interpret (attributes, extra fields, and the configuration of a field
that nothing in the scope claims) are compared as JSON text. JSON text
distinguishes `true` from `1` and `-0.0` from `0.0`, and treats `NaN` as
equal to itself. Equal models have equal hashes, but may write different
documents, because `to_json` writes each document as it was given. A
model holds nothing that can be changed in place.

`node_metadata_json_schema_v3` writes the rules the validators apply as a
JSON Schema (draft 2020-12). Use it in an editor that checks a `zarr.json`
while you write it, or with a validator in another language. Each
extension point is described as a field as its scope reads it: a claimed
configuration follows its definition's TypedDict, including bounds, and a
name that nothing in the scope claims may have any configuration. The fill
value must be a valid fill value for the data type the document names.
`field_json_schema_v3(kind, *, context=None)` writes the schema of one field, and
`json_schema` writes the schema of any TypedDict as `check` reads it. All
three are in `zarr_metadata.json_schema`. A schema describes each member on its own, not the rules that
relate members to each other, so a document the schema accepts may still
have a problem. Every JSON document the validators accept, the schema also
accepts. A schema validator reads JSON as a parser produces it, with
arrays as lists. A model's `to_json` returns tuples, which a Python
validator does not accept as arrays.

```python
import json
from zarr_metadata.json_schema import node_metadata_json_schema_v3

with open("zarr.schema.json", "w") as f:
    json.dump(node_metadata_json_schema_v3(), f, indent=2)
```

## Scope

At minimum, this library supports what Zarr-Python needs: the complete
Zarr v2 and v3 specs, consolidated metadata, and a subset of the metadata
defined in `zarr-extensions`. We are generally open to contributions that
add types, models, or validation for Zarr metadata with a
published spec.

Runtime array behavior is out of scope: nothing here encodes or decodes
chunks, resolves codec or data type names to implementations, or performs
store I/O. The models begin and end at the metadata documents themselves —
`from_key_value` / `to_key_value` map documents to store keys and bytes,
and everything past that belongs to consumer libraries.

## Reference

- [API reference](https://zarr-metadata.readthedocs.io/en/latest/api/)
- [Release notes](https://zarr-metadata.readthedocs.io/en/latest/release-notes/)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-metadata/LICENSE.txt)
