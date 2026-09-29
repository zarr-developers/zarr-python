# zarr-metadata

Basic tools for modelling Zarr metadata, with minimal dependencies.

`zarr-metadata` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata)
and released independently of `zarr` itself. Install it with:

```
pip install zarr-metadata
```

## Who needs this

This library might be useful to you if your software interacts with Zarr metadata documents.

## What this is

This library is *not* a full Zarr implementation. Instead, it's a collection of data structures and routines that
closely model the content of the Zarr specifications, such as:

- **Typed JSON shapes** ([`zarr_metadata.v2`](api/v2.md) and
  [`zarr_metadata.v3`](api/v3/index.md)): `TypedDict` definitions and
  `Literal` aliases for the JSON documents specified by the
  [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html) and
  [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
  specifications, plus types for
  [zarr-extensions](https://github.com/zarr-developers/zarr-extensions/) and a
  few widely-used-but-unspecified entities (e.g. consolidated metadata).
- **Document models** ([`zarr_metadata.model`](api/model.md)): canonical
  frozen-dataclass models of whole metadata documents, with
  validators, loc-aware parsers, and store-key (de)serialization. A document
  produced by `to_json` shares no mutable state with the model that produced
  it.
- **Optional Pydantic integration** ([`zarr_metadata.pydantic`](api/pydantic.md),
  requires Pydantic 2.13 or newer): each model as a Pydantic field type that
  validates raw documents through the same strict parser.

## What this is for

The public `TypedDict` definitions describe the static JSON shape of Zarr
metadata. For strict, loc-aware validation of JSON loaded from disk, use the
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

metadata = TypeAdapter(zmp.ZarrV3ArrayMetadata).validate_python(raw)
encoded = metadata.to_key_value()["zarr.json"]
```

A bare `TypeAdapter` over a public document `TypedDict` is a coercive shape
adapter, not a Zarr conformance validator; it may coerce values or discard
members that the strict model parser rejects.

## Validation boundary

The model validators enforce the declared document structure and a small set
of context-free consistency rules, including fixed format literals, finite
JSON numbers outside `attributes`, non-negative dimensions, and one
`dimension_names` entry per array dimension. In a v3 document they also read
each extension point -- the data type, chunk grid, chunk key encoding, each
codec and each storage transformer -- through the definition that claims its
name in a scope, `CORE_AND_EXTENSIONS` unless a `context` is passed: a
configuration its definition refuses is refused, and a key it does not
declare is reported as `unknown_key`. A name nothing in the scope claims is
left unjudged, and whether to support it is the consumer's decision. A v3
fill value is judged against the data type it names, by that data type's
definition, the chunk grid against the shape, by the grid's definition,
and the codecs as a pipeline: in order, each judged by its definition
against the chunk it is handed, a shard's inner and index codecs too.
The validators do no arithmetic on values: whether a fill value survives
a `cast_value` round trip is not judged.

Two choices the specs' words leave open, or settle two ways:

- **`attributes` may hold `NaN`, `Infinity` and `-Infinity`.** The spec
  interprets no attribute, and zarr-python and xarray write those numbers
  there (a CF `_FillValue`, say). The models read them, and `to_key_value`
  writes them back as those bare tokens, which a strict JSON parser
  refuses. `check`, from `zarr_metadata.typed_json`, refuses a non-finite
  number wherever it is, attributes included.
- **`must_understand: false` is refused at every extension point**, codecs
  and storage transformers too, though the core spec names only the data
  type, chunk grid and chunk key encoding: a reader that skips a codec
  reads wrong bytes as surely as one that skips a data type reads wrong
  values. It keeps its meaning on an unknown top-level member, which a
  reader can skip.

A regular grid's chunk lengths are at least 1, along a dimension of
length 0 too: "Chunk sizes must be greater than zero", the regular grid
spec says. The core spec's "non-zero when the corresponding dimensions
of the arrays have non-zero length" says less, and allows nothing more,
so a document with a 0 there, as zarr-python 3.0 and 3.1 wrote for an
empty dimension, is refused.

`read_array_metadata_v3` reads a document once and returns everything
the read found: each field as the scope read it -- `Read` by the
definition that claims its name, `Unclaimed` when none does, or
`Refused` -- with where it sits and the kind it was read as, each codec
with the chunk it is handed, every problem, and the model when there is
none; `from_json` is that model, or the problems raised. A consumer's
own policy is a walk over the fields, with nothing read twice:
`with_problems` gives each with its problems, those located in it and in
the fields it holds, as zod's `flattenError` groups issues, and
`canonical_of` spells a field with none in the fewest words, without
reading it again. Which fields go beyond the core spec, say -- a field
that names nothing is a problem already -- and how each is spelled most
simply:

```python
from zarr_metadata.model import read_array_metadata_v3
from zarr_metadata.v3.definition import CORE, canonical_of, with_problems

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

`read_group_metadata_v3` reads a group the same way, and each document
its consolidated metadata holds once. `read_node_metadata_v3` reads a
`zarr.json` of either kind as the node its `node_type` says it is, as
a discriminated union reads its tag: a document that says neither reads
as `ZarrV3UnknownNodeReading`, with the problems, and nothing else of it
is read but its `zarr_format`, so a document of another format says it
is not v3. `node_metadata_from_json_v3` and `node_metadata_from_key_value_v3`
build the model of either kind, as the models' own `from_json` and
`from_key_value` build one.

Consolidated metadata holds the hierarchy below its group, the group its
root: the document of the node at `/a/b` sits at the key `a/b`, and the
documents and the group make a tree in which only groups hold nodes and
each node's parent is held. `NodeName` and `NodePath`, in
`zarr_metadata.v3`, are the strings the spec's rules for node names and
paths hold of, modelled on zarrs' types of those names, and
`validate_node_name_v3`, `is_node_name_v3` and `parse_node_name_v3`, and
their `node_path` twins, judge a string by them.

A member the spec does not define is not a field; the model's
`must_understand_fields` names those a reader must understand.

A model checks itself when it is built, as a pydantic model does in
`__init__`: one built by hand, or changed by `dataclasses.replace`, whose
document has a problem raises `MetadataValidationError` with every
problem at the change, so no model is built invalid, and `to_key_value`
writes each as it is. It holds its members as that read refines them, in
containers of its own -- a list given for an array as a tuple -- as
pydantic holds what its `__init__` coerced, and each field, a `Read` or
an `Unclaimed`, as the scope read it: one built by hand is taken as
read. A model a read builds is not read a second time. Change a model by
building another: a container it holds, changed in place, is not
checked again.

Two models are equal when they mean the same document, however each is
spelled. What the package interprets -- each field, and the fill value
against its data type -- compares by its canonical spelling, as
`canonical_of` and `canonical_fill_value` give it: `"NaN"` and
`"0x7fc00000"` are one `float32` fill value, `0.0` and `-0.0` two, and a
blosc with and without the `typesize` that `noshuffle` ignores one
codec. What it does not interpret -- attributes, extra fields, the
configuration of a field nothing in scope claims, and every member of a
v2 document -- compares as JSON text, which tells `true` from `1` and
`-0.0` from `0.0`, and takes `NaN` for itself. Equal models hash alike,
and may write two documents: `to_json` writes each as it was given. A
model's hash is of what its containers held when it was hashed, so a
model in a set, or a key of a dict, is not changed in place.

`node_metadata_json_schema_v3` writes what the validators read as a
JSON Schema, draft 2020-12, for an editor that checks a `zarr.json` as it
is written, or a validator in another language. Each extension point is
a field as its scope reads it: a configuration as its definition's
TypedDict says, bounds and all, and a name nothing in the scope claims
with any configuration. The fill value is what the data type it names
takes. `field_json_schema(kind, context)`, in
`zarr_metadata.v3.definition`, writes one field's schema, and
`json_schema`, in `zarr_metadata.typed_json`, any TypedDict's, as `check`
reads it. A schema says what each member is, and not what the rules say
of members together, so a document it accepts may still have a problem;
a JSON document the validators accept, it accepts. A validator reads
JSON as a parser gives it, arrays as lists: a model's `to_json` writes
tuples, which a Python validator does not take for arrays.

```python
import json
from zarr_metadata.model import node_metadata_json_schema_v3

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

- [API reference](api/index.md)
- [Release notes](release-notes.md)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-metadata/LICENSE.txt)
