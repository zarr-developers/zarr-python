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

Three choices the specs' words leave open, or settle two ways:

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
- **A chunk length of 0 is allowed along a dimension of length 0.** The
  core spec asks for non-zero chunk lengths only "when the corresponding
  dimensions of the arrays have non-zero length"; the regular grid spec
  says chunk sizes are greater than zero. The package follows the core
  spec, which zarr-python 3.0 and 3.1 wrote for an empty dimension.

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
as `ZarrV3UnknownNodeReading`, with the problem, and nothing else of it
is read. `node_metadata_from_json_v3` and `node_metadata_from_key_value_v3`
build the model of either kind, as the models' own `from_json` and
`from_key_value` build one.

A member the spec does not define is not a field; the model's
`must_understand_fields` names those a reader must understand.

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
