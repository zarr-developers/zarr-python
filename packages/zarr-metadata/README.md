# zarr-metadata

Python types, models, and validators for Zarr v2 and v3 metadata.

Documentation: <https://zarr-metadata.readthedocs.io/>

## What this is

Two layers and an optional integration:

- **Typed JSON shapes**: `TypedDict` definitions and `Literal` aliases for the
  JSON documents specified by the [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)
  and [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
  specifications, plus types for [`zarr-extensions`](https://github.com/zarr-developers/zarr-extensions/)
  and a few widely-used-but-unspecified entities (e.g. consolidated metadata).
- **Document models** (`zarr_metadata.model`): each model is a metadata
  document and the scope it was read in, with validators, loc-aware
  readers, and store-key (de)serialization. A document produced by `to_json`
  shares no mutable state with the model that produced it.
- **Optional Pydantic integration** (`zarr_metadata.pydantic`, requires
  Pydantic 2.13 or newer): each model as a Pydantic field type that validates
  raw documents through the same strict parser.

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

adapter: TypeAdapter[zmp.ZarrV3ArrayMetadata] = TypeAdapter(zmp.ZarrV3ArrayMetadata)
metadata = adapter.validate_python(raw)
encoded = metadata.to_key_value()["zarr.json"]
```

A bare `TypeAdapter` over a public document `TypedDict` is a coercive shape
adapter, not a Zarr conformance validator; it may coerce values or discard
members that the strict model parser rejects.

## Dependencies

The core of this package depends on `typing-extensions` and
`annotated-types` only. It does not depend on a validation framework,
and it will not: a dependency on pydantic, or any other framework with
its own release cadence and compiled parts, would pin that framework for
every consumer of zarr-metadata and collide with the pins consumers
already carry. Instead the package reads a `TypedDict` as the typing spec
defines it with its own checker (`zarr_metadata.typed_json.check`), and
spells bounds in the `annotated-types` vocabulary, which pydantic reads
too. `zarr_metadata.pydantic` is an optional integration over the models;
nothing in the core imports it.

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
- **A reader walks 256 levels of nesting.** A value nested deeper is an
  `invalid_value` at the level past the last, wherever it sits. Every
  reader, writer and comparison takes one frame for each level, and
  `copy.deepcopy`, and `pickle` before Python 3.12, two: a document at
  the cap takes about half of the interpreter's default limit, and the
  rest is the caller's.
- **`consolidated_metadata: null` is a problem.** zarr-python 3.0 and 3.1
  wrote it on a group they had not consolidated; the spec says an object,
  and the package models nothing else as right.
  `read_repaired_node_metadata_v3` removes it before reading.
- **An extension is named as the spec names one**, `^[a-z][a-z0-9-_.]+$`,
  or by a URI, which earlier versions of the spec required; any other
  name is refused before a definition is asked, so `""` and `"foo/bar"`
  are not unknown extensions but problems.
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
the read found: each field as the scope read it -- `AcceptedField` by the
definition that claims its name, `UnclaimedField` when none does, or
`RefusedField` -- with where it sits and the kind it was read as, each codec
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
paths hold of, modeled on zarrs' types of those names, and
`validate_node_name_v3`, `is_node_name_v3` and `parse_node_name_v3`, and
their `node_path` twins, judge a string by them.

A member the spec does not define is not a field; the model's
`must_understand_fields` names those a reader must understand.

A v3 model is its document and the scope it was read in:
`ZarrV3ArrayMetadata(document, context=None)` reads the document in the
scope, `CORE_AND_EXTENSIONS` when none is given, and raises
`MetadataValidationError` with every problem, so no model is built
invalid; `to_json` writes the document as it was written, and
`to_key_value` writes it as it is. Every typed member is a view of that
read: each field as the scope read it, an `AcceptedField` or an `UnclaimedField`, and
`shape`, `attributes` and the rest as the read refined them, read-only
at every level: a list given for an array as a tuple, an object as a
read-only mapping; `to_json` gives plain containers. A model is changed
by `update`, which puts
JSON members in place of the document's and reads the result in the
model's own scope, so no scope is passed back in; `with_context` reads
the document in another scope, and `refined_in` only in one that claims
what this one left unclaimed and contradicts nothing, raising
`ScopeConflictError` otherwise. The documents a group's
`consolidated_metadata` holds are models of the group's scope, built from
the group's one read; it takes node models as entries too, each accepted
when its claims refine into the group's scope and refused at its path
otherwise, and `Context.joined` is the scope to consolidate children of
several scopes in. Every reader takes `context=None` for the default
scope, `CORE_AND_EXTENSIONS`.

Two models are equal when they mean the same document, however each is
spelled. What the package interprets -- each field, and the fill value
against its data type -- compares by its canonical spelling, as
`canonical_of` and `canonical_fill_value` give it: `"NaN"` and
`"0x7fc00000"` are one `float32` fill value, `0.0` and `-0.0` two, and a
blosc with and without the `typesize` that `noshuffle` ignores one
codec, and a v2 `dtype` by its family and size, `<b1` and `|b1` one
dtype. What it does not interpret -- attributes, extra fields, and the
configuration of a field nothing in scope claims -- compares as JSON
text, which tells `true` from `1` and `-0.0` from `0.0`, and takes `NaN`
for itself. Equal models hash alike, and may write two documents:
`to_json` writes each as it was given. A model holds nothing that can be
changed in place: what it hands out is read-only at every level.

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

The Pydantic integration's field types have JSON Schemas of their own,
for a model that holds them: an extension point there is a name and any
configuration, read in no scope, and v2 documents have one too. For a
`zarr.json`, use `node_metadata_json_schema_v3`. Neither replaces the
validators: JSON Schema takes a number such as `1.0` for an integer,
where the models require an `int`, and says nothing of what members read
together say, such as `dimension_names` against `shape` or v2 `chunks`
against `shape`. Run the model parser after schema validation.

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

## Developing

Package-scoped development commands live in the [`justfile`](./justfile)
(requires [just](https://github.com/casey/just)):

```
just test        # run the test suite (extra args go to pytest)
just lint        # ruff, same invocation as CI
just typecheck   # pyright, pinned to the version CI uses
just docs-check  # strict build of the docs site
just check       # all of the above
just docs-serve  # serve the docs site locally
```

Run them from this directory, or from anywhere in the repository as
`just packages/zarr-metadata/<recipe>`.

## Releasing

The package version is derived from git tags by `hatch-vcs`. Tags must
match the pattern `zarr_metadata-v<version>` (e.g. `zarr_metadata-v0.2.0`)
so they do not collide with the main `zarr-python` release tags.

To cut a release:

1. Create and push a tag of the form `zarr_metadata-v<version>` on the
   commit you want to publish, e.g.:
   ```
   git tag zarr_metadata-v0.2.0 <commit>
   git push origin zarr_metadata-v0.2.0
   ```
2. Pushing the tag fires the `zarr-metadata release` workflow, which
   builds the wheel/sdist (version resolved from the tag), runs an
   install smoke test, and publishes to PyPI via OIDC trusted publishing.

We intentionally do *not* create a GitHub Release for `zarr-metadata`
versions — GitHub Releases live at the repo level, and a zarr-metadata
release would surface in the zarr-python repo's Releases UI as if it
were a zarr-python release.

To dry-run a build against TestPyPI, dispatch the workflow manually
(`Actions` → `zarr-metadata release` → `Run workflow`). Manual dispatches
build from the current commit; with no recent tag the version will look
like `0.1.devN`, which is fine for TestPyPI.

## License

[MIT](./LICENSE.txt)
