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
  frozen-dataclass models of whole metadata documents, with structural
  validators, loc-aware parsers, and store-key (de)serialization. A document
  produced by `to_json` shares no mutable state with the model that produced
  it.
- **Composition rules** ([`zarr_metadata.rules`](api/rules.md)): cross-field
  judgments over whole documents — fill value against data type, codec
  pipeline ordering, chunk geometry — with `validate_*` / `parse_*` entry
  points that apply structure and composition together.
- **Optional Pydantic integration** ([`zarr_metadata.pydantic`](api/pydantic.md),
  requires Pydantic 2.13 or newer): each model as a Pydantic field type that
  runs raw documents through the rules layer.

## What this is for

The public `TypedDict` definitions describe the static JSON shape of Zarr
metadata. To judge JSON loaded from disk, structure and composition together,
use the rules layer; to get a normalized document model, use the model parser:

```python
import json
from zarr_metadata.model import ZarrV3ArrayMetadata
from zarr_metadata.rules import parse_array_metadata_v3

with open("zarr.json", "rb") as f:
    raw = json.load(f)

document = parse_array_metadata_v3(raw)  # raises with every problem found
metadata = ZarrV3ArrayMetadata.from_json(document)
```

The optional Pydantic integration runs raw input through the rules layer
and returns the same normalized model class:

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
JSON numbers, non-negative dimensions, and non-empty v3 codec pipelines.
They do not interpret extension names or configurations.

The composition rules (`zarr_metadata.rules`) judge the document as a whole:
fill values against data types, codec pipeline ordering, chunk-grid
geometry against `shape`, one `dimension_names` entry per array dimension,
and the canonical configuration shapes of the codecs, chunk grids, chunk
key encodings, and data types this package defines. Unknown extension names
are left unjudged. The rules model canonical documents and are deliberately
stricter than any given implementation: an implementation may coerce
ambiguous input as it sees fit and then validate the canonical result.
Nothing here decides whether a data type, chunk grid, codec, or storage
transformer is *supported*; that belongs to consumer implementations.

An unmodelled member inside a *known* entity's `configuration` is an error
under that strict reading, in `parse_*` and in the Pydantic field types
alike: such a member is almost always a typo or a setting meant for a
different entity, and accepting it silently means silently ignoring what
the writer asked for. It carries its own `unknown_key` problem kind, so a
consumer who wants the tolerant reading can collect problems with
`validate_*` and filter that kind out.

## Scope

At minimum, this library supports what Zarr-Python needs: the complete
Zarr v2 and v3 specs, consolidated metadata, and a subset of the metadata
defined in `zarr-extensions`. We are generally open to contributions that
add types, models, or structural validation for Zarr metadata with a
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
