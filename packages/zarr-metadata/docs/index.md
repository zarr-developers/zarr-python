# zarr-metadata

Types, models and validators for Zarr metadata documents, with minimal
dependencies.

`zarr-metadata` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-metadata)
and released independently of `zarr`. Install it with:

```
pip install zarr-metadata
```

## What this is

A collection of data structures and routines that closely model the
metadata rules in the [Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)
and [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
specifications and in [zarr-extensions](https://github.com/zarr-developers/zarr-extensions/):

- **Typed JSON shapes**: `TypedDict` definitions for every document and
  configuration, for static type checking at the boundary of a Zarr
  application.
- **Document models**: immutable models of whole documents, with readers
  that report every problem at its location in the document.
- **Contexts**: a model is a document read in a **context**. A context
  is a collection of definitions for codecs, data types, chunk grids and so
  on. Not every Zarr implementation supports the same features. A context
  makes a program say exactly which features it reads.

This is *not* a full Zarr implementation. Nothing here reads or writes
chunks, encodes or decodes data, or performs store I/O.

Reading a `zarr.json` from disk looks like this:

```python test="true" session="index"
import json
from zarr_metadata import ZarrV3ArrayMetadata

with open("zarr.json", "rb") as f:
    raw = json.load(f)

metadata = ZarrV3ArrayMetadata.from_json(raw)
```

`from_json` returns the model, or raises `MetadataValidationError` with
every problem it found.

## Where to go next

- **You want type hints for Zarr JSON.** The `TypedDict`s are in
  [`zarr_metadata.v3`](api/v3/index.md) and [`zarr_metadata.v2`](api/v2.md).
  `check` in [`zarr_metadata.typed_json`](api/typed_json.md) turns any of
  them into a runtime check.
- **You validate metadata you did not write.** Start with
  [Parsing and validation](parsing-validation.md). It says what the readers
  check and why. Then read [Models and readers](models.md) for the API.
  [What the readers check](validation-rules.md) lists the rules and the
  decisions made where the spec is silent.
- **You support codecs or data types this package does not know.** Read
  [Context](context.md), then [Writing an extension](extensions.md).
- **You read stores written by older versions of zarr-python.** See
  [Repairing known writer bugs](repair.md).
- **You work with hierarchies.** See
  [Consolidated metadata](consolidated-metadata.md).
- **You build tools around the metadata.** [JSON Schema](json-schema.md)
  writes the readers' rules for an editor or another language.
  [Pydantic](pydantic.md) wraps each model as a field type.
- **You want the module-by-module view.** The [API reference](api/index.md)
  describes each module, and the module docstrings are the detailed guide.

## What the library covers

At minimum, this library supports what zarr-python needs:

- the complete Zarr v2 and v3 specs,
- consolidated metadata,
- a subset of the metadata defined in zarr-extensions.
 We are generally open to contributions that
add types, models, or validation for Zarr metadata with a published spec.

The models begin and end at the metadata documents themselves.
`from_key_value` and `to_key_value` map documents to store keys and bytes.
Everything past that belongs to consumer libraries. That includes resolving
a codec or data type name to an implementation.

This boundary limits how far some codec combinations can be validated. For example,
the readers cannot check that a `cast_value` codec maps scalars one to one.

## Reference

- [API reference](api/index.md)
- [Release notes](release-notes.md)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-metadata/LICENSE.txt)
