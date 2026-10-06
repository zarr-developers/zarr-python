---
title: API reference
---

# API reference

The package is organized to mirror the structure of the Zarr specifications:

- [`zarr_metadata.model`](model.md) — frozen-dataclass document models,
  validators, loc-aware parsers, a `zarr.json`'s JSON Schema, and the
  `UNSET` sentinel
- [`zarr_metadata.pydantic`](pydantic.md) — optional Pydantic field types
  over the models
- [`zarr_metadata.typed_json`](typed_json.md) — `check`, which type-checks
  a JSON value against any of the package's `TypedDict`s, read as the
  typing spec defines them, with every problem located, and
  `json_schema`, which writes what `check` reads as a JSON Schema
- [`zarr_metadata.v2`](v2.md) — `TypedDict` shapes for Zarr v2 documents
  (`.zarray`, `.zgroup`, `.zattrs`, `.zmetadata`), and
  `zarr_metadata.v2.definition`: the v2 dtypes and numcodecs codecs as
  definitions, `CORE_V2`, and the readers of one `dtype` or codec
- [`zarr_metadata.v3`](v3/index.md) — `TypedDict` shapes for Zarr v3
  documents, with subpackages for [chunk grids](v3/chunk_grid.md),
  [chunk key encodings](v3/chunk_key_encoding.md), [codecs](v3/codec.md),
  and [data types](v3/data_type.md)
- [`zarr_metadata.v3.definition`](v3/definition.md) — each extension's
  metadata as a definition: the TypedDict its configuration is, and the
  rules on it; check JSON against a TypedDict, judge a configuration,
  read a whole field in a scope, read a codec pipeline, or write a
  scope's fields as a JSON Schema. Its module docstring is the guide

The document types, models, and spec vocabulary — including the store keys —
are re-exported at the top level, so
`from zarr_metadata import ZarrV3ArrayMetadataJSON` and
`from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON` are equivalent.
The model layer's validators, parsers, type guards, and metadata key sets are
imported from [`zarr_metadata.model`](model.md) directly.

## Common types

A few cross-cutting aliases are exported from the top-level
`zarr_metadata` namespace (`JSONValue` from `zarr_metadata.typed_json`
and `zarr_metadata.v3.definition` too):

::: zarr_metadata.JSONValue

::: zarr_metadata.ZarrV3NamedConfigJSON
