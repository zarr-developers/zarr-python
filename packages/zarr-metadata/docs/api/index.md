---
title: API reference
---

# API reference

The package is organized to mirror the structure of the Zarr specifications:

- [`zarr_metadata.json_schema`](json_schema.md) — JSON Schema output for
  any `TypedDict`, one v3 field, or a whole v3 `zarr.json`
- [`zarr_metadata.model`](model.md) — document models (each one a
  document and the scope it was read in), readers, validators,
  loc-aware parsers, and the `UNSET` sentinel
- [`zarr_metadata.problems`](problems.md) — `ValidationProblem`,
  `ProblemKind`, `Problems` and `MetadataValidationError`, which every
  reader reports, and `shown`, which formats a value for a message
- [`zarr_metadata.pydantic`](pydantic.md) — optional Pydantic field types
  over the models
- [`zarr_metadata.typed_json`](typed_json.md) — `check`, which type-checks
  a JSON value against any of the package's `TypedDict`s, read as the
  typing spec defines them, with every problem located
- [`zarr_metadata.v2`](v2.md) — `TypedDict` shapes for Zarr v2 documents
  (`.zarray`, `.zgroup`, `.zattrs`, `.zmetadata`), and
  `zarr_metadata.v2.definition`: the v2 kinds of metadata, the scope
  `CORE_V2`, and the readers of one `dtype` or codec. Each dtype family's
  definition is in `zarr_metadata.v2.data_type`, and each codec's in
  `zarr_metadata.v2.codec`
- [`zarr_metadata.v3`](v3/index.md) — `TypedDict` shapes for Zarr v3
  documents, with subpackages for [chunk grids](v3/chunk_grid.md),
  [chunk key encodings](v3/chunk_key_encoding.md), [codecs](v3/codec.md),
  and [data types](v3/data_type.md)
- [`zarr_metadata.definition`](definition.md) — what both formats share
  for reading a field: `Definition`, `resolve`, the three results of a
  reading (`AcceptedField`, `UnclaimedField`, `RefusedField`), and the
  scope type `Context`. Its module docstring is the guide
- [`zarr_metadata.v3.definition`](v3/definition.md) — the five v3 kinds
  of metadata (`CodecDefinition`, `DataTypeDefinition`, ...), the codec
  pipeline types, and the v3 scopes `CORE` and `CORE_AND_EXTENSIONS`

The document types, models, and spec vocabulary — including the store keys —
are re-exported at the top level, so
`from zarr_metadata import ZarrV3ArrayMetadataJSON` and
`from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON` are equivalent.
The model layer's validators, parsers and type guards are imported from
[`zarr_metadata.model`](model.md) directly, and the problem types every
reader reports from [`zarr_metadata.problems`](problems.md).

## Common types

Two cross-cutting types are also exported from the top-level
`zarr_metadata` namespace: `JSONValue`, documented in
[`zarr_metadata.typed_json`](typed_json.md), and `ZarrV3NamedConfigJSON`,
documented in [`zarr_metadata.v3`](v3/index.md).
