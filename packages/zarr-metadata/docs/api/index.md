---
title: API reference
---

# API reference

The package is organized to mirror the structure of the Zarr specifications:

- [`zarr_metadata.model`](model.md) — frozen-dataclass document models,
  structural validators, loc-aware parsers, and the `UNSET` sentinel
- [`zarr_metadata.pydantic`](pydantic.md) — optional Pydantic field types
  over the models
- [`zarr_metadata.v2`](v2.md) — `TypedDict` shapes for Zarr v2 documents
  (`.zarray`, `.zgroup`, `.zattrs`, `.zmetadata`)
- [`zarr_metadata.v3`](v3/index.md) — `TypedDict` shapes for Zarr v3
  documents, with subpackages for [chunk grids](v3/chunk_grid.md),
  [chunk key encodings](v3/chunk_key_encoding.md), [codecs](v3/codec.md),
  and [data types](v3/data_type.md)

Every public name is also re-exported at the top level, so
`from zarr_metadata import ZarrV3ArrayMetadataJSON` and
`from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON` are equivalent.

## Common types

A few cross-cutting aliases are exported only from the top-level
`zarr_metadata` namespace:

::: zarr_metadata.JSONValue

::: zarr_metadata.ZarrV3NamedConfigJSON
