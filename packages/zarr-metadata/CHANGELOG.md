# Release notes

<!-- towncrier release notes start -->

## 0.2.0 (2026-05-19)

### Bugfixes

- `GzipCodecConfiguration.level` is now required, and `GzipCodecMetadata`
  no longer accepts the bare-string `"gzip"` form. The codec's compressed
  output depends on `level`, so metadata that omits it cannot reproducibly
  identify the chunk bytes produced by a writer. **Breaking** for consumers
  that previously typed gzip codec metadata as the bare string or
  constructed a `GzipCodecConfiguration` without `level`.
  ([#3978](https://github.com/zarr-developers/zarr-python/issues/3978))
- `BytesCodecObject.configuration` is now `NotRequired`. The configuration
  has no required keys (`endian` is conditionally required at runtime
  based on data type), so the object form may omit it entirely — matching
  the bare-string short-hand. **Soft-breaking** for consumers that
  previously relied on `configuration` always being present.
  ([#3978](https://github.com/zarr-developers/zarr-python/issues/3978))
- Better modelling of Zarr v2 stored metadata. Zarr v2 splits a node's
  metadata across two JSON documents (`.zarray`/`.zgroup` and `.zattrs`),
  but `GroupMetadataV2` had no `attributes` field while `ArrayMetadataV2`
  did — an inconsistency. `GroupMetadataV2` now also has an optional
  `attributes` field, and `ArrayMetadataV2.attributes` is now
  `NotRequired` for symmetry. **Soft-breaking** for consumers that
  relied on `ArrayMetadataV2.attributes` always being present.
  ([#3962](https://github.com/zarr-developers/zarr-python/issues/3962))

### Features

- Added `ArrayMetadataV3Partial`, `GroupMetadataV3Partial`,
  `ArrayMetadataV2Partial`, and `GroupMetadataV2Partial` — sibling
  TypedDicts to the existing full metadata types, declared with
  `total=False` so every field is `NotRequired`. Use these when typing
  dicts that intentionally hold a subset of a complete metadata document
  (test fixtures, fragment templates, in-progress builders). An
  equivalence test pins each `Partial` to the keys and value types of
  its full sibling so the two cannot drift.
  ([#3982](https://github.com/zarr-developers/zarr-python/issues/3982))
- Added three new top-level types modelling the **strict on-disk** shape
  of Zarr v2 metadata documents: `ZArrayMetadata` (the `.zarray` file),
  `ZGroupMetadata` (the `.zgroup` file), and `ZAttrsMetadata` (the
  `.zattrs` file). Use these when you want a type that faithfully matches
  what's stored on disk; use the merged `ArrayMetadataV2`/`GroupMetadataV2`
  when you want the in-memory representation a Python program typically
  works with.
  ([#3962](https://github.com/zarr-developers/zarr-python/issues/3962))
- Added typed constants exposing the spec-permitted values of constrained
  Literal fields, importable at the per-codec module level. For example,
  `from zarr_metadata.v3.codec.bytes import ENDIAN` provides
  `("little", "big")` as a tuple, enabling runtime iteration or validator
  generation without re-stating the Literal values by hand.
  ([#3978](https://github.com/zarr-developers/zarr-python/issues/3978))

## 0.1.1 (2026-05-06)

### Misc

- First usable release on PyPI. Version 0.1.0 was uploaded then deleted to
  reserve the project name; this version is the first one PyPI will install.
  No source changes from 0.1.0.
  ([#3949](https://github.com/zarr-developers/zarr-python/issues/3949))

## 0.1.0 (2026-05-01)

### Features

- Initial release. Provides `TypedDict` definitions and `Literal` aliases
  for the JSON shapes specified by Zarr v2 and v3 metadata, plus a subset
  of `zarr-extensions` types and the un-specified-but-widely-used
  consolidated metadata documents. Pair with a runtime validator like
  `pydantic` to check JSON loaded from disk.
  ([#3919](https://github.com/zarr-developers/zarr-python/issues/3919))
