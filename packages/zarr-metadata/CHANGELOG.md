# Release notes

<!-- towncrier release notes start -->

## 0.4.0 (2026-07-29)

### Features

- Added `zarr_metadata.model`: frozen-dataclass models (`ZarrV2ArrayMetadata`,
  `ZarrV3ArrayMetadata`, `ZarrV2GroupMetadata`, `ZarrV3GroupMetadata`,
  `ZarrV2ConsolidatedMetadata`, `ZarrV3ConsolidatedMetadata`, `ZarrV3NamedConfig`)
  that are canonical, semantically lossless representations of Zarr metadata
  documents, plus structural validators (`validate_*` / `is_*` / `parse_*`).
  Every v3 extension point (data type, chunk grid, chunk key encoding, codecs,
  storage transformers) is held as `ZarrV3NamedConfig`: a name, configuration,
  and `must_understand` obligation; nothing is interpreted. On the wire, an
  empty configuration with the default obligation uses the spec's plain-string
  shorthand. Model fields are annotated with the role alias
  `ZarrV3MetadataField` (today exactly `ZarrV3NamedConfig`), so annotations
  convey the logical meaning and stay put if the spec adds another field form.

  Validation is strict about what the types declare: v2 `dtype` / `order` /
  `compressor` / `filters` / `dimension_separator` shapes and the fixed
  `zarr_format` / `node_type` literals are all enforced. Every
  `ValidationProblem` carries a machine-readable `kind`
  (`missing_key` / `invalid_type` / `invalid_value` / `invalid_json`) so
  consumers can dispatch on the failure mode without matching message strings,
  and every ingestion failure — including missing store keys and undecodable
  bytes in `from_key_value` — surfaces as `MetadataValidationError`. An
  adversarial review added further structural checks: JSON booleans are not
  accepted as dimension lengths, dimensions are non-negative,
  `dimension_names` must have one entry per dimension of `shape`, `attributes`
  and `configuration` values are JSON-checked recursively (like `fill_value`),
  non-finite floats and non-standard JSON constants are rejected, abstract
  mappings and sequences normalize to encoder-safe canonical containers,
  v2 `shape` and `chunks` must have the same rank, non-null v2 filter pipelines
  contain at least one filter, document `TypeIs` guards only narrow values that
  already use the declared canonical containers,
  and the inline consolidated-metadata envelope and entries are deep-validated
  so the group validator's verdict always agrees with the model constructor.

  The v3 models expose `must_understand_fields`: the subset of `extra_fields`
  not explicitly waived with `must_understand: false` (fields are implicitly
  must-understand per the spec). Readers discharge the spec's fail-to-open
  duty by subtracting the extension names they recognize; the model only
  partitions by obligation, since recognition is reader-specific.

  Optional pydantic integration ships as `zarr_metadata.pydantic` (importing it
  requires pydantic 2.13 or newer; the core package does not depend on it): one
  `Annotated`
  field type per model, validating raw documents through `from_json`, passing
  core-model instances through unchanged, serializing via `to_json`, and
  publishing JSON Schemas derived from private constrained document types that
  mirror the independently expressible runtime rules. Cross-field cardinality
  relations still require runtime validation. The instances are the core model
  classes, so values interoperate freely with non-pydantic code.

  `create_default` keeps its output self-consistent: overriding `shape` without
  a chunk grid derives one regular chunk covering the array (v3
  `chunk_shape == shape`; v2 `chunks == shape`) instead of silently keeping the
  scalar default's 0-d grid.

  A v2 `.zarray` that omits `dimension_separator` is interpreted with the v2
  convention's default `"."` (the model previously normalized absence to `"/"`,
  which would misaddress the chunks of real-world default-separator arrays).
  The value is never null: absent, `"."`, or `"/"` are the only spellings.

  Optional document keys use `UNSET` — a PEP 661 sentinel
  (`typing_extensions.Sentinel`), usable directly in type expressions — never
  `None`: in a model, `None` always corresponds to a JSON `null` in the
  document (a v2 `compressor`, an unnamed dimension inside `dimension_names`),
  and `UNSET` always means the key is absent. Checker note: ty types the
  sentinel exactly; pyright needs `<= 1.1.404` until microsoft/pyright#11115
  is fixed (this package's CI pins it); mypy users need a `cast` or
  `type: ignore` at narrowing sites until python/mypy#21647 merges. This keeps semantically distinct spellings
  distinct — an absent `dimension_names` ("there are no dimension names") and
  an explicit `[null, null]` ("every dimension has a name, which is null") are
  different documents and round-trip as such. The `consolidated_metadata: null`
  written by a historical zarr-python bug is the one deliberate exception to
  faithful round-tripping: those stores remain readable, but the bug spelling
  is repaired to absence on read and never written back.

  The v2 models treat the `.zattrs` file's presence as part of the store:
  `attributes` is `UNSET` when no `.zattrs` file exists (and `to_key_value`
  emits none), while an explicit empty `.zattrs` is `{}` and round-trips as a
  file. Previously `to_key_value` always emitted `.zattrs`, silently adding a
  file to stores that never had one.

  The store-key `Literal` aliases (`ZarrV2ArrayMetadataStoreKey`,
  `ZarrV2AttributesStoreKey`, ...) are exported from `zarr_metadata.model`
  alongside their constants, and each `to_key_value` return type is keyed by
  them, so the set of store keys a model can emit is visible in its signature.
  `from_key_value` deliberately keeps `Mapping[str, bytes]` input: it accepts
  any string-keyed store mapping and ignores unrelated keys.

  `to_json` returns a document that shares no mutable state with the model:
  every value that can hold a mutable container (attributes, configurations,
  extra fields, v2 codec configurations, fill values, consolidated entries) is
  deep-copied on the way out, so editing a serialized document can never
  silently mutate the frozen model that produced it. ([#4119](https://github.com/zarr-developers/zarr-python/issues/4119))

### Improved Documentation

- `zarr-metadata` now has a standalone documentation site at
  <https://zarr-metadata.readthedocs.io/>, with a comprehensive API reference
  covering every public module, versioned by this package's release tags. The
  package also gained a `justfile` collecting its development commands
  (`test`, `lint`, `typecheck`, `docs-check`, `docs-serve`, `changelog-draft`),
  which the package CI workflow now delegates to. ([#4208](https://github.com/zarr-developers/zarr-python/issues/4208))

### Deprecations and Removals

- The document (TypedDict) types are renamed to put the format version at the
  front of the name and to mark the JSON-document form with a `JSON` suffix,
  so a format version can never be misread as a class revision and the bare
  entity names are reserved for the `zarr_metadata.model` dataclasses:

  - `ArrayMetadataV2` → `ZarrV2ArrayMetadataJSON` (and `...Partial` accordingly)
  - `ArrayMetadataV3` → `ZarrV3ArrayMetadataJSON` (and `...Partial` accordingly)
  - `GroupMetadataV2` → `ZarrV2GroupMetadataJSON` (and `...Partial` accordingly)
  - `GroupMetadataV3` → `ZarrV3GroupMetadataJSON` (and `...Partial` accordingly)
  - `ConsolidatedMetadataV2` → `ZarrV2ConsolidatedMetadataJSON`
  - `ConsolidatedMetadataV3` → `ZarrV3ConsolidatedMetadataJSON`
  - `NamedConfigV3` → `ZarrV3NamedConfigJSON`
  - `MetadataV3` → `ZarrV3MetadataFieldJSON` (the union of the bare-name and
    named-configuration spellings of one metadata field)
  - `ExtensionFieldV3` → `ZarrV3ExtensionField`
  - `CodecMetadataV2` → `ZarrV2CodecMetadata`
  - `DataTypeMetadataV2` → `ZarrV2DataTypeMetadata`
  - `ArrayOrderV2` → `ZarrV2ArrayOrder`
  - `ArrayDimensionSeparatorV2` → `ZarrV2ArrayDimensionSeparator`
  - `ZArrayMetadata` → `ZarrV2ZArrayJSON` (the strict on-disk `.zarray` document)
  - `ZGroupMetadata` → `ZarrV2ZGroupJSON` (the strict on-disk `.zgroup` document)
  - `ZAttrsMetadata` → `ZarrV2ZAttrsJSON` (the `.zattrs` document)

  The old names are removed, not aliased. The `zarr_metadata.pydantic` field
  types take the bare entity names (`ZarrV3ArrayMetadata`, ...), matching the
  model classes they validate into.

  The conventions, stated once for future additions: CamelCase type names put
  the format version first (`ZarrV2ArrayMetadataJSON`,
  `ZarrV3ArrayMetadataStoreKey`), while SCREAMING_SNAKE constants and
  snake_case functions put it last (`ARRAY_METADATA_STORE_KEY_V2`,
  `validate_array_metadata_v3`). The `JSON` suffix marks a raw-document type
  whose bare name is taken by (or reserved for) a `zarr_metadata.model`
  dataclass; raw field-level types the models hold verbatim
  (`ZarrV2CodecMetadata`, `ZarrV3ExtensionField`) keep their bare names.
  Extension-entity types put the registered entity name first and end in
  exactly one role suffix (`BloscCodecMetadata`, `Uint8DataTypeName`) — the
  `V2` in `V2ChunkKeyEncodingMetadata` is that encoding's entity name, not a
  format version, which is always spelled `ZarrV2`/`ZarrV3`. Every public
  type name is checked against this grammar by
  `tests/test_public_api.py::test_public_type_names_comply_with_naming_grammar`.

  ([#4119](https://github.com/zarr-developers/zarr-python/issues/4119))


## 0.3.0 (2026-06-19)

### Deprecations and Removals

- Introduces a new `JSONValue` type that models python objects that serialize directly to JSON. This type is used to annotate the contents of `attributes` and `fill_value` fields, replacing the use of the overly wide `object` type. This is technically a breaking change. ([#4037](https://github.com/zarr-developers/zarr-python/issues/4037))
- Promoted a curated "front door" of names to the top-level `zarr_metadata`
  namespace, so consumers can write e.g. `from zarr_metadata import
  ArrayMetadataV3, ShardingIndexLocation, BLOSC_CNAME` instead of importing from
  deep submodule paths. The front door covers every metadata-document TypedDict,
  each codec/chunk-grid/chunk-key-encoding canonical type, the full data-type
  trio for every dtype, and every constant + `Literal` pair. Deep submodule paths
  continue to work unchanged.

  Several promoted names were given clearer, less ambiguous spellings than their
  deep-module names, since they now appear bare at the top level:
  `Endian`/`ENDIAN` → `Endianness`/`ENDIANNESS`,
  `IndexLocation`/`INDEX_LOCATION` → `ShardingIndexLocation`/`SHARDING_INDEX_LOCATION`,
  `RoundingMode`/`ROUNDING_MODE` → `CastRoundingMode`/`CAST_ROUNDING_MODE`,
  `OutOfRangeMode`/`OUT_OF_RANGE_MODE` → `CastOutOfRangeMode`/`CAST_OUT_OF_RANGE_MODE`,
  `DateTimeUnit` → `NumpyTimeUnit`,
  `NamedConfig` → `NamedConfigV3`, and
  `MetadataFieldV3` → `MetadataV3` (matching the name `zarrs` uses for this
  `name`-or-`{name, configuration}` shape).

  Also added the `NUMPY_TIME_UNIT` runtime constant (a `Final` tuple paired with
  the `NumpyTimeUnit` Literal) in `zarr_metadata.v3.data_type.numpy_timedelta64`. ([#4083](https://github.com/zarr-developers/zarr-python/issues/4083))


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
