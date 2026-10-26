# Release notes

<!-- towncrier release notes start -->

## 0.6.0 (2026-10-26)

### Features

- **Breaking:** `ValidationProblem` and `MetadataValidationError` check at run
  time what their annotations declare, and raise `TypeError` otherwise. A
  problem's `loc` is a tuple of strings and integers, so a list, which was
  accepted before, is now refused, as is a `bool`; its `message` is a string
  and its `kind` one of the kinds; an error holds problems. A caller that is
  not type-checked fails where it builds the value, not later, where a
  location is read. ([#4422](https://github.com/zarr-developers/zarr-python/pull/4422))
- `zarr_metadata.typed_json.check(value, SomeTypedDict)` type-checks a JSON
  value against any of the package's TypedDicts -- `ZarrV3ArrayMetadataJSON`,
  `GzipCodecConfiguration`, or one of your own -- and gives back a value of
  it, or None, with every problem located. A TypedDict reads as the typing
  spec defines it, whatever the runtime's attributes say: a key is required
  by its own `Required` or `NotRequired`, or else by the `total` of the class
  that declared it, read from the evaluated annotations, so a module written
  with `from __future__ import annotations` reads like any other; `closed`,
  `extra_items` and openness come from the class, or from its bases when it
  says neither; and each annotation is evaluated in the module of the class
  that wrote it. A key a closed TypedDict does not declare is reported, as
  the new problem kind `unknown_key`, and left out of the value, which still
  comes back; a union of TypedDicts tagged by a `Literal` key is read by the
  branch the tag names. ([#4432](https://github.com/zarr-developers/zarr-python/pull/4432))
- **Breaking:** `must_understand: false` is refused at every extension point
  of an array document, codecs and storage transformers included, not only
  at `data_type`, `chunk_grid` and `chunk_key_encoding`: ignoring a codec
  gives wrong bytes as surely as ignoring a data type gives wrong values,
  and the spec naming only three points is read as an oversight. A document
  that declares a codec or a storage transformer ignorable now has a problem
  at `codecs.N.must_understand` or `storage_transformers.N.must_understand`,
  so the readers refuse it, and no model writes one. The JSON schema `zarr_metadata.pydantic` generates for an array
  document says the same, and so does `resolve`, which reads one field on
  its own. ([#4434](https://github.com/zarr-developers/zarr-python/pull/4434))
- Each extension's metadata is a definition: a value holding the name the
  metadata carries, the TypedDict its configuration is -- the one
  declaration of that JSON -- and plain functions over that TypedDict for
  the rules. `zarr_metadata.definition` and `zarr_metadata.v3.definition` are the
  doors, and their module docstrings the guide. Every extension the package models is one: the
  codecs `blosc`, `bytes`, `cast_value`, `crc32c`, `gzip`, `scale_offset`,
  `sharding_indexed`, `transpose` and `zstd`; every data type, raw bits
  included (`r16` is read by the `r*` definition with `{"bits": 16}`, the size
  its name carries); the `regular` and `rectilinear` chunk grids; and the `default`
  and `v2` chunk key encodings. Each is a constant in its own module --
  `GZIP_CODEC`, `STRUCT_DATA_TYPE` -- and `CORE` holds what the
  specification defines, `CORE_AND_EXTENSIONS` what `zarr-extensions`
  registers besides. A third party writes a TypedDict, a rule function and
  a definition, and adds it to a scope; a definition checks itself
  when built, and nothing happens at class creation.

  Reading a field is three steps, each usable on its own by a caller
  holding nothing but JSON. `check(value, SomeTypedDict)`, from
  `zarr_metadata.typed_json`, type-checks JSON against a TypedDict and needs
  nothing else; `definition.read_configuration(configuration)` is the check, each nested
  field's envelope judged, and then the rules; `resolve(field,
  CodecDefinition, scope)` reads a whole field in a scope -- its envelope
  judged, its name related to a definition, its configuration judged, each
  nested field read the same way -- and returns what the scope made of it,
  with every problem located. A name nothing in scope claims is left
  unjudged. A shard's pipelines, a cast's target
  type and a struct's field types are nested fields, annotated `ZarrV3CodecField`
  and `ZarrV3DataTypeField`, and read in the scope their field is read in.
  A codec definition also says whether the size
  of what it gives out is `static` or `dynamic`, and a member annotated
  `ZarrV3StaticCodecField` takes codecs of static size only: a shard's
  `index_codecs`, where the spec forbids a compressor, so a `gzip` there is a
  problem at its place.

  The TypedDict reads as the typing spec defines it -- `total`, `Required`,
  `NotRequired`, `closed` and `extra_items` mean what they mean to a type
  checker, whether or not its module postpones annotations. A key a closed
  TypedDict does not declare is reported, as `unknown_key`, and left out of
  what the rules are handed. A definition refuses a TypedDict that says
  nothing of the keys it does not declare, and a member typed
  `ZarrV3MetadataFieldJSON`, which would check as plain JSON.

  `canonical_of(field, problems)` gives a field without problems in its
  simplest equivalent spelling: each nested field in its own simplest
  spelling, then its definition's own `canonical`, and the envelope in the
  fewest words every reader takes, with no `must_understand`. A field with
  any problem, an unknown key included, has
  none, since a simpler spelling would erase what its author wrote. What
  `canonical` gives is judged again, and one that does not hold is a
  `ValueError`, a fault in the definition. ([#4434](https://github.com/zarr-developers/zarr-python/pull/4434))
- **Breaking:** the v3 readers read each extension point through its
  definition: `read_array_metadata_v3` judges the data type, the chunk
  grid, the chunk key encoding, each codec and each storage transformer
  against the definition that claims its name in a scope,
  `CORE_AND_EXTENSIONS` unless a `context` is passed, so a gzip `level` of
  99, or a key a codec does not declare, is a problem located in the
  configuration, and a document holding one, accepted before, is refused.
  So do the v3 group readers, and the v3 model classes' `from_json` and
  `from_key_value` take the same `context`.
  A name nothing in scope claims is left unjudged. A problem in a field
  that a configuration holds, such as a codec inside a shard, is reported
  at that field's own location, and the field that holds it is still read. ([#4436](https://github.com/zarr-developers/zarr-python/pull/4436))
- **Breaking:** a v3 array's `fill_value` is judged against its `data_type`.
  Each data type's definition declares the JSON shape of its fill value,
  `fill_value`, and the rules for one of that shape, `fill_value_rules`: an
  `int8` fill value of 300, a `float32` hex string of another width, and a
  struct fill value missing a field are each a problem at `fill_value`,
  where the package accepted them before. `fill_value_problems(data_type,
  value)` judges a fill value against a data type field a scope read, and
  a field that is read keeps the fields it read inside as
  `nested`, a `Nested` mapping by location. A data type nothing
  in scope claims leaves its fill value unjudged. ([#4443](https://github.com/zarr-developers/zarr-python/pull/4443))
- **Breaking:** a v3 array's sharding codec is judged against the shard it
  is handed, and its two pipelines are read as the array's is. Its
  `chunk_shape` has an inner chunk length for each axis of the shard,
  dividing each length the shards take along it -- the shard being the
  chunk the codec is handed, so a transposed shard is divided along its
  transposed axes. Its inner codecs are handed the inner chunks, of the
  shard's data type, and its index codecs the shard index, of `uint64`,
  with an axis more than the shard. A shard that does not divide, or a
  pipeline that does not fit what it is handed -- an index `bytes` codec
  without an `endian` -- has a problem where it sits, where the package
  accepted it before. A codec definition
  says what the pipelines it holds are handed, `pipelines`, and each
  codec's `Stage` keeps the stages of those pipelines as `inner`. ([#4443](https://github.com/zarr-developers/zarr-python/pull/4443))
- **Breaking:** a v3 array's `chunk_grid` is judged against its `shape`, by
  the grid's definition. A regular grid whose `chunk_shape` does not have
  one length per dimension of the shape, and a rectilinear grid whose
  `chunk_shapes` does not have one entry per dimension, or whose chunk
  lengths fall short of their dimension, each have a problem in
  `chunk_grid.configuration`, where the package accepted them before. ([#4443](https://github.com/zarr-developers/zarr-python/pull/4443))
- **Breaking:** a v3 array's `codecs` are read as a pipeline: in order --
  array -> array codecs, then one array -> bytes codec, then bytes -> bytes
  codecs -- and each codec judged against the chunk it is handed, which is
  the grid's chunks of the array's data type, as the array -> array codecs
  before it hand them on. A codec out of order, a second array -> bytes
  codec or none at all; a `bytes` codec without an `endian` handed numbers
  of several bytes, or handed values that vary in size; a `transpose` whose
  `order` has another number of axes than its chunk; a `cast_value` to or
  from a data type that models no real numbers, wrapping to one that is
  not integral, or mapping a scalar that is not a fill value of the data
  type on its side; a `scale_offset` handed values that are no numbers, or
  with an `offset` or `scale` that is not a value of its data type; and a
  struct field whose values vary in size are each a problem where they
  sit, where the package accepted them before. A reading records each
  codec with the chunk it is handed, as a `Stage`, and the lengths a
  grid's chunks take; `storage_of(data_type)` says how a data type's
  values are stored. A
  definition's `rules` are handed the fields its configuration holds as
  the scope read them, and the kinds gain `chunk_lengths` (grids),
  `storage` (data types), and `chunk_rules` and `transition` (codecs). ([#4443](https://github.com/zarr-developers/zarr-python/pull/4443))
- **Breaking:** metadata is now read against the definitions in a scope, a
  `ZarrV3Context` or `ZarrV2Context` that a reader builds from the codecs,
  data types and other fields it supports, and a model is the document
  together with the scope it was read in. The `read_*` readers report every
  problem at its location, `unclaimed_fields` lists each field the scope
  does not know, so a reader can tell whether it can open a node, repair
  readers undo known zarr-python writer bugs, and JSON Schemas are exported.
  Code written for 0.5.0 must change: models are no longer dataclasses,
  their field members hold `AcceptedField` or `UnclaimedField` values,
  `update` takes JSON and takes members the spec does not define through
  `extra_fields`, the `validate_*`, `is_*` and `parse_*` functions for
  documents and the `...Partial` TypedDicts are removed, some names moved to
  another module, the v2 constants and models follow the v3 names and
  shapes, and a v2 time dtype without a unit is refused. `to_json` now
  writes each field as it was read, rather than shortening a codec to a bare
  name that zarr-python could not open, and zarr-metadata now depends on
  `annotated-types`. ([#4490](https://github.com/zarr-developers/zarr-python/pull/4490))

### Bugfixes

- The v2 array validator follows the spec on two points it was stricter than: members outside the `.zarray` definition are accepted when they are JSON values with string keys, and are kept and written back by `to_json` and `to_key_value`; any other member is a problem, as in a v3 document (the spec says other keys ["SHOULD NOT be present ... and SHOULD be ignored"](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L91-L92), unlike `.zgroup`'s ["MUST NOT"](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L313)), and `filters: []` is accepted (["a list of JSON objects providing codec configurations, or null"](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L76-L79) sets no minimum). The JSON Schema that `zarr_metadata.pydantic` generates follows: it accepts other members, and the filter pipeline has no minimum length. ([#4365](https://github.com/zarr-developers/zarr-python/pull/4365))
- Concrete v3 entity types are now assignable to the fields they describe.
  Previously, none of the package's canonical codec / chunk-grid /
  chunk-key-encoding / data-type types (e.g. `BloscCodecMetadata`,
  `RegularChunkGridMetadata`) satisfied `ZarrV3MetadataFieldJSON`, so a
  type checker rejected putting them into the very fields they document
  (`codecs`, `chunk_grid`, `data_type`, ...). Three changes fix this:

  - `ZarrV3NamedConfigJSON.name` and `.configuration` are now `ReadOnly`
    (PEP 705), making them covariant so concrete `name: Literal[...]` and
    required-`configuration` shapes are accepted.
  - `ZarrV3NamedConfigJSON` is now `closed` (PEP 728): the spec's
    named-configuration envelope has exactly `name` / `configuration` /
    `must_understand`, and closing the type also makes it usable as a
    `JSONValue` (needed for e.g. the `sharding_indexed` inner `codecs`).
  - Every concrete `*Object` / `*Configuration` TypedDict is now `closed`,
    and object forms declare `must_understand: NotRequired[bool]` (any v3
    metadata field may carry the extension member).

  **Soft-breaking** for type-checking consumers: dicts with keys beyond the
  declared shape no longer satisfy the closed types, and `name` /
  `configuration` can no longer be mutated through `ZarrV3NamedConfigJSON`.
  Both were previously accepted by type checkers but produced documents
  outside the spec's shapes. Generating a pydantic JSON Schema for
  the `zarr_metadata.pydantic` field types no longer emits warnings.

  ([#4378](https://github.com/zarr-developers/zarr-python/pull/4378))
- Attributes are user data, and a non-finite number in them is read,
  validated and written back. zarr-python writes attributes with the
  defaults of Python's `json` module, so an xarray `_FillValue` or a CF
  `missing_value` of NaN is stored as a bare `NaN`, which RFC 8259 lacks;
  the model refused such a document and could not read the store at all.
  A node's `attributes`, a v2 `.zattrs`, the `.zattrs` entries of a v2
  `.zmetadata`, and the attributes of the nodes a v3 group's inline
  `consolidated_metadata` holds may now hold `NaN`, `Infinity` and
  `-Infinity`, in every reader, such as `read_array_metadata_v3`, and in
  each model's `from_json`, `from_key_value` and `to_key_value`. Wherever
  the spec interprets a value it spells those numbers as strings, so
  anywhere else a non-finite number is still refused, and the store reader
  now says where one is instead of failing to decode the document.

  A model whose document the reader refuses raises
  `MetadataValidationError` with every problem, instead of being written:

  - a non-finite number outside attributes, which was refused before too;
  - a non-string attribute key, which was written as a string;
  - a value that is not JSON, which raised `TypeError`;
  - anything else the reader refuses, such as `dimension_names` longer than
    `shape`, which was written and then refused on read.

  Any sequence is written as an array. A number too large for a float, such
  as `1e400`, is read as infinity, so in attributes it is now accepted and
  written back as `Infinity`.

  Through `zarr_metadata.pydantic`, pydantic writes a non-finite attribute
  as `null` unless the enclosing model sets `ser_json_inf_nan="constants"`.

  ([#4420](https://github.com/zarr-developers/zarr-python/pull/4420))
- `from_key_value` takes a mapping keyed by `str` or by any narrower key
  type, so reading back what `to_key_value` wrote type-checks. `to_key_value`
  keys its result by the store-key literals, and since a mapping's key type is
  invariant, a `Mapping[str, bytes]` parameter refused it. ([#4421](https://github.com/zarr-developers/zarr-python/pull/4421))
- `MetadataValidationError` pickles and copies with its message. A copy was
  built from the error's message, and its message came out as the
  original's characters, one per line. ([#4422](https://github.com/zarr-developers/zarr-python/pull/4422))
- A `bytes` or `bytearray` is not a JSON array. Where a document expects an
  array, an empty one passed as an array with no items: the v3 parser
  returned `storage_transformers=b""`, and the v2 readers took `dtype=b""`,
  then failed when the model was written. The readers and models now refuse
  one at `storage_transformers` and at a v2 `dtype` and its field records. At
  `codecs`, `dimension_names` and v2 `filters`, which already refused one,
  the problem now says the value has the wrong type. ([#4423](https://github.com/zarr-developers/zarr-python/pull/4423))
- A model read by `from_json` or `from_key_value` shares no mutable state
  with the document it was read from, as a document from `to_json` already
  shares none with its model. A frozen model held the caller's own
  dictionaries, so editing the input after reading it changed the model,
  and could make `to_key_value` fail. ([#4426](https://github.com/zarr-developers/zarr-python/pull/4426))
- `from_key_value` checks that each store value it reads is `bytes`, as its
  signature says: a `str` was decoded as a JSON document, and `None` or a
  `memoryview` raised `TypeError`. Each is now a `MetadataValidationError`
  with an `invalid_type` problem at the store key. ([#4428](https://github.com/zarr-developers/zarr-python/pull/4428))
- `ZarrV3ArrayMetadata.create_default(shape=...)` derives a chunk length of 1
  for a dimension of length 0, where it wrote 0: the regular grid asks for
  chunk lengths greater than zero, and `zarr` does not open a grid with one. ([#4443](https://github.com/zarr-developers/zarr-python/pull/4443))

### Improved Documentation

- The docs now render the release notes as a page instead of linking to
  `CHANGELOG.md` on GitHub, so each docs version shows its own changelog. ([#4311](https://github.com/zarr-developers/zarr-python/pull/4311))

### Misc

- `ZarrV2ConsolidatedMetadataJSON.zarr_consolidated_format` is typed
  `Literal[1]` rather than `int`. Format 1 is the only defined `.zmetadata`
  format and the runtime validator already rejected anything else, so the
  type now carries the constraint the validator enforces instead of
  contradicting it. ([#4378](https://github.com/zarr-developers/zarr-python/pull/4378))
- **Breaking:** `MetadataValidationError.problems` is now a tuple instead of
  a list. Iteration and indexing are unchanged; code that changes a report
  must first copy it with `list(problems)`. `MetadataValidationError` still
  accepts any problem sequence. ([#4378](https://github.com/zarr-developers/zarr-python/pull/4378))
- `ZarrV2ZGroupJSON` and `ZarrV2GroupMetadataJSON` are closed
  (PEP 728), as the spec's "Other keys MUST NOT be present" and the v2 group
  validator already said, and as its pydantic schema already was. A type
  checker now refuses a v2 group document with a key it does not declare. ([#4430](https://github.com/zarr-developers/zarr-python/pull/4430))


## 0.5.0 (2026-08-14)

### Bugfixes

- `JSONValue`'s array arm is now the covariant `Sequence["JSONValue"]` rather
  than the invariant `list["JSONValue"] | tuple["JSONValue", ...]`. Values typed
  with a narrower element type — a `list[str]` field on a TypedDict, a
  `Sequence[float]` — now count as JSON values, and TypedDicts whose fields
  carry precise types are now assignable to `Mapping[str, JSONValue]`.
  Type-level cost, accepted deliberately: `Sequence` says nothing about the
  concrete container and admits `str`/`bytes`, so runtime code narrowing a JSON
  array must exclude `str`/`bytes`/`bytearray` — as it already had to, since
  `str` was always a union arm. ([#4264](https://github.com/zarr-developers/zarr-python/pull/4264))

### Deprecations and Removals

- Unified the naming grammar for SCREAMING_SNAKE constants with the one used for
  type names. A constant's name is now a purely syntactic transformation of the
  name of the `Literal` type it manifests, so the format version is spelled
  `ZARR_V2`/`ZARR_V3` and comes first, matching the `ZarrV2`/`ZarrV3` prefix on
  the corresponding type:

  - `ARRAY_METADATA_STORE_KEY_V2` → `ZARR_V2_ARRAY_METADATA_STORE_KEY`
  - `ARRAY_METADATA_STORE_KEY_V3` → `ZARR_V3_ARRAY_METADATA_STORE_KEY`
  - `ATTRIBUTES_STORE_KEY_V2` → `ZARR_V2_ATTRIBUTES_STORE_KEY`
  - `GROUP_METADATA_STORE_KEY_V2` → `ZARR_V2_GROUP_METADATA_STORE_KEY`
  - `GROUP_METADATA_STORE_KEY_V3` → `ZARR_V3_GROUP_METADATA_STORE_KEY`
  - `CONSOLIDATED_METADATA_STORE_KEY_V2` → `ZARR_V2_CONSOLIDATED_METADATA_STORE_KEY`
  - `ARRAY_ORDER_V2` → `ZARR_V2_ARRAY_ORDER`
  - `ARRAY_DIMENSION_SEPARATOR_V2` → `ZARR_V2_ARRAY_DIMENSION_SEPARATOR`
  - `CONSOLIDATED_METADATA_KEY_V3` → `ZARR_V3_CONSOLIDATED_METADATA_KEY`

  The old names are removed, not aliased. This supersedes the 0.4.0 convention
  under which type names put the format version first while constants put it
  last: every constant that manifests a `Literal` type now follows the same rule
  as that type.

  The last of those is the one rename the syntactic rule does not force:
  `ZARR_V3_CONSOLIDATED_METADATA_KEY` manifests no `Literal` type, so it is
  outside the rule and was renamed for consistency with its siblings.

  Digit runs stay glued to the token they follow, so spec vocabulary is
  preserved: `Uint8DataTypeName` pairs with `UINT8_DATA_TYPE_NAME` (not
  `UINT_8_...`) and `Crc32cCodecName` with `CRC32C_CODEC_NAME`. No dtype, codec,
  chunk-grid, or chunk-key-encoding constant changed name.

  Constants that do not manifest a `Literal` type are outside the rule and are
  unchanged: the `*_METADATA_*_KEYS_V2`/`_V3` key sets, the
  `CANONICAL_*_HEX_FLOAT*` bit patterns, and `UNSET`. The key sets keep the
  version-last spelling, so `zarr_metadata.model` exports both
  `ARRAY_METADATA_REQUIRED_KEYS_V2` and `ZARR_V2_ARRAY_METADATA_STORE_KEY`. They
  name validation policy rather than a spec document, have no paired type to
  derive from, and renaming them would be a second breaking change buying only
  cosmetic consistency — so it is deliberately deferred.

  `tests/test_public_api.py::test_constant_names_derive_from_their_type_names`
  derives every constant name from the type it manifests and asserts they match,
  so the two grammars cannot diverge again.

  Store keys also moved to the modules that describe the documents they name,
  matching the package's layering (the `v2`/`v3` modules describe the specs; the
  `model` layer is built on top of them). `ZARR_V2_ATTRIBUTES_STORE_KEY` now
  lives in `zarr_metadata.v2.attributes` beside the `.zattrs` type it names,
  rather than in the array model; the other five moved likewise, and
  `ZarrV2AttributesStoreKey` is no longer an array-specific concept.
  `zarr_metadata.model` re-exports all six, so
  `from zarr_metadata.model import ZARR_V2_ARRAY_METADATA_STORE_KEY` is
  unaffected.

  `CONSOLIDATED_METADATA_KEY_V3` moved to `zarr_metadata.v3.consolidated` and was
  renamed to `ZARR_V3_CONSOLIDATED_METADATA_KEY` for consistency. It is not a
  store key: unlike v2's `.zmetadata` file, v3 consolidated metadata is embedded
  as an extension field inside the group's own `zarr.json`.

  All seven keys and the six store-key `Literal` aliases are now also exported
  from the top-level `zarr_metadata` namespace, alongside the document types and
  the rest of the spec vocabulary, so `from zarr_metadata import
  ZARR_V2_ARRAY_METADATA_STORE_KEY` works. The model layer's validators, parsers,
  type guards, and metadata key sets remain `zarr_metadata.model` imports.

  ([#4232](https://github.com/zarr-developers/zarr-python/pull/4232))

### Misc

- The source distribution now ships an explicit allowlist (`/src`, `/tests`,
  `/docs`, `/mkdocs.yml`, `/justfile`, `/CHANGELOG.md`) rather than whatever
  happens to sit in the package directory, so an sdist both tests and documents
  itself and cannot pick up scratch files from the tree it was built in. ([#4248](https://github.com/zarr-developers/zarr-python/pull/4248))


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

- Introduces a new `JSONValue` type that models python objects that serialize directly to JSON. This type is used to annotate the contents of `attributes` and `fill_value` fields, replacing the use of the overly wide `object` type. This is technically a breaking change. ([#4037](https://github.com/zarr-developers/zarr-python/pull/4037))
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
  the `NumpyTimeUnit` Literal) in `zarr_metadata.v3.data_type.numpy_timedelta64`. ([#4083](https://github.com/zarr-developers/zarr-python/pull/4083))


## 0.2.0 (2026-05-19)

### Bugfixes

- `GzipCodecConfiguration.level` is now required, and `GzipCodecMetadata`
  no longer accepts the bare-string `"gzip"` form. The codec's compressed
  output depends on `level`, so metadata that omits it cannot reproducibly
  identify the chunk bytes produced by a writer. **Breaking** for consumers
  that previously typed gzip codec metadata as the bare string or
  constructed a `GzipCodecConfiguration` without `level`.
  ([#3978](https://github.com/zarr-developers/zarr-python/pull/3978))
- `BytesCodecObject.configuration` is now `NotRequired`. The configuration
  has no required keys (`endian` is conditionally required at runtime
  based on data type), so the object form may omit it entirely — matching
  the bare-string short-hand. **Soft-breaking** for consumers that
  previously relied on `configuration` always being present.
  ([#3978](https://github.com/zarr-developers/zarr-python/pull/3978))
- Better modelling of Zarr v2 stored metadata. Zarr v2 splits a node's
  metadata across two JSON documents (`.zarray`/`.zgroup` and `.zattrs`),
  but `GroupMetadataV2` had no `attributes` field while `ArrayMetadataV2`
  did — an inconsistency. `GroupMetadataV2` now also has an optional
  `attributes` field, and `ArrayMetadataV2.attributes` is now
  `NotRequired` for symmetry. **Soft-breaking** for consumers that
  relied on `ArrayMetadataV2.attributes` always being present.
  ([#3962](https://github.com/zarr-developers/zarr-python/pull/3962))

### Features

- Added `ArrayMetadataV3Partial`, `GroupMetadataV3Partial`,
  `ArrayMetadataV2Partial`, and `GroupMetadataV2Partial` — sibling
  TypedDicts to the existing full metadata types, declared with
  `total=False` so every field is `NotRequired`. Use these when typing
  dicts that intentionally hold a subset of a complete metadata document
  (test fixtures, fragment templates, in-progress builders). An
  equivalence test pins each `Partial` to the keys and value types of
  its full sibling so the two cannot drift.
  ([#3982](https://github.com/zarr-developers/zarr-python/pull/3982))
- Added three new top-level types modelling the **strict on-disk** shape
  of Zarr v2 metadata documents: `ZArrayMetadata` (the `.zarray` file),
  `ZGroupMetadata` (the `.zgroup` file), and `ZAttrsMetadata` (the
  `.zattrs` file). Use these when you want a type that faithfully matches
  what's stored on disk; use the merged `ArrayMetadataV2`/`GroupMetadataV2`
  when you want the in-memory representation a Python program typically
  works with.
  ([#3962](https://github.com/zarr-developers/zarr-python/pull/3962))
- Added typed constants exposing the spec-permitted values of constrained
  Literal fields, importable at the per-codec module level. For example,
  `from zarr_metadata.v3.codec.bytes import ENDIAN` provides
  `("little", "big")` as a tuple, enabling runtime iteration or validator
  generation without re-stating the Literal values by hand.
  ([#3978](https://github.com/zarr-developers/zarr-python/pull/3978))

## 0.1.1 (2026-05-06)

### Misc

- First usable release on PyPI. Version 0.1.0 was uploaded then deleted to
  reserve the project name; this version is the first one PyPI will install.
  No source changes from 0.1.0.
  ([#3949](https://github.com/zarr-developers/zarr-python/pull/3949))

## 0.1.0 (2026-05-01)

### Features

- Initial release. Provides `TypedDict` definitions and `Literal` aliases
  for the JSON shapes specified by Zarr v2 and v3 metadata, plus a subset
  of `zarr-extensions` types and the un-specified-but-widely-used
  consolidated metadata documents. Pair with a runtime validator like
  `pydantic` to check JSON loaded from disk.
  ([#3919](https://github.com/zarr-developers/zarr-python/pull/3919))
