# Zarr Metadata Review Fixes Design

## Goal

Correct the significant conformance and public-contract defects found by the
independent branch review without changing the documented ability to construct
temporarily invalid model instances for metadata repair workflows.

## Scope

This change fixes four areas:

1. Validation of raw Zarr v2 `.zarray` and `.zgroup` documents must be distinct
   from validation of the library's merged in-memory document representation.
2. Public `TypeIs` predicates must narrow only values that actually inhabit the
   type named in their return annotation.
3. The public Zarr v2 dtype type must represent nested structured dtypes.
4. Pydantic-generated JSON schemas must encode the structural constraints that
   the corresponding runtime parsers enforce.

The inaccurate consolidated-metadata docstring is minor and outside this fix.
Existing model-instance pass-through in the Pydantic integration remains
unchanged because it is part of the documented repair-state design.

## Raw V2 Documents and Merged Models

The public `ZarrV2ArrayMetadataJSON` and `ZarrV2GroupMetadataJSON` types describe
an in-memory representation in which an optional `attributes` member has been
merged from `.zattrs`. They do not describe the raw `.zarray` and `.zgroup`
objects stored on disk.

Private raw-document parsers will enforce the storage specification before any
merge occurs:

- `.zarray` accepts exactly the required array metadata members plus the
  optional `dimension_separator` member. Every other member, including
  `attributes`, is an error.
- `.zgroup` accepts exactly `zarr_format`. Every other member, including
  `attributes`, is an error.
- `.zattrs`, when present, must be a JSON object with string keys and JSON
  values. It is then added as the merged `attributes` member.
- An absent `.zattrs` remains distinguishable from an explicitly empty object,
  preserving the existing `UNSET` behavior.

The existing public merged-document validators remain strict. This avoids
silently discarding data and keeps their declared `TypedDict` contracts sound.

## Sound Type Guards

Validation and parsing intentionally accept abstract `Mapping` and `Sequence`
inputs. Parsers materialize those inputs into canonical dictionaries and
tuples. Type guards cannot make the same promise because `TypeIs[T]` asserts
that the original object already inhabits `T`.

`is_json` will therefore require recursively canonical JSON container types:
`dict`, `list`, and `tuple`, with string dictionary keys and canonical children.
It will reject abstract containers such as `UserDict` and `range`, even though
`parse_json` continues to accept and normalize them.

`is_metadata_field_v3` will likewise require either a string or a concrete
dictionary whose nested values are already canonical. Parsing continues to
accept abstract mappings and materialize them.

## Recursive V2 Dtype Type

`ZarrV2DataTypeMetadata` will be a named recursive alias. A structured field's
datatype may be either a dtype string or another structured dtype tuple. The
optional subarray shape remains a tuple of integers. Runtime validation already
accepts this structure, so this change aligns static typing and generated schema
with existing spec-compliant behavior.

## Pydantic JSON Schema Parity

The Pydantic field aliases will retain their current runtime behavior: raw
documents are parsed through the core model constructors, and existing core
model instances pass through unchanged.

Their JSON schema input definitions will be replaced or annotated with
constraints that express the runtime document rules, including:

- non-negative shape and chunk dimensions;
- at least one v3 codec;
- closed named-configuration objects;
- closed v2 merged documents and consolidated payloads where runtime parsing is
  closed;
- `must_understand` constraints at mandatory v3 extension points; and
- the recursive v2 structured-dtype representation.

Constraints that depend on semantic interpretation outside this package's
structural validators will not be added. The generated schema should describe
accepted raw document inputs, not the intentionally permissive state of an
already-constructed core model instance.

## Error Handling

All new raw-v2 failures use `MetadataValidationError` and preserve precise
locations. An illegal `.zarray` member named `attributes`, for example, reports
the location `("attributes",)` and the existing `invalid_value` kind.
Malformed `.zattrs` values report locations beneath `attributes` after the
merge boundary, consistent with the public merged representation.

## Testing

Implementation follows red-green-refactor cycles. Regression coverage will
prove:

- raw `.zarray` and `.zgroup` reject `attributes` and other extra members;
- valid sibling `.zattrs` still merges, including absent-versus-empty behavior;
- `is_json(range(3))` and `is_metadata_field_v3(UserDict(...))` are false while
  the corresponding parsers still normalize those inputs;
- nested structured v2 dtypes type-check through Pydantic and validate at
  runtime; and
- generated schemas reject representative inputs that runtime parsing rejects:
  empty codecs, negative dimensions, forbidden extra members, and unsupported
  `must_understand: false` values.

Focused package tests, formatting, linting, type checking where configured, and
the full repository test suite will run before the implementation is committed.
