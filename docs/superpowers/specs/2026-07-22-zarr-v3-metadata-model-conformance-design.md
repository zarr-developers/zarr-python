# Design: Zarr v3 metadata model conformance

**Status:** Approved and implemented.
**Scope:** `packages/zarr-metadata`, principally the raw v3 metadata types,
the immutable model layer, and their validators and tests.
**Normative authority:** Zarr core protocol v3.1. The behavior of zarrs and
zarr-python is interoperability evidence where the specification leaves an
implementation boundary or where a documented legacy-read exception is
required.

## Goal

Make the v3 array and group metadata models conform to the core Zarr v3.1
document grammar without turning the generic models into interpreters for
individual extensions.

The generic layer will validate required and optional core fields, JSON value
and container types, fixed literals and core cross-field constraints, the
common extension envelope, the core rules governing `must_understand`, and the
forward-compatibility obligations for unknown top-level fields.

It will not validate the configuration semantics of a data type, chunk grid,
chunk key encoding, codec, or storage transformer. Those belong to typed
extension models and readers that resolve extension names.

## Sources and precedence

The implementation will be checked against:

1. [Zarr core protocol v3.1](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html),
   especially Array metadata, Group metadata, Codecs, and Extensions.
2. [zarrs](https://github.com/zarrs/zarrs), especially its `ZarrV3MetadataFieldJSON`,
   `AdditionalFieldV3`, array-opening validation, and consolidated-metadata
   extension handling.
3. Existing zarr-python metadata emitted in the wild, but only for explicit
   tolerant-read paths. Legacy data does not redefine what this package emits
   or calls conformant.

If these disagree, the current v3.1 specification wins. An interoperability
exception must be narrow, documented as non-conformant input, normalized on
read, and never emitted.

## Chosen approach

Evolve the existing normalized `ZarrV3NamedConfig` instead of introducing a
new hierarchy of role-specific extension classes or preserving every source
JSON spelling.

This keeps the public shape small and minimizes compatibility churn.
Validation remains context-aware: the same extension envelope is used at all
extension points, while the array-document validator supplies the few rules
that vary by field.

Two alternatives were rejected:

- A spelling-preserving model would retain shorthand versus object form and
  absent versus empty configuration. That fidelity is not needed by this
  semantic model and would substantially change its API.
- Separate required/optional extension classes would encode
  `must_understand` restrictions in types, but would duplicate behavior and
  force callers to use different classes for otherwise identical envelopes.

## Extension envelope

### Raw type

`ZarrV3NamedConfigJSON` will describe the v3.1 object form:

- required `name: str`;
- optional `configuration: Mapping[str, JSONValue]`; and
- optional `must_understand: bool`.

`ZarrV3MetadataFieldJSON` remains the union of a shorthand name string and this object
form.

The object form accepts only those three members. Extension-owned fields must
be placed inside `configuration`. Rejecting other envelope members matches
zarrs' `deny_unknown_fields` behavior and prevents silently discarding data
during normalization.

### Normalized model

`ZarrV3NamedConfig` will hold:

- `name: str`;
- `configuration: dict[str, JSONValue]`, normalized to an empty mapping when
  absent; and
- `must_understand: bool`, normalized to `True` when absent or when the input
  is a shorthand string.

`to_json()` will use one field-independent canonicalization rule. When the
configuration is empty and `must_understand` is true, it emits the shorthand
name string. Otherwise it emits an object, omitting `configuration` when empty
and omitting `must_understand` when true. Thus the model is semantically
lossless, but deliberately not spelling-preserving. The same rule naturally
emits core data types as strings without giving the data-type field a separate
model or serializer.

### Name validation

Names will be checked only to ensure that they are strings. The model will not
validate registered-name syntax, URI syntax, or membership in the external
extension registry. Those properties are not part of the extension-agnostic
structural boundary and are left to applications that resolve extension
names.

## Context-sensitive `must_understand` rules

An omitted `must_understand` value and every shorthand name mean `True`.

The array validator will reject `must_understand=False` for `data_type`,
`chunk_grid`, and `chunk_key_encoding`. It will permit false for individual
codecs and storage transformers.

This validation is structural only. The model does not decide whether an
implementation recognizes a name and does not remove or skip optional
extensions. A reader that resolves extensions owns that decision.

## Core array and group constraints

The generic array model will enforce the core constraints that do not require
interpreting an extension:

- `zarr_format` is exactly `3` and `node_type` is exactly `"array"`;
- `shape` contains non-negative JSON integers (booleans excluded);
- all mandatory fields are present;
- `fill_value` is JSON, while its data-type-dependent meaning remains opaque;
- `codecs` is a non-empty sequence of valid extension envelopes;
- `storage_transformers`, when present, is a sequence of valid envelopes;
- `attributes`, when present, is a string-keyed JSON object; and
- `dimension_names`, when present, contains strings or null and has the same
  length as `shape`.

The model will not enforce regular-grid rank or positive chunk lengths,
data-type-specific fill values, codec kinds or ordering, or storage-transformer
behavior. These rules require resolving an extension name and therefore live
outside the generic model.

The generic group model will enforce the corresponding core document rules:
fixed literals, required fields, JSON attributes, string top-level keys, and
unknown-field handling.

## Unknown top-level fields

Unknown array and group members will be represented as arbitrary `JSONValue`,
not as a TypedDict that requires an object.

For reader obligations:

- an object whose `must_understand` member is the literal JSON boolean `false`
  is explicitly waivable; and
- every other value, including scalars, arrays, objects with no such member,
  and objects with a non-boolean value, implicitly requires understanding.

This matches the v3.1 implicit-true rule and zarrs' `AdditionalFieldV3`
behavior. The raw and model annotations, `is_*` guards, parsers, and
`must_understand_fields` property must all agree on this representation.

The model itself will not fail merely because such a field requires
understanding. It has no extension registry. It exposes the partition so the
reader can fail when a required field is unrecognized.

## Consolidated metadata

Inline consolidated metadata is a known interoperability extension rather
than a core v3.1 group field. Its dedicated raw and model types will remain,
and group parsing may continue recognizing it explicitly.

The standard emitted form is an object containing `kind`, `metadata`, and a
`must_understand` marker. The extension model validates its own payload and
recursively validates embedded v3 nodes. This special handling must not cause
the generic unknown-field type to become object-only.

Historical zarr-python versions wrote `"consolidated_metadata": null`.
zarrs also accepts this input as a compatibility hotfix. `from_json()` will
continue repairing that exact value to absence, and `to_json()` will never
write it. Documentation and tests will identify this as a tolerant-read
exception, not valid core metadata.

## Errors and normalization

All newly enforced constraints use the existing aggregated
`MetadataValidationError`/`ValidationProblem` mechanism. Locations must point
to the envelope member or array entry that failed, and problem kinds remain
machine-readable.

Parsing still recursively converts JSON arrays to the tuple-backed public
representation. Validation and type guards must agree on canonical runtime
containers after normalization.

No constructor-wide semantic validation will be added to `update()`;
`from_json()` remains the validated ingestion boundary. Direct construction
and `update()` continue to support building or repairing intermediate models.

## Public API compatibility

The existing class names remain. The intended public changes are additive or
corrections to annotations that did not match accepted runtime data:

- `ZarrV3NamedConfigJSON` gains optional `must_understand`;
- `ZarrV3NamedConfig` gains a defaulted `must_understand` field; and
- v3 additional-field annotations widen to arbitrary JSON.

Documents currently accepted only because an extension envelope contains
unknown members, a non-string name, a forbidden false `must_understand`, or an
empty codec list will become validation errors. Those documents violate the
agreed core grammar; accepting them is not a compatibility contract to retain.

## Test strategy

Implementation will be test-first and cover:

1. shorthand and object normalization, including implicit true;
2. explicit true/false round trips;
3. invalid `must_understand` types and unknown envelope members;
4. arbitrary string names accepted and non-string names rejected;
5. false `must_understand` rejected at mandatory extension points;
6. false accepted for codecs and storage transformers;
7. empty codecs rejected without interpreting non-empty pipelines;
8. arbitrary JSON unknown top-level fields and their obligation partition;
9. raw type, parser, type-guard, model, and serializer agreement;
10. consolidated-metadata parsing and the legacy-null read repair; and
11. representative metadata serialized by zarrs.

The package test suite, formatting, linting, and static type checking will run
before the implementation commit.

## Completion criteria

The work is complete when every normative core-document rule in scope has an
explicit validator or a documented extension-owned boundary; raw annotations
and runtime behavior agree; tests demonstrate `must_understand` at every
extension point; zarrs-compatible metadata round-trips semantically; and the
package tests and quality checks pass.
