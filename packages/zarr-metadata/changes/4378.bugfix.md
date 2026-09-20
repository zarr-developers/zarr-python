Concrete v3 entity types are now assignable to the fields they describe.
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
outside the spec's shapes. `zarr_metadata.pydantic` serializers now
declare their return schema via the pydantic-facing shadow types, so
pydantic schema generation stays warning-free.
