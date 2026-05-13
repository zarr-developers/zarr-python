# zarr-metadata Follow-Up Recommendations

This document catalogs type-surface gaps and missing runtime constants in zarr-metadata 0.1.1 that prevent deeper integration with zarr-python. Each section identifies a concrete gap, shows what zarr-python requires, and proposes what zarr-metadata should add.

## Working principle

When zarr-metadata and zarr-python disagree on a type or shape, neither side automatically wins. Each disagreement is a prompt to think:

- Is zarr-python doing the right thing and zarr-metadata is wrong?
- Is zarr-metadata doing the right thing and zarr-python is wrong (carrying a quirk that should be fixed)?
- Are both wrong, and a third design fits better?

Examples already known on this branch: zarr-python's `GroupMetadata.to_dict()` emits `node_type` even for v2 (a spec violation that pre-dates strict typing); zarr-python's `RectilinearDimSpecJSON` uses `list` while zarr-metadata uses `tuple` (different concerns — JSON-deserialized form vs canonical form — both valid in their domain).

The recommendations below are *prompts for discussion*, not decrees. Each entry should be evaluated on its own terms before any change lands in zarr-metadata.

## 1. Current Public API Inventory

**zarr-metadata 0.1.1** exports:

### Top-level (`zarr_metadata/__init__.py`)
- `NamedConfig` — externally-tagged union for metadata fields
- `ArrayMetadataV2`, `ArrayMetadataV3`, `GroupMetadataV2`, `GroupMetadataV3`
- `ConsolidatedMetadataV2`, `ConsolidatedMetadataV3`
- `ExtensionFieldV3`, `MetadataFieldV3` — validator types
- `ArrayDimensionSeparatorV2`, `ArrayOrderV2`, `DataTypeMetadataV2`, `CodecMetadataV2`

### Codec v3 (`v3/codec/__init__.py`)
Nine codec-metadata aliases: `BloscCodecMetadata`, `BytesCodecMetadata`, `CastValueCodecMetadata`, `Crc32cCodecMetadata`, `GzipCodecMetadata`, `ScaleOffsetCodecMetadata`, `ShardingIndexedCodecMetadata`, `TransposeCodecMetadata`, `ZstdCodecMetadata`

### Data types v3 (`v3/data_type/__init__.py`)
Twenty dtype names + fill value types (e.g., `Float32DataTypeName`, `Float32FillValue`); plus three envelope types: `NumpyDatetime64`, `NumpyTimedelta64`, `Struct`

### Chunk key encodings v3 (`v3/chunk_key_encoding/`)
- `DefaultChunkKeyEncodingMetadata`, `DefaultChunkKeyEncodingConfiguration`
- `V2ChunkKeyEncodingMetadata` (via `v2.py`)

---

## 2. Missing Runtime Constants (Final Tuples)

**Gap:** zarr-metadata defines literal types (e.g., `BloscShuffle = Literal["noshuffle", "shuffle", "bitshuffle"]`) but doesn't export the corresponding runtime tuple constants zarr-python uses.

**zarr-python has** (`src/zarr/core/dtype/common.py` lines 19–30):
```python
ENDIANNESS_STR: Final = "little", "big"
SPECIAL_FLOAT_STRINGS: Final = ("NaN", "Infinity", "-Infinity")
OBJECT_CODEC_IDS: Final = ("vlen-utf8", "vlen-bytes", "vlen-array", "pickle", "json2", "msgpack2")
```

**zarr-python has** (`src/zarr/codecs/blosc.py` line 42):
```python
SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")
```

**zarr-python has** (`src/zarr/core/common.py` line 48):
```python
ANY_ACCESS_MODE: Final = "r", "r+", "a", "w", "w-"
```

**Proposal:** For each `Literal[...]` type in zarr-metadata (especially those reused across modules), export a sibling `Final` tuple. Examples:
- `BloscShuffle = Literal[...]` → add `BLOSC_SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")`
- `Endian = Literal["little", "big"]` (`v3/codec/bytes.py` line 17) → add `ENDIAN: Final = ("little", "big")`
- `DefaultChunkKeyEncodingSeparator = Literal["/", "."]` → add `DEFAULT_CHUNK_KEY_ENCODING_SEPARATORS: Final = ("/", ".")`
- `ObjectCodecID = Literal[...]` (if added; see below) → add `OBJECT_CODEC_IDS: Final = (...)`

**Status:** Additive; no breaking changes.

---

## 3. Missing Type Aliases: dtype Name and Fill-Value Constants

**Gap:** zarr-python defines string constants and special-value sentinels for each dtype; zarr-metadata lacks a cross-dtype registry or naming convention for them.

**Examples zarr-python has:**
- `src/zarr/core/dtype/npy/float.py` (line 24): per-float dtype v2 names as class variables
- `src/zarr/core/dtype/common.py` (line 28): `ObjectCodecID = Literal["vlen-utf8", "vlen-bytes", ...]`
- `src/zarr/core/dtype/common.py` (lines 19–23): `EndiannessStr`, `SpecialFloatStrings`, `JSONFloatV2`, `JSONFloatV3`
- `src/zarr/core/dtype/npy/common.py` (lines 1–3): `DATETIME_UNIT`, `NUMPY_ENDIANNESS_STR`

**zarr-metadata has:**
- Per-dtype modules (`v3/data_type/float32.py`, etc.) with `Float32DataTypeName = Literal["float32"]`, `Float32SpecialFillValue`, `Float32FillValue`
- No cross-dtype registry of names or fill-value patterns

**Proposal:**
1. Add `ObjectCodecID = Literal["vlen-utf8", "vlen-bytes", "vlen-array", "pickle", "json2", "msgpack2"]` to `v2/data_type/` or `_common.py`
2. Add `OBJECT_CODEC_IDS: Final = (...)` as the runtime constant
3. Optionally add a `v3/data_type/object.py` for object-codec dtype shapes if v3 supports them
4. Create `v2/data_type/common.py` for v2-shared constants (v2 dtype names, endianness, special floats)

**Status:** Additive.

---

## 4. Missing Per-Codec V2 Configuration TypedDicts

**Gap:** zarr-metadata has only `CodecMetadataV2 = {"id": str, ...}` (generic). zarr-python defines per-codec v2 configs for specific behaviors.

**zarr-python has:**
- `src/zarr/codecs/blosc.py` (line 45): `BloscConfigV2 = TypedDict` with fields `cname`, `clevel`, `shuffle`, `blocksize`, `typesize`

**zarr-metadata has:**
- `v2/codec.py` (line 7): only generic `CodecMetadataV2`

**Proposal:** Add `CodecMetadataV2Blosc` (or `BloscCodecConfigurationV2`?) TypedDict to `v2/codec.py` or a new `v2/codec/blosc.py`, matching zarr-metadata's per-codec pattern for v3:
```python
class BloscCodecConfigurationV2(TypedDict):
    cname: str  # Blosc compressor ID
    clevel: int
    shuffle: int
    blocksize: int
    typesize: NotRequired[int]
```

Future codecs may need similar treatment.

**Status:** Additive; may become breaking if zarr-python standardizes on these types for v2 codec dispatch.

---

## 5. Chunk Key Encoding: Missing Types

**Gap:** zarr-python's `ChunkKeyEncodingParams` TypedDict doesn't match zarr-metadata's per-encoding types.

**zarr-python has** (`src/zarr/core/chunk_key_encodings.py` line 27):
```python
class ChunkKeyEncodingParams(TypedDict):
    name: Literal["v2", "default"]
    separator: NotRequired[SeparatorLiteral]
```

**zarr-metadata has:**
- `DefaultChunkKeyEncodingConfiguration`, `DefaultChunkKeyEncodingMetadata` (with optional short-hand form)
- `V2ChunkKeyEncodingMetadata` (with optional short-hand form)
- No `ChunkKeyEncodingParams` union covering both

**Proposal:** Add to `v3/chunk_key_encoding/__init__.py`:
```python
ChunkKeyEncodingParams = DefaultChunkKeyEncodingObject | V2ChunkKeyEncodingObject
```
Or add a generic TypedDict analogous to `NamedConfig` to capture the `{name, configuration}` pattern for all chunk key encodings.

**Status:** Additive.

---

## 6. ConsolidatedMetadataV2 has the wrong value type

**Gap:** `ConsolidatedMetadataV2.metadata` ([packages/zarr-metadata/src/zarr_metadata/v2/consolidated.py:27](packages/zarr-metadata/src/zarr_metadata/v2/consolidated.py#L27)) types values as `GroupMetadataV2 | ArrayMetadataV2`. That's wrong on two counts:

1. The map's keys are file paths including the file suffix — `"foo/.zarray"`, `"foo/.zgroup"`, `"foo/.zattrs"`, etc. (the docstring acknowledges this). `.zattrs` entries have no representation in the union.
2. `ArrayMetadataV2` is the *merged* in-memory shape that folds `.zattrs` into an `attributes` field. The on-disk `.zarray` file has no `attributes` key — that data lives in the sibling `.zattrs` file. So `.zarray` content does not satisfy `ArrayMetadataV2`.

**What exists today (zarr-metadata):**
- `ArrayMetadataV2` ([packages/zarr-metadata/src/zarr_metadata/v2/array.py:42](packages/zarr-metadata/src/zarr_metadata/v2/array.py#L42)) — the merged in-memory form, includes `attributes`.
- `GroupMetadataV2` ([packages/zarr-metadata/src/zarr_metadata/v2/group.py:11](packages/zarr-metadata/src/zarr_metadata/v2/group.py#L11)) — the on-disk `.zgroup` form, no `attributes`. (See section 8 for the symmetric question on the group side.)
- No type for `.zarray` on-disk content.
- No type for `.zattrs` on-disk content.

**Proposal:** Introduce on-disk file types separate from in-memory shapes:

```python
# .zarray content — no attributes field (those live in sibling .zattrs)
class ArrayFileMetadataV2(TypedDict):
    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DataTypeMetadataV2
    compressor: CodecMetadataV2 | None
    fill_value: object
    order: ArrayOrderV2
    filters: tuple[CodecMetadataV2, ...] | None
    dimension_separator: NotRequired[ArrayDimensionSeparatorV2]

# .zgroup content — already matches existing GroupMetadataV2
GroupFileMetadataV2 = GroupMetadataV2  # or rename current GroupMetadataV2

# .zattrs content — arbitrary user attributes
ZAttrsV2 = Mapping[str, object]
```

Then:

```python
class ConsolidatedMetadataV2(TypedDict):
    zarr_consolidated_format: int
    metadata: Mapping[str, ArrayFileMetadataV2 | GroupFileMetadataV2 | ZAttrsV2]
```

The value-type union can't discriminate on the key suffix at the type level — consumers will narrow at runtime based on `key.endswith(".zarray")` etc. — but at least every legal value shape is representable.

**Knock-on:** This change requires a deliberate decision on what to name "the in-memory merged shape" vs "the on-disk file shape" for v2 arrays and groups. Options:

- Keep `ArrayMetadataV2` as the merged in-memory shape (most useful as the principal type in zarr-metadata's public API), introduce `ArrayFileMetadataV2` as a separate on-disk type used inside `ConsolidatedMetadataV2`.
- Or invert: `ArrayMetadataV2` becomes the on-disk shape (matching the v2 spec literally), and add `ArrayInMemoryV2` (or similar) for the merged shape. This is more spec-faithful but breaks existing consumers using the merged form.

Same question applies to `GroupMetadataV2` — see section 8.

**Status:** Breaking. Consumers using `ConsolidatedMetadataV2.metadata` values today are using a wrong type, but their code may have been working by accident (since the runtime values are dicts and TypedDicts don't enforce at runtime). A correct type catches misuse but rejects existing patterns that may have been silently incorrect.

---

## 7. GroupMetadataV2: Attributes Inconsistency

**Gap:** zarr-metadata's v2 group and array metadata treat attributes differently.

**zarr-metadata:**
- `v2/array.py`: `ArrayMetadataV2` includes `attributes: Mapping[str, object]` field
- `v2/group.py` (line 11): `GroupMetadataV2` only has `zarr_format: Literal[2]`; attributes omitted

**zarr-python** (`src/zarr/core/group.py` line 146): `ConsolidatedMetadata` stores `GroupMetadata` objects with an `attributes` field.

**Issue:** GroupMetadataV2 doesn't match the in-memory shape. The v2 spec stores `.zgroup` (just `zarr_format`) separately from `.zattrs`, so the on-disk shape is correct. But zarr-python tests pass consolidated metadata dictionaries with attributes included.

**Proposal:** Either:
1. Split `GroupMetadataV2` into on-disk (`GroupMetadataV2File`) and in-memory (`GroupMetadataV2Consolidated`) variants
2. Add an optional `attributes` field to `GroupMetadataV2` to match `ArrayMetadataV2` (less spec-correct but more practical)
3. Document the asymmetry clearly and suggest zarr-python use `GroupMetadataV2 | dict[str, object]` for consolidated shapes

**Status:** Clarification or minor extension; possibly breaking if zarr-metadata maintainers reject relaxing the spec.

---

## 8. Partial/Update TypedDicts

**Gap:** zarr-python's `replace()` and update operations benefit from `total=False` variants of metadata TypedDicts.

**Context:** zarr-python uses `dataclasses.replace()` on `ArrayV3Metadata`, `GroupMetadata`, etc., which requires partial update shapes.

**Proposal:** Add `<Name>Partial` TypedDicts (with `total=False`) to zarr-metadata for commonly updated types:
```python
# In v3/array.py
class ArrayMetadataV3Partial(TypedDict, total=False):
    """Partial/update shape of ArrayMetadataV3 for use in replace operations."""
    zarr_format: Literal[3]
    node_type: Literal["array"]
    attributes: Mapping[str, object]
    data_type: MetadataFieldV3
    fill_value: object
    codecs: tuple[MetadataFieldV3, ...]
    chunk_grid: MetadataFieldV3
    chunk_key_encoding: MetadataFieldV3
```

This would clarify which fields zarr-python is allowed to modify in-place.

**Status:** Additive; helpful but not blocking.

---

## 9. PEP 728 extra_items Support (mypy Blocker)

**Gap:** mypy v1.20's `extra_items=` support is incomplete. Annotating fields with `ExtensionFieldV3` produces false positives.

**Example:** `src/zarr/core/metadata/v3.py` defines arrays/groups with extension fields; annotating extra keys as `ExtensionFieldV3` fails mypy validation.

**Technical detail:** PEP 728 is not fully honored by mypy v1.20; the type checker doesn't enforce `extra_items=` constraints on TypedDict assignments.

**Proposal:** Document this as an upstream blocker. When mypy gains full support, zarr-python can tighten its type annotations for extension-field validation.

**Status:** Upstream issue; zarr-metadata cannot fix this unilaterally.

---

## 10. Dtype JSON Envelopes and Special Floats

**Gap:** zarr-python's dtype modules define JSON-specific types that zarr-metadata has not factored out.

**zarr-python has:**
- `src/zarr/core/dtype/npy/float.py`: per-float-dtype JSON shape (includes special-value handling)
- `src/zarr/core/dtype/common.py` (line 25–26): `JSONFloatV2`, `JSONFloatV3` (unions of numeric, special strings, hex)

**zarr-metadata has:**
- Per-dtype `<X>FillValue` types (e.g., `Float32FillValue`), which capture some of this
- `SpecialFloatStrings = Literal[...]` in some modules (implicitly via float-specific types)
- No centralized `JSONFloatV2`, `JSONFloatV3` aliases

**Proposal:** Add to `v3/data_type/common.py` (new file):
```python
SpecialFloatStrings = Literal["NaN", "Infinity", "-Infinity"]
SPECIAL_FLOAT_STRINGS: Final = ("NaN", "Infinity", "-Infinity")

JSONFloatV2 = float | SpecialFloatStrings
JSONFloatV3 = float | SpecialFloatStrings | str  # includes hex forms
```

Export these alongside per-dtype fill values.

**Status:** Additive.

---

## 11. ConsolidatedMetadataV3 Ambiguity with ArrayMetadataV3.to_dict()

**Gap:** zarr-python's `ConsolidatedMetadata.to_dict()` and `GroupMetadata.to_dict()` return shapes that don't cleanly map to zarr-metadata TypedDicts.

**zarr-python:**
- `src/zarr/core/group.py` line 150: `ConsolidatedMetadata.to_dict()` returns `{kind, must_understand, metadata: {k: v.to_dict() for ...}}`
- `src/zarr/core/group.py` line 435: `GroupMetadata.to_dict()` returns `dict[str, Any]` (uses `asdict`), which includes `node_type="group"` on v2 groups (spec violation)

**zarr-metadata:**
- `v3/consolidated.py`: `ConsolidatedMetadataV3` has `metadata: Mapping[str, ArrayMetadataV3 | GroupMetadataV3]`
- Cannot annotate `to_dict()` return as `ConsolidatedMetadataV3` because the `.to_dict()` cascade through dict-comp breaks TypedDict narrowing

**Proposal:** Document the gap and consider adding a runtime validator/assertion to catch v2 entries in v3 consolidated metadata. zarr-python maintainers should ensure `to_dict()` output is spec-compliant (no v2 entries in v3 consolidated metadata, no `node_type` on v2 group serialization).

**Status:** Requires coordination between zarr-python and zarr-metadata maintainers; may need spec clarification.

---

## 12. v2 Chunk Key Encoding and Format Detection

**Gap:** zarr-metadata lacks explicit types for zarr v2's chunk key encoding, and zarr-python uses implicit logic.

**Context:** Zarr v2 uses a fixed `.` or `/` separator; zarr-metadata models this under v3 chunk key encodings but doesn't explicitly name the v2 variant.

**zarr-metadata has:**
- `v3/chunk_key_encoding/v2.py`: `V2ChunkKeyEncodingMetadata` (describes v2-style encoding used in v3)

**Proposal:** Ensure v2-specific chunk key encoding is documented and exported clearly; consider adding `v2_chunk_key_encoding` or similar to `__init__.py` if not already there.

**Status:** Clarification/documentation.

---

## Summary of Recommendations

| Category | Priority | Effort | Breaking? | Notes |
|----------|----------|--------|-----------|-------|
| Runtime Final tuples (SHUFFLE, ENDIANNESS_STR, etc.) | High | Low | No | Unlocks direct use of zarr-metadata constants in zarr-python |
| ObjectCodecID and v2 dtype constants | Medium | Low | No | Completes dtype metadata coverage |
| Per-codec v2 configs (BloscConfigV2, etc.) | Medium | Medium | No | Matches v3 pattern; future-proofs v2 codec dispatch |
| GroupMetadataV2 attributes field | Low | Low | Maybe | Clarifies in-memory vs on-disk distinction |
| ChunkKeyEncodingParams union | Low | Low | No | Simplifies chunk key encoding union type |
| Partial TypedDicts (total=False) | Low | Low | No | Documents update-safe fields |

---

## Implementation Priority

1. **Phase 1 (High-value, Low-risk):**
   - Add runtime Final tuple constants (SHUFFLE, ENDIANNESS_STR, SPECIAL_FLOAT_STRINGS, etc.)
   - Export ObjectCodecID + OBJECT_CODEC_IDS
   - Add JSONFloatV2 and JSONFloatV3 aliases

2. **Phase 2 (Medium-risk, Medium-gain):**
   - Add per-codec v2 configuration TypedDicts
   - Resolve GroupMetadataV2 attributes asymmetry (document or extend)

3. **Phase 3 (Coordination-dependent):**
   - Work with mypy team on PEP 728 extra_items support
   - Clarify consolidation metadata spec alignment with zarr-python
   - Add Partial TypedDict variants (documentation benefit)
