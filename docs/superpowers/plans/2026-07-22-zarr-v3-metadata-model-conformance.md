# Zarr v3 Metadata Model Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the extension-agnostic v3 metadata models preserve Zarr 3.1 extension envelopes, enforce core envelope/cardinality rules, and keep raw types, parsers, guards, and serializers consistent.

**Architecture:** Keep one normalized `ZarrV3NamedConfig` for every extension point. It parses shorthand and object forms, stores opaque configuration plus `must_understand`, and applies one field-independent shorthand heuristic. The array validator supplies only core rules that vary by extension point; it never resolves an extension name or configuration.

**Tech Stack:** Python 3.11+, frozen dataclasses, `TypedDict`/PEP 728, pytest, Ruff, Pyright, and optional Pydantic v2 integration.

## Global Constraints

- Zarr v3.1 is normative; zarrs is interoperability evidence.
- Validate extension names only as strings; do not check syntax or registry membership.
- Do not interpret extension configurations, fill values, or codec kinds/order.
- Reject `must_understand=False` for data types, chunk grids, and chunk-key encodings.
- Require a non-empty codec list.
- Preserve the `consolidated_metadata: null` tolerant-read repair and never emit null.
- Observe each new behavioral test fail before changing production code.
- Commits must be conventional and include `Assisted-by: Codex:gpt-5`.

---

### Task 1: Preserve and canonically serialize extension envelopes

**Files:**
- Modify: `packages/zarr-metadata/tests/model/test_array.py`
- Modify: `packages/zarr-metadata/tests/model/test_pydantic_module.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/_common.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_validation.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/pydantic.py`

**Interfaces:**
- Produces: `ZarrV3NamedConfigJSON` with optional `must_understand: bool`.
- Produces: `ZarrV3NamedConfig(name: str, configuration: dict[str, JSONValue], must_understand: bool = True)`.
- Produces: `validate_metadata_field_v3(value: object, *, allow_must_understand_false: bool = True) -> list[ValidationProblem]`.

- [x] **Step 1: Write failing normalized-model tests**

Update the existing case tables and canonical-document expectation:

```python
ZARR_TO_JSON_CASES = [
    Expect(
        ZarrV3NamedConfig(name="regular", configuration={"chunk_shape": [1]}),
        {"name": "regular", "configuration": {"chunk_shape": [1]}},
        id="with-configuration",
    ),
    Expect(
        ZarrV3NamedConfig(name="bytes", configuration={}),
        "bytes",
        id="empty-configuration-shorthand",
    ),
    Expect(
        ZarrV3NamedConfig(name="optional", configuration={}, must_understand=False),
        {"name": "optional", "must_understand": False},
        id="false-obligation-needs-object",
    ),
]
```

Add `from_json` cases for implicit true and explicit false. Expect the default
array document to contain `"uint8"`, `("bytes",)`, and `"default"`, while its
configured regular grid remains an object.

- [x] **Step 2: Run the focused tests and verify RED**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests/model/test_array.py \
  -k 'zarr_metadata_v3 or canonical_document or explicit_false' -q
```

Expected: empty configurations remain objects, the constructor lacks
`must_understand`, or explicit false is lost.

- [x] **Step 3: Write failing envelope-validation tests**

```python
@pytest.mark.parametrize("name", ["bytes", "ANY string", "urn:example:codec"])
def test_metadata_field_accepts_any_string_name(name: str) -> None:
    assert validate_metadata_field_v3({"name": name}) == []


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_metadata_field_must_understand_must_be_boolean(value: object) -> None:
    problems = validate_metadata_field_v3({"name": "x", "must_understand": value})
    assert [(problem.loc, problem.kind) for problem in problems] == [
        (("must_understand",), "invalid_type")
    ]


def test_metadata_field_rejects_unknown_envelope_member() -> None:
    problems = validate_metadata_field_v3({"name": "x", "typo": 1})
    assert [(problem.loc, problem.kind) for problem in problems] == [
        (("typo",), "invalid_value")
    ]
```

- [x] **Step 4: Run the envelope tests and verify RED**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests/model/test_array.py \
  -k 'metadata_field_accepts or must_understand_must_be or unknown_envelope' -q
```

Expected: invalid obligation values and the unknown member are accepted before the fix.

- [x] **Step 5: Implement the raw type, validator, and normalized model**

```python
class ZarrV3NamedConfigJSON(TypedDict):
    name: str
    configuration: NotRequired[Mapping[str, JSONValue]]
    must_understand: NotRequired[bool]
```

Reject envelope keys outside `name`, `configuration`, and `must_understand`;
require a real boolean for the last member; keep all string names valid.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ZarrV3NamedConfig:
    name: str
    configuration: dict[str, JSONValue]
    must_understand: bool = True

    def to_json(self) -> ZarrV3MetadataFieldJSON:
        if not self.configuration and self.must_understand:
            return self.name
        out: ZarrV3NamedConfigJSON = {"name": self.name}
        if self.configuration:
            out["configuration"] = self.configuration
        if not self.must_understand:
            out["must_understand"] = False
        return out
```

`from_json` sets true for shorthand/absence and preserves explicit false.

- [x] **Step 6: Update and test the Pydantic serializer**

Change `ZarrV3MetadataField`'s serializer return type from `dict` to `str | dict`.

```python
def test_metadata_field_serializes_shorthand_and_false_object() -> None:
    adapter = TypeAdapter(zmp.ZarrV3MetadataField)
    assert adapter.dump_python(adapter.validate_python({"name": "bytes"})) == "bytes"
    assert adapter.dump_python(
        adapter.validate_python({"name": "optional", "must_understand": False})
    ) == {"name": "optional", "must_understand": False}
```

- [x] **Step 7: Run Task 1 tests and verify GREEN**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests/model/test_array.py \
  packages/zarr-metadata/tests/model/test_pydantic_module.py -q
```

Expected: both files pass.

- [x] **Step 8: Commit Task 1**

```bash
git add packages/zarr-metadata/src/zarr_metadata/_common.py \
  packages/zarr-metadata/src/zarr_metadata/model/_array.py \
  packages/zarr-metadata/src/zarr_metadata/model/_validation.py \
  packages/zarr-metadata/src/zarr_metadata/pydantic.py \
  packages/zarr-metadata/tests/model/test_array.py \
  packages/zarr-metadata/tests/model/test_pydantic_module.py
git commit -m "fix(zarr-metadata): preserve v3 extension obligations" \
  -m "Assisted-by: Codex:gpt-5"
```

### Task 2: Enforce context-sensitive core array rules

**Files:**
- Modify: `packages/zarr-metadata/tests/model/test_array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_validation.py`

**Interfaces:**
- Consumes: `validate_metadata_field_v3(..., allow_must_understand_false=...)`.
- Produces: rejection of forbidden false obligations and empty codecs.

- [x] **Step 1: Write failing context and cardinality tests**

```python
@pytest.mark.parametrize("field", ["codecs", "storage_transformers"])
def test_optional_extension_points_allow_must_understand_false(field: str) -> None:
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc[field] = ({"name": "optional", "must_understand": False},)
    assert validate_array_metadata_v3(doc) == []


@pytest.mark.parametrize("field", ["data_type", "chunk_grid", "chunk_key_encoding"])
def test_required_extension_points_reject_must_understand_false(field: str) -> None:
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc[field] = {"name": "optional", "must_understand": False}
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(doc)] == [
        ((field, "must_understand"), "invalid_value")
    ]


def test_v3_codecs_cannot_be_empty() -> None:
    doc = dict(ZarrV3ArrayMetadata.create_default().to_json())
    doc["codecs"] = ()
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(doc)] == [
        (("codecs",), "invalid_value")
    ]
```

- [x] **Step 2: Run Task 2 tests and verify RED**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests/model/test_array.py \
  -k 'optional_extension_points or required_extension_points or codecs_cannot' -q
```

Expected: required points and empty codecs are accepted before the fix.

- [x] **Step 3: Implement the context rules**

Call `validate_metadata_field_v3(..., allow_must_understand_false=False)` for
`data_type`, `chunk_grid`, and `chunk_key_encoding`. Leave codecs and storage
transformers at the default. After establishing that `codecs` is a sequence,
add one field-level `invalid_value` problem when it is empty.

- [x] **Step 4: Run the complete array model tests and verify GREEN**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests/model/test_array.py -q
```

Expected: the file passes.

- [x] **Step 5: Commit Task 2**

```bash
git add packages/zarr-metadata/src/zarr_metadata/model/_validation.py \
  packages/zarr-metadata/tests/model/test_array.py
git commit -m "fix(zarr-metadata): enforce v3 core extension rules" \
  -m "Assisted-by: Codex:gpt-5"
```

### Task 3: Align additional-field types, guards, and models

**Files:**
- Modify: `packages/zarr-metadata/tests/model/test_array.py`
- Modify: `packages/zarr-metadata/tests/model/test_group.py`
- Modify: `packages/zarr-metadata/tests/test_partial_equivalence.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/group.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_group.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_validation.py`

**Interfaces:**
- Produces: public `ZarrV3ExtensionField` alias for arbitrary `JSONValue`.
- Produces: `extra_fields: dict[str, JSONValue]` and matching guards/parsers.

- [x] **Step 1: Write failing parser/guard agreement tests**

```python
def test_v3_scalar_extra_field_agrees_across_parser_guard_and_model() -> None:
    raw = dict(ZarrV3ArrayMetadata.create_default().to_json()) | {"ext": 1}
    parsed = parse_array_metadata_v3(raw)
    assert is_array_metadata_v3(parsed)
    model = ZarrV3ArrayMetadata.from_json(raw)
    assert model.extra_fields == {"ext": 1}
    assert model.must_understand_fields == {"ext": 1}


def test_group_v3_scalar_extra_field_roundtrips_as_must_understand() -> None:
    raw = {"zarr_format": 3, "node_type": "group", "ext": [1, 2]}
    model = ZarrV3GroupMetadata.from_json(raw)
    assert model.to_json()["ext"] == (1, 2)
    assert model.must_understand_fields == {"ext": (1, 2)}
```

- [x] **Step 2: Run the agreement tests and verify RED**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests/model/test_array.py \
  packages/zarr-metadata/tests/model/test_group.py -k 'scalar_extra_field' -q
```

Expected: the canonical array guard rejects the parsed scalar extra field.

- [x] **Step 3: Correct raw/model annotations and the guard**

Replace the object-shaped `ZarrV3ExtensionField` TypedDict with a public alias to
`JSONValue`; use it as `extra_items` on full and partial array/group types.
Update model partials, dataclass fields, helper signatures, and casts to
`dict[str, JSONValue]`. Remove the canonical array guard's requirement that
every non-standard value be a dict; JSON validation already establishes the
domain and `arrays_to_tuples` establishes canonical sequences.

- [x] **Step 4: Run selected integration tests and verify GREEN**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests/model/test_array.py \
  packages/zarr-metadata/tests/model/test_group.py \
  packages/zarr-metadata/tests/test_partial_equivalence.py \
  packages/zarr-metadata/tests/test_public_api.py -q
```

Expected: all selected files pass and `ZarrV3ExtensionField` remains exported.

- [x] **Step 5: Commit Task 3**

```bash
git add packages/zarr-metadata/src/zarr_metadata/v3/array.py \
  packages/zarr-metadata/src/zarr_metadata/v3/group.py \
  packages/zarr-metadata/src/zarr_metadata/model/_array.py \
  packages/zarr-metadata/src/zarr_metadata/model/_group.py \
  packages/zarr-metadata/src/zarr_metadata/model/_validation.py \
  packages/zarr-metadata/tests/model/test_array.py \
  packages/zarr-metadata/tests/model/test_group.py \
  packages/zarr-metadata/tests/test_partial_equivalence.py
git commit -m "fix(zarr-metadata): align v3 additional field types" \
  -m "Assisted-by: Codex:gpt-5"
```

### Task 4: Documentation reconciliation and full verification

**Files:**
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/_common.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/consolidated.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/__init__.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/model/_group.py`
- Modify: `docs/superpowers/specs/2026-07-22-zarr-v3-metadata-model-conformance-design.md`
- Create: `docs/superpowers/plans/2026-07-22-zarr-v3-metadata-model-conformance.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: documentation matching semantic normalization and the extension-agnostic boundary.

- [x] **Step 1: Reconcile documentation**

Remove object-only and v3.0-over-v3.1 claims. Document the exact heuristic:

```text
empty configuration + must_understand true -> shorthand string
otherwise -> object containing name and non-default members
```

Document arbitrary JSON additional fields, type-only name validation, and the
known non-core consolidated extension. Retain the historical null repair.

- [x] **Step 2: Run the complete package suite**

```bash
packages/zarr-metadata/.venv/bin/pytest packages/zarr-metadata/tests -q
```

Expected: zero failures.

- [x] **Step 3: Run format, lint, typing, and whitespace checks**

```bash
.venv/bin/ruff format --check packages/zarr-metadata
.venv/bin/ruff check packages/zarr-metadata
cd packages/zarr-metadata
uvx --from pyright==1.1.404 pyright --pythonpath .venv/bin/python src
git diff --check
```

Expected: every command exits zero with no diagnostics.

- [x] **Step 4: Audit scope against the final diff**

Confirm there is no registry lookup, name-syntax validation, extension
configuration interpretation, fill-value semantics, or codec-kind/order
validation. Confirm `consolidated_metadata: null` is still accepted on read
and never emitted.

- [x] **Step 5: Commit docs and final integration adjustments**

```bash
git add -f docs/superpowers/specs/2026-07-22-zarr-v3-metadata-model-conformance-design.md \
  docs/superpowers/plans/2026-07-22-zarr-v3-metadata-model-conformance.md
git add packages/zarr-metadata
git commit -m "docs(zarr-metadata): document v3 model conformance" \
  -m "Assisted-by: Codex:gpt-5"
```

- [x] **Step 6: Verify committed state**

```bash
git status --short --branch
git log -4 --format='%h %s%n%(trailers:key=Assisted-by,valueonly)'
```

Expected: only the user's pre-existing untracked files remain; every new
commit has a conventional subject and `Assisted-by: Codex:gpt-5`.
