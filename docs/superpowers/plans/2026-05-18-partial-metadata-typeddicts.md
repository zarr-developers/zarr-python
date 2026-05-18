# Partial Metadata TypedDicts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ArrayMetadataV3Partial`, `ArrayMetadataV2Partial`, and `GroupMetadataV3Partial` TypedDicts to the `zarr-metadata` package as siblings of the existing full TypedDicts, with an equivalence test that prevents drift.

**Architecture:** Each `*Partial` is a hand-written TypedDict declared with `total=False` and the same `extra_items=` setting as its full counterpart. Field annotations are duplicated verbatim; a test in the same package asserts `__annotations__` and `__extra_items__` are equal between each pair, so a field added to the full type without a matching addition to the partial fails CI. `GroupMetadataV2` has no required fields beyond `zarr_format` plus an already-`NotRequired` `attributes`, but for symmetry and to express "no required fields at all" we add `GroupMetadataV2Partial` too.

**Tech Stack:** Python 3.11+, `typing_extensions.TypedDict` (PEP 728 `extra_items=`), pytest.

**Scope notes:**
- This plan is restricted to `packages/zarr-metadata/`. It does not touch `src/zarr/` or zarr-python's test suite.
- Consuming the new `*Partial` types in zarr-python's test fixtures (`tests/test_metadata/conftest.py`, `tests/test_metadata/test_v3.py`, `tests/test_metadata/test_consolidated.py`) is a follow-up plan that lands after the in-flight PR which brings `zarr_metadata.*` types into `src/zarr/`.
- `ConsolidatedMetadataV2`/`V3` partials are out of scope — current evidence doesn't show a need; revisit if one appears.

---

## File Structure

**Modify:**
- `packages/zarr-metadata/src/zarr_metadata/v3/array.py` — add `ArrayMetadataV3Partial` below `ArrayMetadataV3`.
- `packages/zarr-metadata/src/zarr_metadata/v3/group.py` — add `GroupMetadataV3Partial` below `GroupMetadataV3`.
- `packages/zarr-metadata/src/zarr_metadata/v2/array.py` — add `ArrayMetadataV2Partial` below `ArrayMetadataV2`.
- `packages/zarr-metadata/src/zarr_metadata/v2/group.py` — add `GroupMetadataV2Partial` below `GroupMetadataV2`.
- `packages/zarr-metadata/src/zarr_metadata/__init__.py` — re-export the four new symbols and add them to `__all__`.

**Create:**
- `packages/zarr-metadata/tests/test_partial_equivalence.py` — single equivalence test that loops over the four pairs.

Each `*Partial` lives in the same file as its full counterpart so the two stay visually adjacent during edits. The equivalence test is centralized rather than co-located so that adding a fifth pair later requires touching one test file, not four.

---

## Task 1: ArrayMetadataV3Partial

**Files:**
- Test: `packages/zarr-metadata/tests/test_partial_equivalence.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/__init__.py`

### - [ ] Step 1: Write the failing equivalence test

Create `packages/zarr-metadata/tests/test_partial_equivalence.py`:

```python
"""Drift-prevention tests for Partial* TypedDict variants.

Each *Partial TypedDict in the package must declare the same fields
(with the same annotations) and the same extra_items setting as its
full counterpart. The only intentional difference is total=False
(i.e. every field becomes NotRequired). This test enforces that
invariant so adding a field to the full type without mirroring it
on the partial fails CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata.v3.array import ArrayMetadataV3, ArrayMetadataV3Partial

# (full, partial) pairs to check. Add new pairs here as more are introduced.
PAIRS: list[tuple[type, type]] = [
    (ArrayMetadataV3, ArrayMetadataV3Partial),
]


@pytest.mark.parametrize(("full", "partial"), PAIRS, ids=lambda p: p.__name__)
def test_partial_matches_full(full: Any, partial: Any) -> None:
    """Partial TypedDict has identical fields and extra_items, only total differs."""
    assert full.__annotations__ == partial.__annotations__, (
        f"{partial.__name__} fields drifted from {full.__name__}: "
        f"full={set(full.__annotations__)}, partial={set(partial.__annotations__)}"
    )
    assert getattr(full, "__extra_items__", None) == getattr(
        partial, "__extra_items__", None
    ), f"{partial.__name__} extra_items differs from {full.__name__}"
    assert partial.__total__ is False, (
        f"{partial.__name__} must be declared with total=False"
    )
    assert full.__total__ is True, (
        f"{full.__name__} must be declared with total=True (default)"
    )
```

### - [ ] Step 2: Run the test to verify it fails

Run from the repo root:

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: `ImportError: cannot import name 'ArrayMetadataV3Partial' from 'zarr_metadata.v3.array'`.

### - [ ] Step 3: Add `ArrayMetadataV3Partial`

In `packages/zarr-metadata/src/zarr_metadata/v3/array.py`, append after the existing `ArrayMetadataV3` class definition (after line 62, before `__all__`):

```python
class ArrayMetadataV3Partial(TypedDict, total=False, extra_items=ExtensionFieldV3):  # type: ignore[call-arg]
    """
    Partial form of `ArrayMetadataV3`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `ArrayMetadataV3` exactly.
    The only difference is `total=False`, which makes every key optional
    at the type level.

    Use this when typing dicts that intentionally hold a subset of a
    complete v3 array metadata document — e.g. test fixtures that
    override only a few fields of a base template, or callers that
    build a fragment to be merged into a complete document elsewhere.

    Drift between this type and `ArrayMetadataV3` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["array"]
    data_type: MetadataFieldV3
    shape: tuple[int, ...]
    chunk_grid: MetadataFieldV3
    chunk_key_encoding: MetadataFieldV3
    fill_value: object
    codecs: tuple[MetadataFieldV3, ...]
    attributes: NotRequired[Mapping[str, object]]
    storage_transformers: NotRequired[tuple[MetadataFieldV3, ...]]
    dimension_names: NotRequired[tuple[str | None, ...]]
```

Note: the `NotRequired[...]` wrappers on the last three fields are intentionally identical to the full `ArrayMetadataV3`. `__annotations__` preserves those wrappers verbatim, so keeping them byte-identical lets the equivalence test be a plain `==` on annotation dicts. PEP 655 explicitly permits `NotRequired` inside a `total=False` TypedDict — it's redundant at the type level (the field was already optional) but it's the right way to keep the two definitions textually mirrored.

Also update the file's `__all__` to include `"ArrayMetadataV3Partial"`:

```python
__all__ = [
    "ArrayMetadataV3",
    "ArrayMetadataV3Partial",
    "ExtensionFieldV3",
]
```

### - [ ] Step 4: Re-export from the package root

In `packages/zarr-metadata/src/zarr_metadata/__init__.py`:

Change the v3 array import line from:

```python
from zarr_metadata.v3.array import ArrayMetadataV3, ExtensionFieldV3
```

to:

```python
from zarr_metadata.v3.array import ArrayMetadataV3, ArrayMetadataV3Partial, ExtensionFieldV3
```

And insert `"ArrayMetadataV3Partial"` into `__all__` in alphabetical order (between `"ArrayMetadataV3"` and `"ArrayOrderV2"`).

### - [ ] Step 5: Run the test to verify it passes

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: 1 passed.

### - [ ] Step 6: Run mypy via prek to catch type-level issues

```bash
prek run --all-files mypy
```

Expected: no new errors. (Per repo convention, do not run `uvx mypy` ad-hoc.)

### - [ ] Step 7: Commit

```bash
git add packages/zarr-metadata/src/zarr_metadata/v3/array.py \
        packages/zarr-metadata/src/zarr_metadata/__init__.py \
        packages/zarr-metadata/tests/test_partial_equivalence.py
git commit -m "feat(zarr-metadata): add ArrayMetadataV3Partial

Sibling TypedDict to ArrayMetadataV3 with total=False, intended for
typing dicts that intentionally hold a subset of a complete v3 array
metadata document (test fixtures, fragment templates). Drift between
the two is prevented by a new equivalence test."
```

---

## Task 2: GroupMetadataV3Partial

**Files:**
- Modify: `packages/zarr-metadata/src/zarr_metadata/v3/group.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/__init__.py`
- Modify: `packages/zarr-metadata/tests/test_partial_equivalence.py`

### - [ ] Step 1: Extend the equivalence test

In `packages/zarr-metadata/tests/test_partial_equivalence.py`, add an import:

```python
from zarr_metadata.v3.group import GroupMetadataV3, GroupMetadataV3Partial
```

And extend the `PAIRS` list:

```python
PAIRS: list[tuple[type, type]] = [
    (ArrayMetadataV3, ArrayMetadataV3Partial),
    (GroupMetadataV3, GroupMetadataV3Partial),
]
```

### - [ ] Step 2: Run the test to verify it fails

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: `ImportError: cannot import name 'GroupMetadataV3Partial' from 'zarr_metadata.v3.group'`.

### - [ ] Step 3: Add `GroupMetadataV3Partial`

In `packages/zarr-metadata/src/zarr_metadata/v3/group.py`, append after the existing `GroupMetadataV3` class (after line 25, before `__all__`):

```python
class GroupMetadataV3Partial(TypedDict, total=False, extra_items=ExtensionFieldV3):  # type: ignore[call-arg]
    """
    Partial form of `GroupMetadataV3`: every field is `NotRequired`.

    Field annotations and `extra_items=` mirror `GroupMetadataV3` exactly.
    The only difference is `total=False`. See `ArrayMetadataV3Partial`
    for the rationale.

    Drift between this type and `GroupMetadataV3` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: NotRequired[Mapping[str, object]]
```

Update `__all__` in the same file:

```python
__all__ = [
    "GroupMetadataV3",
    "GroupMetadataV3Partial",
]
```

### - [ ] Step 4: Re-export from the package root

In `packages/zarr-metadata/src/zarr_metadata/__init__.py`:

Change the v3 group import from:

```python
from zarr_metadata.v3.group import GroupMetadataV3
```

to:

```python
from zarr_metadata.v3.group import GroupMetadataV3, GroupMetadataV3Partial
```

And insert `"GroupMetadataV3Partial"` into `__all__` after `"GroupMetadataV3"`.

### - [ ] Step 5: Run the test to verify it passes

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: 2 passed.

### - [ ] Step 6: Run mypy

```bash
prek run --all-files mypy
```

Expected: no new errors.

### - [ ] Step 7: Commit

```bash
git add packages/zarr-metadata/src/zarr_metadata/v3/group.py \
        packages/zarr-metadata/src/zarr_metadata/__init__.py \
        packages/zarr-metadata/tests/test_partial_equivalence.py
git commit -m "feat(zarr-metadata): add GroupMetadataV3Partial

Sibling TypedDict to GroupMetadataV3 with total=False."
```

---

## Task 3: ArrayMetadataV2Partial

**Files:**
- Modify: `packages/zarr-metadata/src/zarr_metadata/v2/array.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/__init__.py`
- Modify: `packages/zarr-metadata/tests/test_partial_equivalence.py`

### - [ ] Step 1: Extend the equivalence test

In `packages/zarr-metadata/tests/test_partial_equivalence.py`, add an import:

```python
from zarr_metadata.v2.array import ArrayMetadataV2, ArrayMetadataV2Partial
```

And extend `PAIRS`:

```python
PAIRS: list[tuple[type, type]] = [
    (ArrayMetadataV3, ArrayMetadataV3Partial),
    (GroupMetadataV3, GroupMetadataV3Partial),
    (ArrayMetadataV2, ArrayMetadataV2Partial),
]
```

### - [ ] Step 2: Run the test to verify it fails

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: `ImportError: cannot import name 'ArrayMetadataV2Partial'`.

### - [ ] Step 3: Add `ArrayMetadataV2Partial`

In `packages/zarr-metadata/src/zarr_metadata/v2/array.py`, append after the existing `ArrayMetadataV2` class (after line 98, before `__all__`):

```python
class ArrayMetadataV2Partial(TypedDict, total=False):
    """
    Partial form of `ArrayMetadataV2`: every field is `NotRequired`.

    Field annotations mirror `ArrayMetadataV2` exactly. The only
    difference is `total=False`. See `ArrayMetadataV3Partial` for
    the rationale.

    Note: v2 array metadata has no `extra_items` setting (extra keys
    are not part of the v2 spec), so this partial inherits the same
    closed shape.

    Drift between this type and `ArrayMetadataV2` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[2]
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DataTypeMetadataV2
    compressor: CodecMetadataV2 | None
    fill_value: object
    order: ArrayOrderV2
    filters: tuple[CodecMetadataV2, ...] | None
    dimension_separator: NotRequired[ArrayDimensionSeparatorV2]
    attributes: NotRequired[Mapping[str, object]]
```

Update `__all__` in the same file to insert `"ArrayMetadataV2Partial"` after `"ArrayMetadataV2"`:

```python
__all__ = [
    "ARRAY_DIMENSION_SEPARATOR_V2",
    "ARRAY_ORDER_V2",
    "ArrayDimensionSeparatorV2",
    "ArrayMetadataV2",
    "ArrayMetadataV2Partial",
    "ArrayOrderV2",
    "DataTypeMetadataV2",
    "ZArrayMetadata",
]
```

### - [ ] Step 4: Re-export from the package root

In `packages/zarr-metadata/src/zarr_metadata/__init__.py`, update the v2 array import:

```python
from zarr_metadata.v2.array import (
    ArrayDimensionSeparatorV2,
    ArrayMetadataV2,
    ArrayMetadataV2Partial,
    ArrayOrderV2,
    DataTypeMetadataV2,
    ZArrayMetadata,
)
```

And insert `"ArrayMetadataV2Partial"` into `__all__` after `"ArrayMetadataV2"`.

### - [ ] Step 5: Run the test to verify it passes

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: 3 passed.

### - [ ] Step 6: Run mypy

```bash
prek run --all-files mypy
```

Expected: no new errors.

### - [ ] Step 7: Commit

```bash
git add packages/zarr-metadata/src/zarr_metadata/v2/array.py \
        packages/zarr-metadata/src/zarr_metadata/__init__.py \
        packages/zarr-metadata/tests/test_partial_equivalence.py
git commit -m "feat(zarr-metadata): add ArrayMetadataV2Partial

Sibling TypedDict to ArrayMetadataV2 with total=False."
```

---

## Task 4: GroupMetadataV2Partial

**Files:**
- Modify: `packages/zarr-metadata/src/zarr_metadata/v2/group.py`
- Modify: `packages/zarr-metadata/src/zarr_metadata/__init__.py`
- Modify: `packages/zarr-metadata/tests/test_partial_equivalence.py`

### - [ ] Step 1: Extend the equivalence test

In `packages/zarr-metadata/tests/test_partial_equivalence.py`, add an import:

```python
from zarr_metadata.v2.group import GroupMetadataV2, GroupMetadataV2Partial
```

And extend `PAIRS`:

```python
PAIRS: list[tuple[type, type]] = [
    (ArrayMetadataV3, ArrayMetadataV3Partial),
    (GroupMetadataV3, GroupMetadataV3Partial),
    (ArrayMetadataV2, ArrayMetadataV2Partial),
    (GroupMetadataV2, GroupMetadataV2Partial),
]
```

### - [ ] Step 2: Run the test to verify it fails

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: `ImportError: cannot import name 'GroupMetadataV2Partial'`.

### - [ ] Step 3: Add `GroupMetadataV2Partial`

In `packages/zarr-metadata/src/zarr_metadata/v2/group.py`, append after the existing `GroupMetadataV2` class (after line 42, before `__all__`):

```python
class GroupMetadataV2Partial(TypedDict, total=False):
    """
    Partial form of `GroupMetadataV2`: every field is `NotRequired`.

    Field annotations mirror `GroupMetadataV2` exactly. The only
    difference is `total=False`. Provided for symmetry with the other
    `*Partial` types — `GroupMetadataV2` already has only one required
    field (`zarr_format`), so this partial differs from the full type
    only in that `zarr_format` becomes optional.

    Drift between this type and `GroupMetadataV2` is prevented by
    `tests/test_partial_equivalence.py`.
    """

    zarr_format: Literal[2]
    attributes: NotRequired[Mapping[str, object]]
```

Update `__all__` in the same file:

```python
__all__ = [
    "GroupMetadataV2",
    "GroupMetadataV2Partial",
    "ZGroupMetadata",
]
```

### - [ ] Step 4: Re-export from the package root

In `packages/zarr-metadata/src/zarr_metadata/__init__.py`, update the v2 group import:

```python
from zarr_metadata.v2.group import GroupMetadataV2, GroupMetadataV2Partial, ZGroupMetadata
```

And insert `"GroupMetadataV2Partial"` into `__all__` after `"GroupMetadataV2"`.

### - [ ] Step 5: Run the test to verify it passes

```bash
uv run --directory packages/zarr-metadata --with . pytest tests/test_partial_equivalence.py -v
```

Expected: 4 passed.

### - [ ] Step 6: Run mypy

```bash
prek run --all-files mypy
```

Expected: no new errors.

### - [ ] Step 7: Commit

```bash
git add packages/zarr-metadata/src/zarr_metadata/v2/group.py \
        packages/zarr-metadata/src/zarr_metadata/__init__.py \
        packages/zarr-metadata/tests/test_partial_equivalence.py
git commit -m "feat(zarr-metadata): add GroupMetadataV2Partial

Sibling TypedDict to GroupMetadataV2 with total=False. Added for
symmetry with the other v2/v3 *Partial types; the only practical
difference from GroupMetadataV2 is that zarr_format becomes optional."
```

---

## Task 5: Final verification

**Files:** none — runs only.

### - [ ] Step 1: Run the full zarr-metadata test suite

```bash
uv run --directory packages/zarr-metadata --with . pytest -v
```

Expected: all tests pass, including the four equivalence cases.

### - [ ] Step 2: Confirm the partials are importable from the package root

```bash
uv run --directory packages/zarr-metadata --with . python -c "
from zarr_metadata import (
    ArrayMetadataV3, ArrayMetadataV3Partial,
    ArrayMetadataV2, ArrayMetadataV2Partial,
    GroupMetadataV3, GroupMetadataV3Partial,
    GroupMetadataV2, GroupMetadataV2Partial,
)
print('OK')
"
```

Expected: prints `OK`.

### - [ ] Step 3: Run prek across all files

```bash
prek run --all-files
```

Expected: all hooks pass. (Single-file prek runs are unreliable for mypy here — always use `--all-files`.)

### - [ ] Step 4: Confirm no zarr-python tests regressed

```bash
uv run pytest tests/ -x --timeout=30 -q
```

Expected: no new failures attributable to this change. (None should appear since `src/zarr/` was not modified.)

---

## Follow-up (out of scope for this plan)

After the in-flight PR that adopts `zarr_metadata.*` types in zarr-python merges, write a separate plan to:

1. Replace `dict[str, Any]` / `dict[str, JSON]` annotations on `Expect.input` and `ExpectFail.input` in `tests/test_metadata/test_v3.py` with `ArrayMetadataV3Partial`.
2. Retype the `array_metadata` template in `tests/test_metadata/test_consolidated.py:146` from `dict[str, JSON]` to `ArrayMetadataV3Partial` and tighten the surrounding spread call sites.
3. Change the `**overrides: Any` parameter of `minimal_metadata_dict_v3` in `tests/test_metadata/conftest.py` to `**overrides: Unpack[ArrayMetadataV3Partial]` and remove the three `# type: ignore[typeddict-item]` comments at lines 34, 38, 40.

Those changes depend on `zarr_metadata` being importable from zarr-python's test environment, which the upstream PR provides.
