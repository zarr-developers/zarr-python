# Indexing test suite cleanup — design

Date: 2026-05-22
Branch: `indexing-test-cleanup`
Target file: `tests/test_indexing.py` (2177 lines, 50 test functions)

## Goals

Three goals, in priority order as confirmed with the user:

1. **Per-test isolation** — each test verifies one behavior. A failure should
   point at a named case, not a 20-iteration loop inside a helper.
2. **Faster** — shrink the oversized arrays and cut redundant loop iterations.
   Speed comes purely from smaller work per test (see "Speed model" below),
   not from any matrix change.
3. **Clarity / consolidation** — remove duplicate coverage, replace loop +
   helper-function patterns with `@pytest.mark.parametrize`, give every test a
   docstring stating the behavior it verifies.

Scope: **whole file**. Already-parametrized tests get swept for consistency
(docstrings, shared helpers, array sizes) alongside the loop-based ones.

## Background / current state

The file is a mix of two eras:

- **Ported-from-v2 tests** (the slow ones): build a large array, set
  `np.random.seed(42)`, then loop `for p in 0.5, 0.1, 0.01` generating random
  boolean/integer index arrays at several sparsity levels, and call a private
  helper (`_test_get_orthogonal_selection_3d`, etc.) that itself loops over
  7–20 hardcoded selection tuples. One test function exercises dozens of
  selections with bundled assertions.
  - Example arrays: `np.arange(32400).reshape(120, 30, 9)` (3d orthogonal),
    `np.arange(5400).reshape(600, 9)` (2d), `np.arange(1050)` (1d bool).
- **Modern tests**: `@pytest.mark.parametrize` with explicit selection lists,
  e.g. `test_indexing_equals_numpy`, `test_orthogonal_bool_indexing_like_numpy_ix`.

### Key facts that constrain the design

- `tests/test_indexing.py` defines its **own** `store` fixture (line 42–44):
  `StorePath(await MemoryStore.open())`. It does **not** use the conftest
  matrix-parametrized `store` fixture. Therefore every test runs **once** —
  there is no store/format/codec amplification for this file.
- Default `addopts` (pyproject.toml:428) does not include `-n auto`; xdist is
  available but opt-in. Tests run single-process by default.
- Local full-file runtime: ~3.5 s in pytest, ~4 s wall. Slowest tests are
  exactly the loop-heavy orthogonal ones (`test_set_orthogonal_selection_3d`
  0.92 s, `test_get_orthogonal_selection_1d_bool` 0.79 s,
  `test_set_orthogonal_selection_2d` 0.54 s, `test_set_orthogonal_selection_1d`
  0.41 s).

### Speed model

Runtime per test ≈ array construction + encode (`z[()] = a`) + per-selection
decode/encode. The large arrays dominate. Halving each dimension of
`(120, 30, 9)` and keeping a chunk grid of (≈3 chunks per axis with a partial
edge chunk) cuts the encoded volume by ~8× while preserving every structural
property the tests rely on (multi-chunk spans, partial edge chunks,
cross-chunk selections, negative/wraparound indices).

## Approach (selected: A — surgical consolidation, applied file-wide)

For each test family:

1. **Shrink arrays** under a hard rule (confirmed with user): every indexed
   axis spans **≥3 chunks**, and at least one axis has a **partial (non-full)
   edge chunk**. Smallest shapes meeting that:
   - 1d: `arange(30)`, chunks `(7,)` → 5 chunks, last (size 2) partial.
   - 2d: `arange(60).reshape(12, 5)`, chunks `(5, 2)` → 3×3 chunks; axis-0
     edge (size 2) and axis-1 edge (size 1) partial.
   - 3d: `arange(420).reshape(7, 6, 10)`, chunks `(3, 2, 4)` → 3×3×3 chunks;
     partial edges on every axis.
   Exact numbers may shift slightly during implementation so the hand-picked
   selections stay in-bounds and meaningful, but the ≥3-chunks / partial-edge
   rule is fixed.

2. **Replace `np.random` selections with hand-picked deterministic ones**
   (user decision). For each axis kind we keep explicit cases that name what
   they cover:
   - boolean mask: empty (all False), full (all True), alternating, single-True,
     a sparse hand-chosen mask.
   - integer array: sorted, unsorted, with-duplicates, with-wraparound
     (negative), single-element.
   The 0.5 / 0.1 / 0.01 sparsity loop collapses into the "alternating" and
   "sparse" cases — the density sweep was fuzzing, not targeted coverage, and
   the user de-prioritized RNG breadth.

3. **Convert loop-over-selections helpers to `@pytest.mark.parametrize`.**
   The private `_test_*` helpers that loop over selection lists become a
   `selection` parameter on the public test. The shared
   `_test_get_orthogonal_selection` / `_test_set_orthogonal_selection`
   oracle (compare zarr result to numpy via `oindex`/`oindex_set`) is **kept**
   as a small assertion helper — it is the right abstraction, just called once
   per parametrized case instead of in a loop.

4. **Docstrings** on every test stating the behavior verified (per project
   convention / memory `feedback_test_docstrings`).

5. **Preserve error-path tests as their own named tests** (per memory
   `feedback_test_structure`: one test per failure mode). The `IndexError`
   blocks currently bundled at the end of the get/set tests (too-short mask,
   too-long mask, too-many-dims, out-of-bounds) become a `test_*_raises`
   parametrized over `ExpectFail` cases, using `pytest.raises(case.exception,
   match=case.msg)`.

## Prerequisite: deduplicate the two `Expect` dataclass pairs

The repo currently has **two divergent** test-case dataclass pairs. The new
indexing tests should use one canonical pair, so this PR unifies them first.

| | `tests/conftest.py` (canonical) | `tests/test_codecs/conftest.py` (to delete) |
|---|---|---|
| success | `Expect[TIn, TOut]`: `input`, `output`, `id`; not frozen | `Expect[TIn, TOut]`: `input`, `expected`; frozen |
| failure | `ExpectFail[TIn]`: `input`, `exception`, `id`, `msg` | `ExpectErr[TIn]`: `input`, `msg`, `exception_cls` |

**Decision:** keep the `tests/conftest.py` pair (`Expect` + `ExpectFail`) as
the single source of truth, because it carries `id` — the id lives with the
case, cannot drift out of sync with a separate `ids=[...]` list, and survives
reordering. Add `frozen=True` to both (the codecs version's one good idea;
these are value objects). Delete the definitions in
`tests/test_codecs/conftest.py` (that file contains only these two classes).

**Migration of the three codecs/chunk-grid consumers**
(`tests/test_chunk_grids.py`, `tests/test_codecs/test_cast_value.py`,
`tests/test_codecs/test_scale_offset.py`):

1. Change imports to `from tests.conftest import Expect, ExpectFail`.
2. Rename `expected=` → `output=` at every `Expect(...)` call site.
3. Replace `ExpectErr` → `ExpectFail`, renaming `exception_cls=` → `exception=`.
4. Add `id=` to every case, sourced from the existing `ids=[...]` list that the
   parametrize call passes positionally, then replace `ids=[...]` with
   `ids=lambda c: c.id`. Where a parametrize block has no `ids=` list today,
   synthesize concise ids.
5. Run the affected suites green before touching indexing tests.

This is a mechanical but real change (~40 call sites). It is a true
prerequisite: doing it first means the indexing rewrite imports the final,
stable `Expect`/`ExpectFail` and we never write against a soon-to-change shape.

### Shared helpers introduced

- **Use the conftest `Expect[TIn, TOut]` / `ExpectFail[TIn]` dataclasses** as
  the selection-table mechanism, matching the established idiom in
  `tests/test_metadata/test_v3.py`:
  - Selection cases: `Expect[Selection, None]` where `input` is the selection
    and `id` names the case (e.g. `"alternating-mask"`, `"wraparound-int"`).
    `output` is unused for the oracle-style tests (the numpy result is computed,
    not hardcoded) and set to `None`; the oracle helper does the comparison.
  - Error cases: `ExpectFail[Selection]` carrying `input` (the bad selection),
    `exception` (`IndexError`), `id`, and `msg` (regex for `pytest.raises`).
  - Parametrize with `ids=lambda c: c.id` so `pytest -k <case-id>` selects a
    single named case — this is the readable-id payoff that plain tuple
    parametrize lacks.
  - Module-level case lists (`_ORTHO_1D_CASES`, `_ORTHO_BAD_1D_CASES`, etc.)
    are shared across get/set tests of the same dimensionality so 1d/2d/3d
    don't each redefine "alternating mask".
  - Field is `Expect.output` (the canonical `tests/conftest.py` name). The
    `expected=` spelling came from the now-deleted codecs copy (see the dedup
    section above).
- Keep `zarr_array_from_numpy_array`; it is already the right builder.
- Keep the `_test_get/set_orthogonal_selection` zarr-vs-numpy oracle helpers;
  they consume one `Expect.input` per parametrized case instead of looping.

### Things explicitly NOT changed

- The local `store` fixture stays MemoryStore-only. Widening it to the conftest
  matrix would multiply runtime against goal 2; out of scope.
- `xfail`/`skip` markers (structured-field tests, repeated-index test) stay as-is.
- Regression tests tied to specific GH issues (`test_set_item_1d_last_two_chunks`,
  `test_indexing_with_zarr_array`, `test_vectorized_indexing_incompatible_shape`,
  `test_zero_sized_chunks`) keep their concrete reproducing values — shrinking
  them would weaken the regression. Only docstrings added.
- `CountingDict` / `test_accessed_chunks` logic (verifies which chunks are
  touched) keeps shapes that produce a meaningful access pattern.

## Rewrite inventory

Loop/helper-based (primary rewrite targets):
`test_get_basic_selection_1d`, `test_get_basic_selection_2d`,
`test_get_orthogonal_selection_1d_bool`, `test_get_orthogonal_selection_1d_int`,
`test_get_orthogonal_selection_2d`, `test_get_orthogonal_selection_3d`,
`test_set_orthogonal_selection_1d/2d/3d`,
`test_get_coordinate_selection_1d/2d`, `test_set_coordinate_selection_1d/2d`,
`test_get_block_selection_1d/2d`, `test_set_block_selection_1d/2d`,
`test_get_mask_selection_1d/2d`, `test_set_mask_selection_1d/2d`,
`test_get_selection_out`.

Already parametrized (sweep for consistency + array size only):
`test_get_basic_selection_0d`, `test_set_basic_selection_0d`,
`test_indexing_equals_numpy`, `test_orthogonal_bool_indexing_like_numpy_ix`,
the `*_fallback_*` family, `test_iter_grid`, `test_iter_regions`.

Leave essentially as-is (docstring only):
the GH-regression tests and `xfail`/`skip` tests listed above,
`test_normalize_integer_selection`, `test_replace_ellipsis`,
`test_iter_grid_invalid`, `test_iter_chunk_regions`, `TestAsync`.

## Testing / verification

- After the `Expect` dedup (step 0): the migrated suites must stay green —
  `uv run --frozen pytest tests/test_chunk_grids.py tests/test_codecs/test_cast_value.py tests/test_codecs/test_scale_offset.py tests/test_metadata/test_v3.py -q`.
- After each indexing family rewrite:
  `uv run --frozen pytest tests/test_indexing.py -q` must stay green (same
  pass/skip/xfail counts modulo intentional restructure).
- Coverage sanity: the rewritten cases must still exercise, for each selection
  type, at least one cross-chunk selection, one partial-edge-chunk selection,
  one negative/wraparound index, and the documented error paths.
- Final: `--durations=25` before/after to record the speedup, and a mypy pass
  via `prek --all-files` (per memory) since type annotations on the parametrize
  lists change and the `Expect` shape moves.

## Success criteria

- All loop-based selection tests replaced by parametrized, docstring'd tests
  with one behavior per case.
- Error paths are individually named tests.
- Full-file wall time measurably lower (target: the four >0.4 s tests each
  drop below ~0.15 s; expect total well under the current ~3.5 s).
- No loss of structural coverage per the checklist above.
- `git diff` is reviewable as a per-family sequence of commits.
