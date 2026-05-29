# Design: Close the "silently-unexecuted docs block" gap

**Date:** 2026-05-29
**Issue:** [zarr-developers/zarr-python#4016](https://github.com/zarr-developers/zarr-python/issues/4016)

## Problem & root cause

Issue #4016 reports invalid code in the docs:

```python
z = zarr.create_array("s3://example-bucket/foo", mode="w", shape=(100, 100), chunks=(10, 10), dtype="f4")
```

`create_array` has no `mode` parameter, so this raises `TypeError: unexpected keyword
argument 'mode'`. The code was wrong because **nothing validated it**: it is a bare
` ```python ` block, and both the renderer (`markdown-exec`) and the test suite
(`tests/test_docs.py`, which filters on `settings.get("exec") != "true"`) only act on
blocks tagged `exec="true"`. Omitting that attribute is a *silent* opt-out from all
validation.

This is not a one-off. An audit of all docs found **12 of 180** python blocks
unexecuted, including a second instance of the same failure mode:
`docs/contributing.md:231` is tagged `exec="on"` (a typo for `"true"`), so a block
meant to run silently does not.

**Root cause:** validation is opt-in via an easily-mistyped, easily-omitted attribute,
with no signal when a block opts out.

### Audit of the 12 bare blocks

| Block | Why bare | Disposition |
|---|---|---|
| `docs/quick-start.md:134` (S3) | hits real S3 | **Execute, `markers="s3"`** (moto infra, default doctest env) |
| `docs/user-guide/gpu.md:19` | needs cupy + GPU | **Execute, `markers="gpu"`** (runs in `gputest` env) |
| `docs/user-guide/performance.md:207` | left bare | **`exec="true"`** (plain `zarr.config.set`) |
| `docs/user-guide/performance.md:237` | left bare | **`exec="true"`** |
| `docs/user-guide/performance.md:263` | left bare | **`exec="true"`** (uses dask + a local array path) |
| `docs/user-guide/arrays.md:622` | left bare | **`exec="true"`** (`zarr.config.set`) |
| `docs/user-guide/cli.md:48` | left bare | **`exec="true"`** (`zarr.open`; needs a runnable path/store) |
| `docs/contributing.md:231` | **`exec="on"` typo** | **Fix typo** → `exec="true"` |
| `docs/contributing.md:15` | pseudocode (`# etc.`) | **Explicit opt-out** + reason |
| `docs/user-guide/data_types.md:363` | REPL transcript (`<class ...>`) | **Explicit opt-out** + reason |
| `docs/user-guide/examples/custom_dtype.md:5` | `--8<--` file include | **Explicit opt-out** + reason |
| `docs/user-guide/v3_migration.md:42` | intentionally-wrong import | **Explicit opt-out** + reason |

(`performance.md:263` and `cli.md:48` need a small adjustment — a memory store or a
real local path — to be runnable; confirm during implementation.)

## Approach

Two complementary parts.

### Part A — Per-case remediation of the 12 bare blocks

Not one mechanism — a triage. Each block gets the treatment that fits *why* it is not
executing:

- **Make executable against fakes** — the S3 example, via `markers="s3"`. The marker
  binds the block to the repo's existing `moto` mock-S3 infra (pattern from
  `tests/test_store/test_fsspec.py`) so it runs for real in CI with no real-cloud
  contact. Execution validates the whole write path, not just the signature; `mode="w"`
  dies by construction. See "Marker-bound execution".
- **Just turn on** — the config/open blocks (`performance.md` ×3, `arrays.md:622`,
  `cli.md:48`) are plain runnable API calls; flip them to `exec="true"`.
- **Fix the typo** — `contributing.md:231` `exec="on"` → `exec="true"`.
- **Execute, env-gated** — the GPU block, via `markers="gpu"`. It *can* run, but only in
  the `gputest` env (cupy + GPU hardware), not the default `doctest` env. See
  "Marker-bound execution".
- **Explicit opt-out** — blocks that genuinely cannot run anywhere and are not
  executable Python: REPL transcript, `--8<--` include, intentionally-wrong import,
  pseudocode. These get a *documented, greppable* opt-out marker carrying a reason.

### Part B — A guard test

So the gap cannot silently reopen: every python block in `docs/` must either be
`exec="true"` *or* carry the explicit opt-out marker with a reason. A bare or
mistyped block fails the guard. This would have caught both `mode="w"` and the
`exec="on"` typo.

### Dropped from scope

The type-checking / markdown-extractor machinery considered earlier. Execution-against-
fakes strictly dominates type-checking for the cloud case (and the untyped `s3fs`/`cupy`
imports make strict type-checking least clean exactly where it was wanted most), and the
guard handles everything else. Proportionate to ~7 genuinely-affected blocks.

## Key insight: doctests are already pytest tests

`tests/test_docs.py::test_documentation_examples` is an ordinary `@pytest.mark.parametrize`d
pytest test — one case per `(file, session)`. It is not a separate doctest mechanism.
Therefore everything pytest already provides for gating tests (markers, `-m` selection,
skips) is available; the design uses it rather than inventing harness concepts.

There are two distinct executors of docs blocks, and conflating them is what made
marker-bound execution look hard:

- **`markdown-exec` at docs-build time** — runs blocks to render output into the
  published site. Build runners have no cupy (and the S3 setup is test infra), so a
  marker-bound block must render as static source here (no build-time execution).
- **`tests/test_docs.py` at test time** — the validation. This is pytest, and this is
  where markers live, where infra fixtures bind, and where env-gating happens.

## Two flags: `exec` (render output) vs `test` (validate)

A code block can be *run* for two unrelated reasons, and conflating them breaks the
build. They are separate fence attributes:

- **`exec="true"`** — markdown-exec executes the block **at docs-build time to render its
  output** into the published page. This is markdown-exec's own attribute (it hard-codes
  the name `exec`, see `markdown_exec/_internal/main.py`), so we cannot rename it. Read it
  as *"execute to render output."*
- **`test="true"`** — **our** `tests/test_docs.py` harness executes the block **as a
  validation test**. markdown-exec does not recognize `test=` and ignores it.

Why two: a block that needs special infra to run (GPU/cupy, or S3) must be **validated in
tests** but must **not run at build** — build runners have no GPU and no moto server, so
an `exec="true"` GPU block makes `mkdocs build --strict` abort (`ModuleNotFoundError:
cupy`). Separating the flags lets such a block be `test="true"` (tested) without
`exec="true"` (so it renders as static source at build, never executed there).

**Harness rule:** a block is collected as a test if `exec="true"` **OR** `test="true"`.
So existing `exec="true"` example blocks stay tested as before (backward-compatible), and
test-only blocks add `test="true"` without `exec`.

**The combinations:**

| Block | `exec` | `test` | Effect |
|---|---|---|---|
| Tutorial examples (quickstart, config, …) | `true` | — | Run at build (render output); also tested. |
| GPU / S3 examples | — | `true` | Tested (under markers); rendered static at build. |
| Non-runnable (transcript, include, wrong-import) | `false`+`reason` | — | Neither; explicit reasoned opt-out. |

**Placement constraint (markdown-exec quirk).** markdown-exec registers a SuperFences
custom fence for `python`; its validator *rejects* any fence lacking `exec="true"`
(`exec="false"` and `test="true"` alike — both are "not executed at build"). Established by
experiment: a rejected python fence positioned **before** an `exec="true"` block disrupts
markdown-exec's build-time execution of a **later, state-dependent** block (regardless of
session). Observed concretely: any non-exec python fence inserted before the quickstart
`ZipStore` write/read pair made the read block fail (`FileNotFoundError` — the write never
took effect) and `mkdocs build --strict` aborted. The effect only surfaces with a
cross-block dependency, so it does **not** affect the standalone `exec="true"` blocks in
`data_types.md`/`performance.md` that already carry `exec="false"` opt-out blocks above
them.

Because we cannot statically tell which later blocks are state-dependent, the response is
twofold: (1) **a `test="true"`-only block must come last on its page** (or be the only
python block, as on `gpu.md`) — a conservative convention enforced by
`test_test_only_blocks_come_last` for the blocks we author this way; and (2) the
**authoritative** build-hazard check is `mkdocs build --strict` (the `docs:check` CI job),
which catches the `exec="false"` case too. The S3 example is placed at the end of
`quick-start.md` accordingly.

## Marker-bound execution

A block declares the pytest marker it needs via a **fence attribute**. Marker-bound
blocks are `test="true"` (validated) but **not** `exec="true"` (not build-run), e.g.:

````
```python test="true" markers="gpu" source="above"
```python test="true" markers="s3" source="above"
````

`group_examples_by_session()` parses `markers=` and emits
`pytest.param(session_key, marks=pytest.mark.<m>)`. The marker then **binds the case to
whatever that marker means** — and the two markers mean different things, which is the
point of unifying the model rather than special-casing each:

- **`gpu` — env-gate.** A registered marker does **not** auto-skip under plain `pytest`
  (markers only *filter* when you pass `-m`). The repo's convention is
  `pytest.importorskip("cupy")` in the test body (cf. `tests/conftest.py`), so the harness
  calls `importorskip("cupy")` for gpu-marked docs cases: in the default `doctest` env the
  case is **skipped** (no cupy), and `pytest -m gpu` in the `gputest` env runs it on real
  cupy. The block is `test="true"` (not `exec="true"`), so it is never run at build.

- **`s3` — infra-binding.** A new `s3` marker (must be registered in the `markers`
  table). An autouse-style fixture keyed on the marker stands up the `moto` server and
  registers a default endpoint, so an `s3`-marked docs case runs against the fake S3
  with no real-cloud contact. Because the infra is just pip deps already present in the
  `doctest` env (`s3fs`, `moto[s3,server]`), the case **runs in the default doctest
  run** — the marker binds infra, it does not gate the case out. The moto/endpoint
  plumbing lives in named pytest fixtures, not a hidden markdown setup block.

Both blocks therefore follow one rule: *declare the marker; the harness binds the marker
to the infra/env it needs.* The asymmetry is in what each marker resolves to (gpu →
hardware env, s3 → fixture), not in the declaration mechanism.

## Components & data flow

**`docs/` markdown** — source of truth. Each python block is in one of three declared
states (see the two-flags table above):

1. `exec="true"` and/or `test="true"` (optionally `+ markers="<m>"`) — validated, by
   build-render and/or by the test harness.
2. `exec="false"` with a `reason="..."` — explicit, documented opt-out.
3. anything else (bare, `exec="on"`, …) — **illegal**, fails the guard.

The opt-out form is `exec="false" reason="..."`: explicit, greppable, carries a
human-readable reason, and is not executed by markdown-exec at build time.

**`tests/test_docs.py`** — already-parametrized pytest harness. Changes:

- `group_examples_by_session()` parses the `markers=` attribute and emits
  `pytest.param(..., marks=pytest.mark.<m>)` so marker-binding rides existing marker
  machinery.
- A marker-keyed fixture for `s3` that stands up the `moto` server and registers a
  default endpoint (pattern lifted from `tests/test_store/test_fsspec.py`), applied to
  `s3`-marked docs cases.
- New guard test `test_no_unvalidated_blocks` — walks every python block in `docs/`,
  asserts each is `exec="true"` or carries the explicit opt-out marker. Fails on
  bare/typo'd blocks.

**`pyproject.toml`** — register the new `s3` marker in the `markers` table (alongside
`gpu`).

**`docs/quick-start.md` S3 block** — gains `markers="s3"`. The visible code stays a clean
`create_array("s3://...")`; the moto server and default-endpoint registration are
supplied by the `s3` fixture, not by an in-markdown setup block.

## Risks & spikes (resolve during implementation; do not guess)

1. **Default S3 endpoint without `storage_options`.** Existing tests always pass
   `endpoint_url` explicitly (`test_fsspec.py:131`). Confirm the `s3` fixture can register
   a *process-wide* default endpoint (via `fsspec.config` or `AWS_ENDPOINT_URL`) so the
   visible `create_array("s3://...")` works clean with no `storage_options`. **Fallback:**
   show the honest `storage_options={"endpoint_url": ...}` form in the visible block.

2. **`markdown-exec` + unknown `markers=` attribute.** Confirm the build-time renderer
   ignores `markers=` (or is told to), and that marker-bound blocks render as static
   source in the published site (render source only, no build-time execution — the build
   has neither cupy nor the moto fixture). **Fallback:** a per-session marker map in
   `test_docs.py`, keeping markdown untouched.

3. **moto teardown / loop affinity in the docs session.** `s3fs`/`aiobotocore` finalizers
   are noisy at teardown and s3fs instances bind to the event loop they were created on
   (see the filterwarnings note in `pyproject.toml` and the loop comments in
   `test_fsspec.py`). Ensure the docs `s3` fixture starts/stops moto cleanly and does not
   leak across sessions/tests.

## Testing the change

- Guard test is self-validating: after remediation, the full docs suite passes with zero
  bare/typo'd blocks.
- Negative check: temporarily introduce a bare block, confirm the guard fails, remove it.
- S3 block: `hatch run doctest:test` runs it green against moto in the default doctest
  env (the `s3` marker binds the fixture; it is not gated out).
- GPU block: `pytest -m gpu` in `gputest` executes it; the default `doctest` run reports
  it **skipped**, not absent.

## Out of scope

- Type-checking machinery / markdown extractor.
- The 168 already-executing blocks.
- Broad docs rewrites beyond the 12 bare blocks.

## Upstream

[zarr-developers/zarr-python#4017](https://github.com/zarr-developers/zarr-python/issues/4017)
captures the root-cause framing (silent opt-out hides bugs; `mode="w"` and `exec="on"` as
two instances) and the Part B guard proposal for community discussion, independent of the
immediate fix in #4016.
