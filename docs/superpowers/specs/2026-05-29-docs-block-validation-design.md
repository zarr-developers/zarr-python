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
| `docs/quick-start.md:134` (S3) | hits real S3 | **Execute against moto** mock-S3 |
| `docs/user-guide/gpu.md:19` | needs cupy + GPU | **Execute, gated on `gpu` marker** (runs in `gputest` env) |
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

- **Make executable against fakes** — the S3 example. Reuse the repo's existing `moto`
  mock-S3 pattern from `tests/test_store/test_fsspec.py` so the block runs for real in
  CI with no real-cloud contact. Execution validates the whole write path, not just the
  signature; `mode="w"` dies by construction.
- **Just turn on** — the config/open blocks (`performance.md` ×3, `arrays.md:622`,
  `cli.md:48`) are plain runnable API calls; flip them to `exec="true"`.
- **Fix the typo** — `contributing.md:231` `exec="on"` → `exec="true"`.
- **Execute, env-gated** — the GPU block. It *can* run, but only in the `gputest` env
  (cupy + GPU hardware), not the default `doctest` env. See "Env-gated execution".
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
env-gating look hard:

- **`markdown-exec` at docs-build time** — runs blocks to render output into the
  published site. Build runners have no cupy, so a GPU block must render as static
  source here (no build-time execution).
- **`tests/test_docs.py` at test time** — the validation. This is pytest, and this is
  where markers live and where env-gating happens.

## Env-gated execution

A block declares the pytest marker it needs via a **fence attribute**, e.g.:

````
```python exec="true" markers="gpu"
````

`group_examples_by_session()` parses `markers=` and emits
`pytest.param(session_key, marks=pytest.mark.gpu)`. Then:

- Default `doctest` env runs `pytest` → the gpu-marked param is **skipped/deselected**,
  exactly like every other `gpu`-marked test in `tests/`.
- The `gputest` env runs `pytest -m gpu` → the param **executes** against real cupy.

This reuses the existing `gpu` marker (`pyproject.toml`, `markers` table) and the existing
`pytest -m gpu` selection — no new harness concept.

## Components & data flow

**`docs/` markdown** — source of truth. Each python block is in one of three declared
states:

1. `exec="true"` (optionally `+ markers="<m>"`) — executed as a test.
2. explicit opt-out marker **with a reason** — deliberately not executed.
3. anything else (bare, `exec="on"`, …) — **illegal**, fails the guard.

The exact spelling of the opt-out marker (e.g. `exec="false"` plus a `reason="..."`
attribute, versus a dedicated sentinel attribute) is an implementation-plan decision.
Requirement: it must be explicit, greppable, carry a human-readable reason, and be a
form `markdown-exec` will not execute at build time.

**`tests/test_docs.py`** — already-parametrized pytest harness. Changes:

- `group_examples_by_session()` parses the `markers=` attribute and emits
  `pytest.param(..., marks=pytest.mark.<m>)` so env-gating rides existing marker machinery.
- New guard test `test_no_unvalidated_blocks` — walks every python block in `docs/`,
  asserts each is `exec="true"` or carries the explicit opt-out marker. Fails on
  bare/typo'd blocks.

**`docs/quick-start.md` S3 session** — a hidden setup block (`exec="true"`, no `source=`,
matching the existing setup block at `quick-start.md:8`) starts a `moto` server and
registers a default endpoint so the *visible* `create_array("s3://...")` block runs
against the fake. Pattern lifted from `tests/test_store/test_fsspec.py`.

## Risks & spikes (resolve during implementation; do not guess)

1. **Default S3 endpoint without `storage_options`.** Existing tests always pass
   `endpoint_url` explicitly (`test_fsspec.py:131`). Confirm a setup block can register
   a *process-wide* default endpoint (via `fsspec.config` or `AWS_ENDPOINT_URL`) so the
   visible `create_array("s3://...")` works clean. **Fallback:** show the honest
   `storage_options={"endpoint_url": ...}` form in the visible block.

2. **`markdown-exec` + unknown `markers=` attribute.** Confirm the build-time renderer
   ignores `markers=` (or is told to), and decide how a `gpu` block renders in the
   published site without cupy (render source only, no build-time execution).
   **Fallback:** a per-session marker map in `test_docs.py`, keeping markdown untouched.

3. **moto teardown in the docs session.** `s3fs`/`aiobotocore` finalizers are known to be
   noisy at teardown (see the filterwarnings note in `pyproject.toml`). Ensure the docs
   session's moto server starts/stops cleanly without leaking into other sessions.

## Testing the change

- Guard test is self-validating: after remediation, the full docs suite passes with zero
  bare/typo'd blocks.
- Negative check: temporarily introduce a bare block, confirm the guard fails, remove it.
- S3 block: `hatch run doctest:test` runs it green against moto.
- GPU block: `pytest -m gpu` in `gputest` executes it; the default `doctest` run reports
  it **skipped**, not absent.

## Out of scope

- Type-checking machinery / markdown extractor.
- The 168 already-executing blocks.
- Broad docs rewrites beyond the 12 bare blocks.

## Upstream

A separate GitHub issue will capture the root-cause framing (silent opt-out hides bugs;
`mode="w"` and `exec="on"` as two instances) and the Part B guard proposal for community
discussion, independent of the immediate fix.
