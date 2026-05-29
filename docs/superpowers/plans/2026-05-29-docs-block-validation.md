# Docs Block Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every python code block in `docs/` either execute (and thus get validated) or explicitly opt out with a documented reason, and add a guard test so a block can never again silently opt out of validation.

**Architecture:** The doctests in `tests/test_docs.py` are already parametrized pytest tests. We (1) teach the parametrizer to read a `markers="..."` fence attribute and attach the matching pytest marker to each session's `pytest.param`, (2) add an `s3` marker bound to a `moto` mock-S3 fixture so the S3 example runs in the default doctest env, (3) reuse the existing `gpu` marker for the GPU block, (4) remediate the 12 currently-unexecuted blocks per-case, and (5) add a guard test asserting every docs python block is `exec="true"` or explicitly opted out with a reason.

**Tech Stack:** pytest, pytest-examples, markdown-exec (mkdocs), moto[s3,server], s3fs, hatch envs (`doctest`, `gputest`).

**Upstream:** Fixes [#4016](https://github.com/zarr-developers/zarr-python/issues/4016); implements the guard from [#4017](https://github.com/zarr-developers/zarr-python/issues/4017). Design spec: `docs/superpowers/specs/2026-05-29-docs-block-validation-design.md`.

---

## File Structure

- `tests/test_docs.py` — **modify.** Add `markers=` parsing in `group_examples_by_session()`, an `s3` fixture + marker-binding, and the new `test_no_unvalidated_blocks` guard test.
- `pyproject.toml` — **modify.** Register the `s3` marker in `[tool.pytest.ini_options] markers`.
- `docs/quick-start.md` — **modify.** S3 block: fix `mode="w"`, add `markers="s3"`, make it executable.
- `docs/user-guide/performance.md` — **modify.** Turn on the two config-only blocks; opt out (or fix) the dask block.
- `docs/user-guide/arrays.md` — **modify.** Turn on the config block.
- `docs/user-guide/cli.md` — **modify.** Make the `zarr.open` block runnable or opt it out.
- `docs/user-guide/gpu.md` — **modify.** Add `exec="true" markers="gpu"`.
- `docs/contributing.md` — **modify.** Fix `exec="on"` typo; opt out the pseudocode block.
- `docs/user-guide/data_types.md` — **modify.** Opt out the REPL-transcript block.
- `docs/user-guide/examples/custom_dtype.md` — **modify.** Opt out the `--8<--` include block.
- `docs/user-guide/v3_migration.md` — **modify.** Opt out the intentionally-wrong-import block.
- `changes/4016.bugfix.md` — **create.** Towncrier news fragment.

### Opt-out convention (decided here, used throughout)

A block that must not execute is tagged:

````
```python exec="false" reason="<human-readable reason>"
````

- `exec="false"` is an explicit, greppable opt-out that `markdown-exec` will **not** execute (only `exec="true"` triggers execution).
- `reason="..."` documents *why*. The guard test requires it on any non-`exec="true"` block.

---

## Task 1: Spike — can the `s3` fixture provide a default endpoint with no `storage_options`?

This is the load-bearing unknown. The existing S3 tests always pass `endpoint_url` explicitly via `client_kwargs`/`storage_options` (`tests/test_store/test_fsspec.py:109-116, 131`). The docs block must read clean — `zarr.create_array("s3://...")` with **no** `storage_options`. We must confirm a process-wide default endpoint works before writing the real fixture.

**Files:**
- Test (scratch): `tests/test_docs_s3_spike.py` (deleted at end of task)

- [ ] **Step 1: Write a scratch test that starts moto, sets a default endpoint via env, and creates an array with a bare `s3://` URL**

```python
# tests/test_docs_s3_spike.py
import os

import pytest

moto_server = pytest.importorskip("moto.moto_server.threaded_moto_server")
pytest.importorskip("s3fs")
botocore = pytest.importorskip("botocore")
requests = pytest.importorskip("requests")

PORT = 5556  # different from test_fsspec.py's 5555 to avoid collisions
ENDPOINT = f"http://127.0.0.1:{PORT}/"


def test_bare_s3_url_with_default_endpoint() -> None:
    """A create_array('s3://...') call with no storage_options should reach a
    moto server when the endpoint is configured process-wide (env var)."""
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=PORT)
    server.start()
    try:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foo")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "foo")
        # Candidate mechanism A: aiobotocore/botocore honors AWS_ENDPOINT_URL
        os.environ["AWS_ENDPOINT_URL"] = ENDPOINT

        # create the bucket via boto3 sync client
        session = botocore.session.Session()
        client = session.create_client("s3", endpoint_url=ENDPOINT, region_name="us-east-1")
        client.create_bucket(Bucket="docs-bucket")
        client.close()

        import s3fs

        import zarr

        s3fs.S3FileSystem.clear_instance_cache()
        z = zarr.create_array(
            "s3://docs-bucket/foo", shape=(8, 8), chunks=(4, 4), dtype="f4"
        )
        z[:, :] = 1.0
        assert z[0, 0] == 1.0
    finally:
        requests.post(f"{ENDPOINT}/moto-api/reset")
        server.stop()
```

- [ ] **Step 2: Run the spike**

Run: `hatch run doctest:test tests/test_docs_s3_spike.py -v` (or `uv run pytest tests/test_docs_s3_spike.py -v` inside the doctest env)
Expected: **One of two outcomes** — record which:
- **PASS** → `AWS_ENDPOINT_URL` works as a process-wide default. Use env-var mechanism in Task 3.
- **FAIL** (connection refused / NoCredentials / hits real AWS) → env var insufficient. Try candidate B below.

- [ ] **Step 3: If Step 2 failed, try fsspec default config**

Replace the `AWS_ENDPOINT_URL` line with:

```python
        import fsspec

        fsspec.config.conf["s3"] = {"client_kwargs": {"endpoint_url": ENDPOINT}, "anon": False}
```

Run: `hatch run doctest:test tests/test_docs_s3_spike.py -v`
Expected: PASS → use `fsspec.config.conf` mechanism in Task 3.

- [ ] **Step 4: If both failed, record the fallback decision**

If neither bare-URL mechanism works, the visible block will show `storage_options={"endpoint_url": ...}` honestly (spec fallback for spike #1). Note which mechanism (env var, fsspec config, or fallback) won, in the commit message — Task 3 depends on it.

- [ ] **Step 5: Delete the scratch test and commit the finding**

```bash
git rm tests/test_docs_s3_spike.py
git commit -m "test: spike s3 default-endpoint mechanism for docs (no storage_options)

Result: <env-var | fsspec-config | fallback-to-storage_options>"
```

---

## Task 2: Register the `s3` pytest marker

**Files:**
- Modify: `pyproject.toml` (the `[tool.pytest.ini_options]` `markers` list, currently at lines 446-450)

- [ ] **Step 1: Add the `s3` marker**

In `pyproject.toml`, change the `markers` list from:

```toml
markers = [
    "asyncio: mark test as asyncio test",
    "gpu: mark a test as requiring CuPy and GPU",
    "slow_hypothesis: slow hypothesis tests",
]
```

to:

```toml
markers = [
    "asyncio: mark test as asyncio test",
    "gpu: mark a test as requiring CuPy and GPU",
    "s3: mark a test as requiring a (mock) S3 backend via moto",
    "slow_hypothesis: slow hypothesis tests",
]
```

- [ ] **Step 2: Verify pytest accepts the marker (no unknown-marker warning)**

Run: `hatch run doctest:test --markers | grep s3`
Expected: shows `@pytest.mark.s3: mark a test as requiring a (mock) S3 backend via moto`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "test: register s3 pytest marker"
```

---

## Task 3: Teach `test_docs.py` to parse `markers=` and bind the `s3` fixture

This task adds (a) `markers=` parsing so a session carries the right pytest marker, and (b) the moto-backed `s3` fixture using the mechanism chosen in Task 1.

**Files:**
- Modify: `tests/test_docs.py`

- [ ] **Step 1: Write a failing test that a markered session carries its marker**

Add to `tests/test_docs.py`:

```python
def test_markers_attribute_is_parsed(tmp_path: Path) -> None:
    """A block tagged markers="s3" must surface that marker on its parametrized case,
    so pytest can gate/bind it (e.g. attach the moto fixture)."""
    md = tmp_path / "ex.md"
    md.write_text(
        '```python exec="true" session="demo" markers="s3"\n'
        "import zarr\n"
        "```\n",
        encoding="utf-8",
    )
    params = _session_params(md.parent)
    assert len(params) == 1
    marks = params[0].marks
    assert any(m.name == "s3" for m in marks)
```

(This references a new helper `_session_params(root)` that returns a list of `pytest.param(...)`; we extract the grouping logic into it in Step 3.)

- [ ] **Step 2: Run it to confirm it fails**

Run: `hatch run doctest:test tests/test_docs.py::test_markers_attribute_is_parsed -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_session_params'` (or `NameError`).

- [ ] **Step 3: Refactor grouping into `_session_params` that emits markers**

Replace `group_examples_by_session()` (currently `tests/test_docs.py:39-64`) and the parametrize decorator (`tests/test_docs.py:72-75`) with a version that returns `pytest.param` objects carrying marks. Add near the top of the file:

```python
def _markers_for(settings: dict[str, str]) -> list[pytest.MarkDecorator]:
    """Translate a block's markers="a b" attribute into pytest mark decorators."""
    raw = settings.get("markers", "")
    return [getattr(pytest.mark, name) for name in raw.split() if name]


def _session_params(root: Path) -> list[pytest.param]:
    """Group exec="true" examples by (file, session) and emit one pytest.param per
    session, carrying the union of markers declared by that session's blocks."""
    sessions: defaultdict[tuple[str, str], list[CodeExample]] = defaultdict(list)
    marks_by_session: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for example in find_examples(str(root)):
        settings = example.prefix_settings()
        if settings.get("exec") != "true":
            continue
        session_name = settings.get("session", "_default")
        key = (str(example.path), session_name)
        sessions[key].append(example)
        for mark in _markers_for(settings):
            marks_by_session[key].add(mark.name)

    params = []
    for key in sorted(sessions.keys(), key=lambda x: (x[0], x[1])):
        marks = tuple(getattr(pytest.mark, name) for name in sorted(marks_by_session[key]))
        params.append(pytest.param(key, marks=marks, id=name_example(key[0], key[1])))
    return params
```

Keep `name_example()` as-is. Add `CodeExample` to the existing pytest-examples import if not already imported (it is: `from pytest_examples import CodeExample, EvalExample, find_examples`).

- [ ] **Step 4: Update the parametrized test to use `_session_params` and request the fixtures**

Replace the decorator + signature of `test_documentation_examples` (`tests/test_docs.py:72-79`) with:

```python
@pytest.mark.parametrize("session_key", _session_params(DOCS_ROOT))
def test_documentation_examples(
    session_key: tuple[str, str],
    eval_example: EvalExample,
    request: pytest.FixtureRequest,
) -> None:
```

Inside the body, before running examples, activate the `s3` fixture when the case is s3-marked:

```python
    if request.node.get_closest_marker("s3") is not None:
        request.getfixturevalue("docs_s3_backend")
```

(Leave the rest of the body — the `find_examples` loop and `eval_example.run(...)` — unchanged.)

- [ ] **Step 5: Add the `docs_s3_backend` fixture**

Add to `tests/test_docs.py` (using the mechanism Task 1 selected — shown here for the `AWS_ENDPOINT_URL` variant; swap to `fsspec.config` or the `storage_options` fallback per Task 1's result):

```python
S3_PORT = 5556
S3_ENDPOINT = f"http://127.0.0.1:{S3_PORT}/"
S3_BUCKET = "example-bucket"


@pytest.fixture
def docs_s3_backend() -> Generator[None, None, None]:
    """Stand up a moto mock-S3 server and configure a process-wide default endpoint
    so docs blocks can use a bare s3:// URL with no storage_options."""
    moto_server = pytest.importorskip("moto.moto_server.threaded_moto_server")
    s3fs = pytest.importorskip("s3fs")
    botocore = pytest.importorskip("botocore")
    requests = pytest.importorskip("requests")

    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=S3_PORT)
    server.start()
    prev_endpoint = os.environ.get("AWS_ENDPOINT_URL")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foo")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "foo")
    os.environ["AWS_ENDPOINT_URL"] = S3_ENDPOINT

    session = botocore.session.Session()
    client = session.create_client("s3", endpoint_url=S3_ENDPOINT, region_name="us-east-1")
    client.create_bucket(Bucket=S3_BUCKET)
    client.close()
    s3fs.S3FileSystem.clear_instance_cache()
    try:
        yield
    finally:
        requests.post(f"{S3_ENDPOINT}/moto-api/reset")
        if prev_endpoint is None:
            os.environ.pop("AWS_ENDPOINT_URL", None)
        else:
            os.environ["AWS_ENDPOINT_URL"] = prev_endpoint
        server.stop()
```

Add the required imports at the top of `tests/test_docs.py`:

```python
import os
from collections.abc import Generator
```

- [ ] **Step 6: Run the marker-parsing test — it should now pass**

Run: `hatch run doctest:test tests/test_docs.py::test_markers_attribute_is_parsed -v`
Expected: PASS

- [ ] **Step 7: Run the full docs test to confirm no regression in existing sessions**

Run: `hatch run doctest:test -v`
Expected: PASS for all existing `quickstart` etc. sessions (the S3 block isn't markered yet — that's Task 4).

- [ ] **Step 8: Commit**

```bash
git add tests/test_docs.py
git commit -m "test: parse markers= on docs blocks and add moto s3 fixture binding"
```

---

## Task 4: Fix and enable the S3 example (#4016)

**Files:**
- Modify: `docs/quick-start.md:134-140`

- [ ] **Step 1: Replace the bare, invalid S3 block**

Replace lines 134-140 (the ```` ```python `` … ```` block containing `mode="w"`) with:

````markdown
```python exec="true" session="s3demo" markers="s3" source="above"
import zarr
import numpy as np

z = zarr.create_array(
    "s3://example-bucket/foo", shape=(100, 100), chunks=(10, 10), dtype="f4"
)
z[:, :] = np.random.random((100, 100))
```
````

Notes:
- `mode="w"` removed (the #4016 bug; `create_array` has no `mode` parameter — see `src/zarr/api/synchronous.py:799`).
- Unused `import s3fs` removed.
- `import numpy as np` added — this is a fresh `s3demo` session, so `np` is not in scope from the `quickstart` session.
- New session `s3demo` keeps the moto fixture scoped to just this block (the `quickstart` session must NOT become s3-marked).
- The displayed URL stays `s3://example-bucket/foo`; the moto endpoint is supplied by the `docs_s3_backend` fixture (bucket name `example-bucket` matches `S3_BUCKET` in Task 3).
- **If Task 1 chose the `storage_options` fallback:** add `storage_options={"endpoint_url": "..."}` to the visible call instead, and adjust the prose to explain it.

- [ ] **Step 2: Run the S3 docs example against moto**

Run: `hatch run doctest:test "tests/test_docs.py::test_documentation_examples[quick-start.md:s3demo]" -v`
Expected: PASS (executes against moto; no real-cloud contact).

- [ ] **Step 3: Commit**

```bash
git add docs/quick-start.md
git commit -m "docs: fix invalid s3 create_array example and run it against moto (#4016)"
```

---

## Task 5: Enable the config-only blocks

These are plain `zarr.config.set(...)` calls that run as-is. Each gets its own self-contained session so config mutations don't bleed into other examples (config is process-global; reset is out of scope — separate sessions keep ids distinct but note config is not auto-restored, which is acceptable for these read-only-style demos).

**Files:**
- Modify: `docs/user-guide/performance.md:207`, `docs/user-guide/performance.md:237`
- Modify: `docs/user-guide/arrays.md:622`

- [ ] **Step 1: Enable `performance.md:207` (concurrency config)**

Change the fence from ```` ```python ```` to:

````markdown
```python exec="true" session="perf-concurrency"
````

(Body unchanged — `import zarr` + `zarr.config.set({'async.concurrency': 128})` + the commented env-var line, which is inert.)

- [ ] **Step 2: Enable `performance.md:237` (max_workers config)**

Change the fence to:

````markdown
```python exec="true" session="perf-workers"
````

- [ ] **Step 3: Enable `arrays.md:622` (rectilinear_chunks config)**

Change the fence to:

````markdown
```python exec="true" session="arrays-rectilinear"
````

- [ ] **Step 4: Run the three sessions**

Run:
```bash
hatch run doctest:test \
  "tests/test_docs.py::test_documentation_examples[performance.md:perf-concurrency]" \
  "tests/test_docs.py::test_documentation_examples[performance.md:perf-workers]" \
  "tests/test_docs.py::test_documentation_examples[arrays.md:arrays-rectilinear]" -v
```
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add docs/user-guide/performance.md docs/user-guide/arrays.md
git commit -m "docs: execute config-setting examples in performance.md and arrays.md"
```

---

## Task 6: Make the CLI `zarr.open` block runnable

`docs/user-guide/cli.md:48` opens `'path/to/input.zarr'` which doesn't exist. Rewrite it to create then open a real local array so it executes and still illustrates `zarr_format=3`.

**Files:**
- Modify: `docs/user-guide/cli.md:46-51`

- [ ] **Step 1: Replace the block**

Replace the bare block with:

````markdown
```python exec="true" session="cli-open" source="above"
import zarr

# create a small array to open (stands in for the migrated store)
zarr.create_array("data/cli-demo.zarr", shape=(4, 4), chunks=(2, 2), dtype="i4")

zarr_with_v3_metadata = zarr.open("data/cli-demo.zarr", zarr_format=3)
```
````

(Keep the surrounding prose; the example now demonstrates `open(..., zarr_format=3)` on a real store. The illustrative `'path/to/input.zarr'` filename was the only reason it couldn't run.)

- [ ] **Step 2: Run it**

Run: `hatch run doctest:test "tests/test_docs.py::test_documentation_examples[cli.md:cli-open]" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/cli.md
git commit -m "docs: make cli zarr.open example runnable against a local store"
```

---

## Task 7: Enable the GPU block (env-gated via `gpu` marker)

**Files:**
- Modify: `docs/user-guide/gpu.md:19-28`

- [ ] **Step 1: Tag the GPU block**

Change the fence from ```` ```python ```` to:

````markdown
```python exec="true" session="gpu-demo" markers="gpu" source="above"
````

(Body unchanged: `import cupy as cp`, `zarr.config.enable_gpu()`, `create_array("memory://gpu-demo", ...)`, etc.)

- [ ] **Step 2: Confirm it is SKIPPED in the default doctest env (no GPU)**

Run: `hatch run doctest:test "tests/test_docs.py::test_documentation_examples[gpu.md:gpu-demo]" -v`
Expected: SKIPPED (the `gpu` marker is deselected without `-m gpu`), **not** an error, **not** absent.

- [ ] **Step 3: Confirm it is COLLECTED for the gpu selection**

Run: `hatch run doctest:test -m gpu --co -q | grep gpu-demo`
Expected: the `gpu.md:gpu-demo` case is collected (it will actually execute only on real GPU hardware in the `gputest` env, which we can't run here).

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/gpu.md
git commit -m "docs: execute gpu example under the gpu marker"
```

---

## Task 8: Fix the `exec="on"` typo and opt out the genuinely-non-executable blocks

**Files:**
- Modify: `docs/contributing.md:15` (pseudocode) and `docs/contributing.md:231` (`exec="on"` typo)
- Modify: `docs/user-guide/data_types.md:363` (REPL transcript)
- Modify: `docs/user-guide/examples/custom_dtype.md:5` (`--8<--` include)
- Modify: `docs/user-guide/v3_migration.md:42` (intentionally-wrong import)

- [ ] **Step 1: Fix the `exec="on"` typo in `contributing.md:231`**

Change the fence attribute `exec="on"` to `exec="true"`. Then run that block to confirm it actually executes cleanly:

Run: `hatch run doctest:test -v -k contributing`
Expected: the formerly-`exec="on"` block now runs. **If it fails** (the code was broken too, having never run), fix the code in the block minimally so it passes, or — if it's not meant to run — convert it to `exec="false" reason="..."`. Record which in the commit.

- [ ] **Step 2: Opt out `contributing.md:15` (pseudocode)**

Change ```` ```python ```` to:

````markdown
```python exec="false" reason="illustrative pseudocode with a '# etc.' placeholder, not runnable"
````

- [ ] **Step 3: Opt out `data_types.md:363` (REPL transcript)**

Change ```` ```python ```` to:

````markdown
```python exec="false" reason="REPL output transcript, not executable source"
````

- [ ] **Step 4: Opt out `custom_dtype.md:5` (`--8<--` include)**

Change ```` ```python ```` to:

````markdown
```python exec="false" reason="pymdownx snippet include directive, not python source"
````

- [ ] **Step 5: Opt out `v3_migration.md:42` (intentionally-wrong import)**

Change ```` ```python ```` to:

````markdown
```python exec="false" reason="intentionally shows the old/incorrect import for contrast"
````

- [ ] **Step 6: Commit**

```bash
git add docs/contributing.md docs/user-guide/data_types.md docs/user-guide/examples/custom_dtype.md docs/user-guide/v3_migration.md
git commit -m "docs: fix exec=on typo and explicitly opt out non-runnable blocks"
```

---

## Task 9: Handle the dask block in performance.md

`docs/user-guide/performance.md:263` uses `dask.array` and opens `'data/large_array.zarr'` (nonexistent). Two viable dispositions — pick based on whether `dask` is in the doctest env.

**Files:**
- Modify: `docs/user-guide/performance.md:263-280`

- [ ] **Step 1: Check whether dask is available in the doctest env**

Run: `hatch run doctest:list-env | grep -i dask`
Expected: either shows a `dask` line (available) or nothing (not available).

- [ ] **Step 2a: If dask IS available — make it runnable**

Replace the `'data/large_array.zarr'` open with a created array, keeping the dask demonstration:

````markdown
```python exec="true" session="perf-dask" source="above"
import zarr
import dask.array as da

zarr.config.set({
    'async.concurrency': 4,
    'threading.max_workers': 4,
})

# create a small array to read with Dask
zarr.create_array("data/perf-dask-demo.zarr", shape=(16, 16), chunks=(8, 8), dtype="f4")
z = zarr.open_array("data/perf-dask-demo.zarr", mode="r")

arr = da.from_array(z, chunks=z.chunks)
result = arr.mean(axis=0).compute()
```
````

Run: `hatch run doctest:test "tests/test_docs.py::test_documentation_examples[performance.md:perf-dask]" -v`
Expected: PASS

- [ ] **Step 2b: If dask is NOT available — opt out with a reason**

Change ```` ```python ```` to:

````markdown
```python exec="false" reason="requires dask, which is not in the docs test environment"
````

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/performance.md
git commit -m "docs: make dask performance example runnable (or opt out if dask absent)"
```

---

## Task 10: Add the guard test

The guard asserts every python block in `docs/` is either `exec="true"` or `exec="false"` with a non-empty `reason`. Anything else (bare, `exec="on"`, missing reason) fails.

**Files:**
- Modify: `tests/test_docs.py`

- [ ] **Step 1: Write the guard test**

Add to `tests/test_docs.py`:

```python
def test_no_unvalidated_blocks() -> None:
    """Every python code block in docs/ must declare its validation state:
    either exec="true" (it is executed as a test) or exec="false" with a reason
    (an explicit, documented opt-out). A bare or mistyped fence (e.g. exec="on")
    fails here, so a block can never silently opt out of validation — the gap
    that hid the invalid create_array(mode="w") example in #4016."""
    offenders: list[str] = []
    for example in find_examples(str(DOCS_ROOT)):
        settings = example.prefix_settings()
        exec_val = settings.get("exec")
        loc = f"{Path(example.path).relative_to(DOCS_ROOT)}:{example.start_line}"
        if exec_val == "true":
            continue
        if exec_val == "false" and settings.get("reason", "").strip():
            continue
        offenders.append(f"{loc} (exec={exec_val!r}, reason={settings.get('reason')!r})")

    assert not offenders, (
        "Docs python blocks must be exec=\"true\" or exec=\"false\" with a reason:\n"
        + "\n".join(offenders)
    )
```

(`find_examples` from pytest-examples only yields fenced code blocks for languages it recognizes as runnable, which includes python; confirm in Step 2 that the count matches the audit. If it also yields non-python fences, filter on `example.prefix` / language — adjust to `if not str(example.path).endswith(".md"): continue` is unnecessary since DOCS_ROOT is all markdown.)

- [ ] **Step 2: Run the guard — it must PASS now that all blocks are remediated**

Run: `hatch run doctest:test tests/test_docs.py::test_no_unvalidated_blocks -v`
Expected: PASS (zero offenders). **If it lists offenders**, they are blocks missed by Tasks 4-9 — fix each (turn on or opt out) until the list is empty.

- [ ] **Step 3: Negative check — confirm the guard actually catches a bare block**

Temporarily add a bare block to any docs file:

````markdown
```python
1 / 0
```
````

Run: `hatch run doctest:test tests/test_docs.py::test_no_unvalidated_blocks -v`
Expected: FAIL, listing the new bare block's location.

Then remove the temporary block and re-run:
Run: `hatch run doctest:test tests/test_docs.py::test_no_unvalidated_blocks -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_docs.py
git commit -m "test: guard that every docs python block is executed or opted out (#4017)"
```

---

## Task 11: Full suite + news fragment

**Files:**
- Create: `changes/4016.bugfix.md`

- [ ] **Step 1: Run the entire docs test suite**

Run: `hatch run doctest:test -v`
Expected: PASS — all `exec="true"` sessions run (S3 against moto; config/cli/dask as applicable), the GPU session reports SKIPPED, and the guard passes.

- [ ] **Step 2: Add the towncrier news fragment**

Create `changes/4016.bugfix.md`:

```markdown
Fixed an invalid ``zarr.create_array`` example in the quick-start docs (it passed an unsupported ``mode`` argument) and made the cloud-storage example execute against a mock S3 backend in CI. Added a test ensuring every python code block in the docs is either executed or explicitly opted out with a documented reason.
```

- [ ] **Step 3: Run the full prek/lint pass**

Run: `prek run --all-files`
Expected: PASS (ruff, mypy, towncrier-check, etc. all green).

- [ ] **Step 4: Commit**

```bash
git add changes/4016.bugfix.md
git commit -m "docs: add news fragment for docs-block validation (#4016, #4017)"
```

---

## Self-review notes (resolved during planning)

- **Spec coverage:** Part A (remediate 12 blocks) → Tasks 4-9; Part B (guard) → Task 10. Marker-bound execution (s3 + gpu) → Tasks 2, 3, 4, 7. Spike #1 → Task 1. `pyproject.toml` s3 marker → Task 2. All three spec spikes are addressed: #1 in Task 1; #2 (markdown-exec tolerance of `markers=`) is implicitly verified by `hatch run docs:build` — **add a build check**: see Task 11 Step 1 note below; #3 (moto teardown) handled by the fixture's `finally` block in Task 3 Step 5.
- **Spike #2 verification:** `markers=` and `reason=`/`exec="false"` are unknown attributes to markdown-exec; it ignores unrecognized prefix settings and only acts on `exec="true"`. Confirm by running `hatch run docs:build` once after Task 11 and checking it succeeds and that the gpu/s3 blocks render as static source. If the build errors on unknown attributes, fall back to the per-session marker map (spec fallback for spike #2).
- **The 12 blocks, accounted for:** quick-start S3 (T4), perf×2 config (T5), arrays config (T5), cli (T6), gpu (T7), contributing exec=on typo + pseudocode (T8), data_types transcript (T8), custom_dtype include (T8), v3_migration wrong-import (T8), perf dask (T9). = 12. ✓
- **Naming consistency:** `_session_params`, `_markers_for`, `docs_s3_backend`, `test_no_unvalidated_blocks`, `S3_BUCKET="example-bucket"` (matches the URL in the T4 block) used consistently across tasks.
