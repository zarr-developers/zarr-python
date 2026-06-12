# AGENTS.md

Guidance for AI coding agents working in this repository. Human contributors should also read `docs/contributing.md`, especially the **AI-assisted contributions** section: the human submitting the PR must understand and being able to explain every change, and PR descriptions / review responses must be in the human's own words. Keep diffs small and reviewable.

## Project overview

`zarr-python` (PyPI package `zarr`) implements chunked, compressed, N-dimensional arrays for Python. This is the 3.x line, which reads and writes both Zarr format v2 and v3 data. Requires Python >= 3.12. The public API is re-exported from `src/zarr/__init__.py` (`Array`, `Group`, `create_array`, `open`, etc.).
## Related projects

`zarr-python` depends on the contents of the Zarr [v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html) and Zarr [v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) storage specifications. We are committed to compliance with the specs, and also consistency with other Zarr implementations, namely:
-  [TensorStore](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) (C++ / Python)
- [Zarrs](https://github.com/zarrs/zarrs) (Rust)
- [Zarrita](https://github.com/manzt/zarrita.js/) (Javascript)
## Environment and common commands

Development uses **hatch** (with `uv` as the installer) for managed environments, and `uv` directly for ad-hoc commands. The canonical test environments are named `test.py3.{12,13,14}-{minimal,optional}`; `optional` pulls in remote stores (fsspec, obstore, s3fs), the CLI, and universal-pathlib.

```bash
# Run the full test suite in a managed env (benchmarks excluded by default)
hatch env run --env test.py3.12-optional run

# Run with coverage (XML report); coverage must reach 100% for CI to pass
hatch env run --env test.py3.12-optional run-coverage

# Ad-hoc test runs with uv (faster iteration than spinning up a hatch env).
# Prefer uv run pytest for narrow runs; reach for hatch envs for full/coverage runs.
uv run pytest tests/test_array.py                 # one test file
uv run pytest tests/test_array.py::test_name      # one test function
uv run pytest tests/test_array.py -k "expr"       # tests matching a -k expression
uv run pytest "tests/test_array.py::test_name[param-id]"  # one parametrized case
uv run pytest tests/test_array.py -x --lf         # stop on first failure, rerun last-failed

# Run tests in parallel across CPUs with pytest-xdist (-n). Big speedup for the
# full suite; for a handful of tests the worker startup cost usually isn't worth it.
uv run pytest -n auto tests/                       # auto = one worker per core
uv run pytest -n 4 tests/test_codecs/             # fixed worker count

# Type-check (strict mypy over src + tests) — this is what CI's Lint job runs
uv run --frozen mypy

# Hypothesis property tests (slow; opt in)
hatch env run --env test.py3.12-optional run-hypothesis

# Docs: live-reloading server / strict build
hatch --env docs run serve
hatch --env docs run check
```

Note: `pytest` `testpaths` include `src`, `tests`, and `docs/user-guide`, and `--doctest-modules` is on — docstrings in `src/` are executed as tests, and the user-guide markdown is doctested. `xfail_strict = true` and `filterwarnings = ["error", ...]` mean an unexpected pass or an unfiltered warning fails the suite.

## Linting and pre-commit

The project uses [`prek`](https://github.com/j178/prek) (a drop-in pre-commit runner) against `.pre-commit-config.yaml`. Ruff (lint + format, line length 100) and `mypy` run here, plus codespell, numpydoc validation, towncrier-check, and zizmor.

```bash
prek run --all-files          # all hooks
prek run --last-commit        # only files changed in the last commit
```

Two local rules worth knowing because they will reject otherwise-valid code:
- **No `.lstrip("...")` / `.rstrip("...")` with multi-char string args** (`ban-lstrip-rstrip`) — these are character-set operations and almost always a bug where `removeprefix`/`removesuffix` was intended.
- **numpydoc validation** is enforced on docstrings (Parameters/Returns sections must match signatures).

## Changelog (required for most PRs)

Every user-facing change needs a news fragment in `changes/` named `{issue-or-pr-number}.{type}.md`, where type is one of `feature`, `bugfix`, `doc`, `removal`, `misc`. Generated into `docs/release-notes.md` by towncrier at release time. `towncrier create` scaffolds one. The `towncrier-check` pre-commit hook will flag PRs missing a fragment.

## Architecture

### Async core with a generated sync facade

Every storage and array operation is fundamentally **async**. The two-layer pattern is pervasive and important to preserve:

- `AsyncArray` / `AsyncGroup` (`core/array.py`, `core/group.py`) hold the real async implementations.
- `Array` / `Group` are thin synchronous wrappers that drive the async methods through `sync()` (`core/sync.py`), which runs coroutines on a single dedicated background event loop (a module-global loop in its own thread). Never call `asyncio.run` or create new loops in library code — route through `sync()`.
- The user-facing top-level functions split the same way: `api/asynchronous.py` (async) and `api/synchronous.py` (sync wrappers). `__init__.py` re-exports the synchronous ones.

When adding functionality, implement it on the `Async*` class / in `api/asynchronous.py` first, then add the sync wrapper.

### Store abstraction

`abc/store.py` defines the abstract `Store` (async `get`/`set`/`delete`/`list`/`exists`, partial reads via `RangeByteRequest`/`OffsetByteRequest`/`SuffixByteRequest`). Concrete stores live in `storage/`: `_local.py`, `_memory.py`, `_zip.py`, `_fsspec.py` (any fsspec filesystem), `_obstore.py` (Rust `obstore` backend), plus `_wrapper.py` and `_logging.py` decorators. `StorePath` (`storage/_common.py`) couples a store with a key prefix and is what arrays/groups actually hold.

### Codecs and the codec pipeline

Reading/writing a chunk runs it through an ordered pipeline of codecs, typed by what they transform (`abc/codec.py`): `ArrayArrayCodec` → `ArrayBytesCodec` (exactly one, e.g. `bytes`) → `BytesBytesCodec` (e.g. compressors). `core/codec_pipeline.py` (`BatchedCodecPipeline`) orchestrates encode/decode and batches/concurrency-limits chunk I/O. Built-in codecs are in `codecs/` (`blosc`, `gzip`, `zstd`, `crc32c`, `transpose`, `sharding`, `vlen_utf8`, `scale_offset`, ...); `codecs/numcodecs/` and `codecs/_v2.py` bridge to numcodecs for v2 filters/compressors. **Sharding** (`codecs/sharding.py`) is itself an `ArrayBytesCodec` that packs a grid of sub-chunks into one store object — the most intricate codec.

### Metadata and v2/v3 duality

`AsyncArray` is generic over `ArrayV2Metadata | ArrayV3Metadata` (`core/metadata/`). Most of the codebase branches on format version through this metadata type rather than ad-hoc version flags. v2 uses `.zarray`/`.zattrs`/`.zgroup`; v3 uses `zarr.json`. `metadata/migrate_v3.py` and `core/metadata/` hold the conversion and parsing logic. When touching read/write paths, check both metadata classes.

### dtype system

`core/dtype/` is a pluggable dtype layer (not raw NumPy dtypes) that maps Zarr data types to/from NumPy and handles fill values, endianness, and v2-vs-v3 dtype naming. `dtype.py` at the package root re-exports it.

### Registry and config

`registry.py` is the extension mechanism: codecs, pipelines, buffers, ndbuffers, and chunk-key-encodings are registered by name and discovered via entry points (`zarr.codecs`, `zarr.codec_pipelines`, etc.), so third-party packages can plug in. `core/config.py` uses `donfig` (the `zarr.config` object) for runtime settings — default codecs, concurrency limits, the active buffer/pipeline implementation. Resolving "which class implements X" generally goes through config → registry.

### Buffers and the GPU path

`abc/buffer.py` + `core/buffer/` abstract `Buffer` (1-D bytes) and `NDBuffer` (N-D array) so the same code runs on CPU (`core/buffer/cpu.py`, NumPy) or GPU (`core/buffer/gpu.py`, CuPy). GPU-only tests are marked `@pytest.mark.gpu`. Don't hardcode `np.ndarray`; go through the buffer prototype.

## Conventions

- **Public API surface:** anything added to `__all__` in `__init__.py` / `api/synchronous.py` is public and must have a numpydoc docstring and an entry under `docs/api/*.md`. New user-facing behavior also belongs in `docs/user-guide/`.
- **Experimental features** go under `src/zarr/experimental/`, are documented in `docs/user-guide/experimental.md`, and carry no stability guarantees (may be removed in any release). The team aims to promote or remove them within ~6 months.
- **Versioning is EffVer, not SemVer** — breaking changes are possible in minor (and rarely patch) releases, judged by upgrade effort. Prefer backwards-compatible changes and deprecation warnings over hard breaks.
