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

## Conventions

- **Public API surface:** anything added to `__all__` in `__init__.py` / `api/synchronous.py` is public and must have a numpydoc docstring and an entry under `docs/api/*.md`. New user-facing behavior also belongs in `docs/user-guide/`.
- **Experimental features** go under `src/zarr/experimental/`, are documented in `docs/user-guide/experimental.md`, and carry no stability guarantees (may be removed in any release). The team aims to promote or remove them within ~6 months.
- **Versioning is EffVer, not SemVer** — breaking changes are possible in minor (and rarely patch) releases, judged by upgrade effort. Prefer backwards-compatible changes and deprecation warnings over hard breaks.
