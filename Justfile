# Development and CI verbs live here; Hatch owns Python environments in pyproject.toml.
# Install: pip install hatch==1.16.5 rust-just==1.58.0 uv
# Select test dependencies/interpreter: HATCH_ENV=test.py3.13-minimal just test
# On Windows, use Git Bash (the same shell used by the test workflow).
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set windows-shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments

hatch_env := env("HATCH_ENV", "test.py3.12-optional")
# Deliberately a different variable from HATCH_ENV: `just gpu` must not inherit a
# CPU test environment that happens to be exported in the caller's shell, which
# would run `pytest -m gpu` against an environment built without the gpu feature.
gpu_env := env("GPU_HATCH_ENV", "gputest.py3.12")
# Pinned so a prek release cannot change what CI lints without a commit here.
prek_version := "0.5.3"

# List available recipes
default:
    @just --list

# List available Python environments
envs:
    hatch env show

# Create the selected Python environment and list its installed packages
setup:
    hatch env create {{ quote(hatch_env) }}
    just list-env

# List packages in the selected Python environment
list-env:
    hatch run {{ quote(hatch_env) }}:pip list

# Run unit tests; pass pytest arguments, e.g. just test -k 'array and resize'
test *args:
    hatch run {{ quote(hatch_env) }}:pytest --ignore tests/benchmarks "$@"

# Run unit tests and write coverage.xml and junit.xml
coverage *args:
    hatch run {{ quote(hatch_env) }}:coverage run --source=src -m pytest --ignore tests/benchmarks --junitxml=junit.xml -o junit_family=legacy "$@"
    hatch run {{ quote(hatch_env) }}:coverage xml

# Run unit tests and generate an HTML coverage report
coverage-html *args:
    hatch run {{ quote(hatch_env) }}:coverage run --source=src -m pytest --ignore tests/benchmarks "$@"
    hatch run {{ quote(hatch_env) }}:coverage html

# Serve the HTML coverage report (default port 8000)
coverage-serve *args:
    hatch run {{ quote(hatch_env) }}:python -m http.server -d htmlcov "$@"

# Run slow Hypothesis tests and write coverage.xml
hypothesis *args:
    hatch run {{ quote(hatch_env) }}:coverage run --source=src -m pytest -nauto --run-slow-hypothesis tests/test_properties.py tests/test_store/test_stateful* "$@"
    hatch run {{ quote(hatch_env) }}:coverage xml

# Validate executable documentation code blocks
doctest *args:
    hatch run doctest:pytest tests/test_docs.py -v "$@"

# Run the benchmark suite
benchmark *args:
    hatch run {{ quote(hatch_env) }}:pytest --benchmark-enable tests/benchmarks "$@"

# Run benchmarks under CodSpeed
benchmark-codspeed *args:
    hatch run {{ quote(hatch_env) }}:pytest tests/benchmarks --codspeed "$@"

# Run GPU tests with coverage; select the environment with GPU_HATCH_ENV
gpu *args:
    HATCH_ENV={{ quote(gpu_env) }} just coverage -m gpu "$@"

# Build documentation (warnings are errors)
docs-build *args:
    hatch run docs:mkdocs build --strict "$@"

# Serve documentation with live reload
docs-serve *args:
    hatch run docs:mkdocs serve --watch src "$@"

# Check that every public export has API documentation
check-doc-exports *args:
    hatch run docs:python ci/check_documented_exports.py docs/api "$@"

# Check documentation source conventions
lint-docs *args:
    hatch run docs:python ci/lint_docs.py "$@"

# Report unlinked types in built documentation
check-doc-links *args:
    hatch run docs:python ci/check_unlinked_types.py "$@"

# Run source documentation checks followed by a strict build
docs-check: check-doc-exports lint-docs docs-build

# Run all pre-commit hooks (ruff, codespell, mypy, repo-review, ...)
lint *args:
    uvx prek@{{ prek_version }} run --all-files "$@"

# Run hooks with a custom selection, e.g. just hooks run --last-commit
hooks +args:
    uvx prek@{{ prek_version }} "$@"

# Install local pre-commit hooks
hooks-install:
    uvx prek@{{ prek_version }} install

# Type-check the library using the locked tooling environment
typecheck *args:
    uv run --frozen mypy "$@"

# Check that uv.lock is in sync with pyproject.toml
lock-check:
    uv lock --check

# Update the dependency lockfile
lock *args:
    uv lock "$@"

# Build the source distribution and wheel
build *args:
    hatch build "$@"

# Create a changelog fragment (interactive without arguments)
changelog *args:
    hatch run docs:towncrier create "$@"

# Preview the next release's changelog
changelog-draft *args:
    hatch run docs:towncrier build --draft --version Unreleased "$@"

# Build release notes; pass --version and --yes when preparing a release
changelog-build *args:
    hatch run docs:towncrier build "$@"

# Check changelog filenames (default: changes/; accepts a package changes directory)
check-changelogs *args:
    uv run --no-project python ci/check_changelog_entries.py "$@"

# Check recipe formatting of the root Justfile and every package justfile
just-check:
    just --fmt --check
    for f in packages/*/justfile; do just --justfile "$f" --fmt --check; done

# Run a zarr-metadata recipe, or list its recipes with no arguments
zarr-metadata *args:
    just --justfile packages/zarr-metadata/justfile "$@"

# Run a zarr-indexing recipe, or list its recipes with no arguments
zarr-indexing *args:
    just --justfile packages/zarr-indexing/justfile "$@"

# Run a zarr-http-server recipe, or list its recipes with no arguments
zarr-http-server *args:
    just --justfile packages/zarr-http-server/justfile "$@"
