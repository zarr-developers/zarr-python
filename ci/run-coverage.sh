#!/usr/bin/env bash
# Coverage-instrumented pytest run + XML report — the former hatch `run-coverage`
# script, shared by the test / upstream / min_deps / gpu CI jobs and runnable
# locally. Assumes the target environment is already synced (e.g.
# `uv sync --locked --group test`). Extra args are forwarded to pytest, e.g.
#   bash ci/run-coverage.sh -m gpu
set -euo pipefail

# In CI, log the resolved environment so version issues are debuggable.
# In GitHub Actions, the CI environment variable is always set to true.
if [ -n "${CI:-}" ]; then uv pip list; fi

uv run --no-sync coverage run --source=src -m pytest \
  --ignore tests/benchmarks \
  --junitxml=junit.xml -o junit_family=legacy "$@"
uv run --no-sync coverage xml
