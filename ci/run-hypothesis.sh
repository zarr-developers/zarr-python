#!/usr/bin/env bash
# Slow Hypothesis run — the former hatch `run-hypothesis` script. Assumes the
# target environment is already synced. Extra args are forwarded to pytest.
set -euo pipefail

# In CI, log the resolved environment so version issues are debuggable.
# In GitHub Actions, the CI environment variable is always set to true.
if [ -n "${CI:-}" ]; then uv pip list; fi

uv run --no-sync coverage run --source=src -m pytest -nauto \
  --run-slow-hypothesis tests/test_properties.py tests/test_store/test_stateful* "$@"
uv run --no-sync coverage xml
