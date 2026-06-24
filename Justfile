# zarr-python developer task runner (https://github.com/casey/just).
#
# `just` is a thin verb-runner over uv; uv owns the environments. Each recipe is
# the single source of truth for a dev/CI task — CI calls the same recipes (via
# `uvx --from rust-just just <recipe>`), so local and CI behavior cannot drift.
#
# Install just:  uv tool install rust-just   (or `brew install just`, `cargo install just`)
# List recipes:  just         (or `just --list`)
#
# The matrix lives in GitHub Actions; pass the Python version via UV_PYTHON
# (setup-uv sets it from `python-version`). Locally, override per call, e.g.
#   UV_PYTHON=3.13 just test-optional

# Extras + groups that make up the "optional" (full integration) test environment.
optional_deps := "--extra remote --extra optional --extra cli --extra cast-value-rs --group remote-tests"

[private]
default:
    @just --list

[doc("Run the unit tests with the minimal dependency set")]
test-minimal *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --group test
    if [ -n "${CI:-}" ]; then uv pip list; fi
    uv run --no-sync coverage run --source=src -m pytest --ignore tests/benchmarks \
        --junitxml=junit.xml -o junit_family=legacy {{args}}
    uv run --no-sync coverage xml

[doc("Run the unit tests with the full (optional) integration dependency set")]
test-optional *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --group test {{optional_deps}}
    if [ -n "${CI:-}" ]; then uv pip list; fi
    uv run --no-sync coverage run --source=src -m pytest --ignore tests/benchmarks \
        --junitxml=junit.xml -o junit_family=legacy {{args}}
    uv run --no-sync coverage xml

[doc("Generate an HTML coverage report (optional deps); open htmlcov/index.html")]
coverage-html *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --group test {{optional_deps}}
    uv run --no-sync coverage run --source=src -m pytest --ignore tests/benchmarks {{args}}
    uv run --no-sync coverage html

[doc("Run the slow Hypothesis property tests")]
hypothesis *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --group test {{optional_deps}}
    if [ -n "${CI:-}" ]; then uv pip list; fi
    uv run --no-sync coverage run --source=src -m pytest -nauto --run-slow-hypothesis \
        tests/test_properties.py tests/test_store/test_stateful* {{args}}
    uv run --no-sync coverage xml

[doc("Validate executable code blocks in the docs (tests/test_docs.py)")]
doctest *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --extra remote --group remote-tests
    if [ -n "${CI:-}" ]; then uv pip list; fi
    uv run --no-sync --with pytest-examples pytest tests/test_docs.py -v {{args}}

[doc("Run the benchmark suite (minimal deps)")]
benchmark *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --group test
    uv run --no-sync pytest --benchmark-enable tests/benchmarks {{args}}

[doc("Run the tests against the lowest supported direct dependency versions")]
min_deps *args:
    #!/usr/bin/env bash
    set -euo pipefail
    # uv derives the floors from the `>=` constraints in pyproject.toml.
    uv sync --resolution lowest-direct --no-default-groups \
        --group test --group remote-tests --extra remote --extra optional
    if [ -n "${CI:-}" ]; then uv pip list; fi
    uv run --no-sync coverage run --source=src -m pytest --ignore tests/benchmarks \
        --junitxml=junit.xml -o junit_family=legacy {{args}}
    uv run --no-sync coverage xml

[doc("Run the tests against bleeding-edge (nightly + git main) dependencies")]
upstream *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --no-default-groups --group test --group remote-tests --extra remote
    uv pip install --prerelease=allow \
        --index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        numpy \
        "packaging @ git+https://github.com/pypa/packaging" \
        "numcodecs @ git+https://github.com/zarr-developers/numcodecs" \
        "s3fs @ git+https://github.com/fsspec/s3fs" \
        "universal_pathlib @ git+https://github.com/fsspec/universal_pathlib" \
        "typing_extensions @ git+https://github.com/python/typing_extensions" \
        "donfig @ git+https://github.com/pytroll/donfig" \
        "obstore @ git+https://github.com/developmentseed/obstore@main#subdirectory=obstore"
    if [ -n "${CI:-}" ]; then uv pip list; fi
    uv run --no-sync coverage run --source=src -m pytest --ignore tests/benchmarks \
        --junitxml=junit.xml -o junit_family=legacy {{args}}
    uv run --no-sync coverage xml

[doc("Run the GPU tests (requires CUDA + a GPU); `pytest -m gpu`")]
gpu *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --group test --extra gpu --extra optional
    uv pip install pytest-examples
    if [ -n "${CI:-}" ]; then uv pip list; fi
    uv run --no-sync coverage run --source=src -m pytest -m gpu --ignore tests/benchmarks \
        --junitxml=junit.xml -o junit_family=legacy {{args}}
    uv run --no-sync coverage xml

[doc("Build the documentation (strict: warnings are errors)")]
docs-build *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --extra remote --group docs
    DISABLE_MKDOCS_2_WARNING=true NO_MKDOCS_2_WARNING=true \
        uv run --no-sync mkdocs build --strict {{args}}

[doc("Serve the documentation locally with live reload at http://0.0.0.0:8000/")]
docs-serve *args:
    #!/usr/bin/env bash
    set -euo pipefail
    uv sync --locked --no-default-groups --extra remote --group docs
    DISABLE_MKDOCS_2_WARNING=true NO_MKDOCS_2_WARNING=true \
        uv run --no-sync mkdocs serve --watch src {{args}}

[doc("Run all pre-commit hooks (ruff, codespell, mypy, repo-review, ...)")]
lint *args:
    prek run --all-files {{args}}

[doc("Check that uv.lock is in sync with pyproject.toml")]
lock-check:
    uv lock --check
