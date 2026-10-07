"""`zarr.testing` must stay importable without its test dependencies.

pytest and hypothesis are not dependencies of zarr; they live in the `testing` extra. Each
check runs in a fresh interpreter where the missing packages are made unimportable by
setting their `sys.modules` entries to `None`, so the result does not depend on what the
test environment has installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def run_python(code: str, *, block: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter, with warnings as errors and `block` unimportable."""
    prelude = "import sys\n" + "".join(f"sys.modules[{name!r}] = None\n" for name in block)
    return subprocess.run(
        [sys.executable, "-W", "error", "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_imports_without_test_dependencies() -> None:
    """
    Without pytest or hypothesis, `zarr.testing` (which pytest loads as a plugin in
    every environment where zarr is installed) imports without warning.
    """
    result = run_python(
        """
        import zarr.testing
        import zarr.testing.buffer
        """,
        block=("pytest", "hypothesis"),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("statement", "missing"),
    [
        ("import zarr.testing.store", "pytest"),
        ("import zarr.testing.utils", "pytest"),
        ("from zarr.testing import StoreTests", "pytest"),
        ("from zarr.testing import assert_bytes_equal", "pytest"),
        ("import zarr.testing.strategies", "hypothesis"),
        ("import zarr.testing.stateful", "hypothesis"),
    ],
)
def test_missing_dependency_error(statement: str, missing: str) -> None:
    """A module that needs a missing test dependency names it and the extra that provides it."""
    result = run_python(statement, block=(missing,))
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert f"requires {missing}, which is not installed" in result.stderr
    assert "pip install 'zarr[testing]'" in result.stderr
