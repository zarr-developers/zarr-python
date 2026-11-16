from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ZARR_PROJECT_PATH = Path(__file__).parent.parent
EXAMPLE_PATHS = tuple(sorted((ZARR_PROJECT_PATH / "examples").glob("*/*.py")))


def test_example_paths() -> None:
    """The example scripts are found, so the parametrized test below is not empty."""
    assert len(EXAMPLE_PATHS) > 0


@pytest.mark.examples
@pytest.mark.parametrize("script_path", EXAMPLE_PATHS, ids=lambda path: path.stem)
def test_example_runs(script_path: Path, tmp_path: Path) -> None:
    """
    Each example runs against this checkout, in the environment its PEP 723 header defines.

    The test copies the script and uses `uv add` to point the copy's zarr requirement at
    this checkout. `uv add` also resolves the header, so a header that cannot be
    satisfied fails here.
    """
    dest_path = tmp_path / script_path.name
    dest_path.write_text(script_path.read_text())

    add = subprocess.run(
        ["uv", "add", "--quiet", "--script", str(dest_path), str(ZARR_PROJECT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert add.returncode == 0, f"uv add failed for {script_path}:\n{add.stderr}"

    # `--reinstall-package` rebuilds zarr, so the run sees the current source.
    run = subprocess.run(
        ["uv", "run", "--reinstall-package", "zarr", str(dest_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, f"{script_path} failed:\n{run.stdout}\n{run.stderr}"
