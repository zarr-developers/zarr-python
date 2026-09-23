from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
script_paths = tuple((REPO_ROOT / "examples").rglob("*.py"))

PEP_723_REGEX: Final = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"


def read_script_metadata(script: str) -> dict[str, object]:
    """
    Read the PEP-723 "script" metadata block of a script.

    Source code modified from
    https://packaging.python.org/en/latest/specifications/inline-script-metadata/#reference-implementation
    """
    match = re.search(PEP_723_REGEX, script)
    if match is None or match.group("type") != "script":
        raise ValueError("PEP-723 script metadata not found")
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in match.group("content").splitlines(keepends=True)
    )
    return tomllib.loads(content)


def test_script_paths() -> None:
    """
    Test that our test fixture is working properly and collecting script paths.
    """
    assert len(script_paths) > 0


@pytest.mark.parametrize("script_path", script_paths, ids=lambda p: p.stem)
def test_script_uses_local_zarr(script_path: Path) -> None:
    """
    An example installs zarr from the checkout it is in, so running it tests this zarr rather than
    a release or a branch on GitHub.
    """
    metadata = read_script_metadata(script_path.read_text())
    assert "zarr" in metadata["dependencies"]  # type: ignore[operator]
    source = metadata["tool"]["uv"]["sources"]["zarr"]  # type: ignore[index]
    assert (script_path.parent / source["path"]).resolve() == REPO_ROOT


@pytest.mark.skipif(
    sys.platform == "win32", reason="This test fails for unknown reasons on Windows in CI."
)
@pytest.mark.parametrize("script_path", script_paths, ids=lambda p: p.stem)
def test_scripts_can_run(script_path: Path) -> None:
    # The script's own metadata installs zarr from this checkout (see test_script_uses_local_zarr),
    # so it runs as a user running it from a checkout would.
    result = subprocess.run(
        ["uv", "run", str(script_path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"Script at {script_path} failed to run. Output: {result.stdout} Error: {result.stderr}"
    )
