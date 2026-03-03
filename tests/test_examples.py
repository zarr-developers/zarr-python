from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
import tomlkit
from packaging.requirements import Requirement

examples_dir = "examples"
script_paths = tuple(Path(examples_dir).rglob("*.py"))

PEP_723_REGEX: Final = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"

# This is the absolute path to the local Zarr installation. Moving this test to a different directory will break it.
ZARR_PROJECT_PATH = Path(".").absolute()


def _get_zarr_extras(script_path: Path) -> str:
    """Extract extras from the zarr dependency in a script's PEP 723 header.

    For example, if the script declares ``zarr[server]``, this returns ``[server]``.
    If the script declares ``zarr`` with no extras, this returns ``""``.
    """
    source_text = script_path.read_text()
    match = re.search(PEP_723_REGEX, source_text)
    if match is None:
        return ""

    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in match.group("content").splitlines(keepends=True)
    )
    config = tomlkit.parse(content)
    for dep in config.get("dependencies", []):
        req = Requirement(dep)
        if req.name == "zarr" and req.extras:
            return "[" + ",".join(sorted(req.extras)) + "]"
    return ""


def test_script_paths() -> None:
    """
    Test that our test fixture is working properly and collecting script paths.
    """
    assert len(script_paths) > 0


@pytest.mark.skipif(
    sys.platform == "win32", reason="This test fails for unknown reasons on Windows in CI."
)
@pytest.mark.parametrize("script_path", script_paths)
def test_scripts_can_run(script_path: Path) -> None:
    # Override the zarr dependency with the local project, preserving any extras
    # declared in the script's PEP 723 header (e.g. zarr[server]).
    extras = _get_zarr_extras(script_path)
    zarr_dep = f"zarr{extras} @ file:///{ZARR_PROJECT_PATH}"
    result = subprocess.run(
        ["uv", "run", "--with", zarr_dep, "--refresh", str(script_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Script at {script_path} failed to run. Output: {result.stdout} Error: {result.stderr}"
    )
