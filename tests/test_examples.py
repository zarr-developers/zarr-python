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
script_paths = Path(examples_dir).glob("*.py")

PEP_723_REGEX: Final = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"

# This is the absolute path to the local Zarr installation. Moving this test to a different directory will break it.
ZARR_PROJECT_PATH = Path(".").absolute()


def set_dep(script: str, dependency: str) -> str:
    """
    Set a dependency in a PEP-723 script header.
    If the package is already in the list, it will be replaced.
    If the package is not already in the list, it will be added.

    Source code modified from
    https://packaging.python.org/en/latest/specifications/inline-script-metadata/#reference-implementation
    """
    match = re.search(PEP_723_REGEX, script)

    if match is None:
        raise ValueError(f"PEP-723 header not found in {script}")

    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in match.group("content").splitlines(keepends=True)
    )

    config = tomlkit.parse(content)
    for idx, dep in enumerate(tuple(config["dependencies"])):
        if Requirement(dep).name == Requirement(dependency).name:
            config["dependencies"][idx] = dependency

    new_content = "".join(
        f"# {line}" if line.strip() else f"#{line}"
        for line in tomlkit.dumps(config).splitlines(keepends=True)
    )

    start, end = match.span("content")
    return script[:start] + new_content + script[end:]


def resave_script(source_path: Path, dest_path: Path) -> None:
    """
    Read a script from source_path and save it to dest_path after inserting the absolute path to the
    local Zarr project directory in the PEP-723 header.
    """
    source_text = source_path.read_text()
    dest_text = set_dep(source_text, f"zarr @ file:///{ZARR_PROJECT_PATH}")
    dest_path.write_text(dest_text)


@pytest.mark.skipif(
    sys.platform in ("win32",), reason="This test fails due for unknown reasons on Windows in CI."
)
@pytest.mark.parametrize("script_path", script_paths)
def test_scripts_can_run(script_path: Path, tmp_path: Path) -> None:
    dest_path = tmp_path / script_path.name
    # We resave the script after inserting the absolute path to the local Zarr project directory,
    # and then test its behavior.
    # This allows the example to be useful to users who don't have Zarr installed, but also testable.
    resave_script(script_path, dest_path)
    result = subprocess.run(["uv", "run", str(dest_path)], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Script at {script_path} failed to run. Output: {result.stdout} Error: {result.stderr}"
    )
