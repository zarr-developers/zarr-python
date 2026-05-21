"""Tests for the git-tag-based version derivation configured in `pyproject.toml`.

zarr-python derives its version from git tags via `hatch-vcs`, which runs a
`git describe` command configured under `[tool.hatch] version.raw-options`.
Because this repository is a monorepo that also releases the `zarr-metadata`
subpackage using `zarr_metadata-v*` tags, the describe command must be
restricted to zarr-python's own `v*` tags. Otherwise a newer
`zarr_metadata-v*` tag hijacks the derived version (e.g. zarr reports
`0.2.x` instead of `3.x`), which silently breaks downstream version gates.

These tests pin that behavior by running the *actual configured command*
against synthetic repositories, so they would catch a regression in the
config without depending on the ambient repository's tag state.
"""

from __future__ import annotations

import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).parent.parent


def _configured_describe_command() -> list[str]:
    """The `git describe` command zarr-python configures for version derivation.

    Read from `pyproject.toml` so this test exercises the real config rather
    than a hand-retyped copy that could drift from it.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    raw_options = pyproject["tool"]["hatch"]["version"]["raw-options"]
    command = raw_options["git_describe_command"]
    if isinstance(command, str):
        return shlex.split(command)
    # setuptools_scm also accepts a pre-split list
    return list(command)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")


def _commit(repo: Path, message: str) -> None:
    # Touch a file so each commit has content.
    (repo / "file.txt").write_text(message)
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", message)


def _run_describe(repo: Path, command: Sequence[str]) -> str:
    """Run the configured describe command (minus the leading `git`) in *repo*."""
    # The configured command starts with "git"; drop it and re-target at *repo*.
    assert command[0] == "git", f"expected a git command, got {command!r}"
    return _git(repo, *command[1:])


def test_configured_command_targets_v_tags() -> None:
    """The configured describe command restricts matching to `v*` tags.

    A lightweight guard so the intent is obvious even before the behavioral
    test below runs.
    """
    command = _configured_describe_command()
    assert "--match" in command
    match_value = command[command.index("--match") + 1]
    assert match_value == "v*"


def test_configured_command_includes_unannotated_tags() -> None:
    """The configured command must pass `--tags`.

    zarr-python's release tags (`v3.x`) are lightweight (unannotated), so a
    bare `git describe` ignores them and walks back to an old *annotated* tag
    (`v2.13.0a1`), producing a wildly wrong version. `--tags` is required for
    `git describe` to consider lightweight tags at all.
    """
    command = _configured_describe_command()
    assert "--tags" in command


def test_describe_ignores_newer_non_v_tag(tmp_path: Path) -> None:
    """A `zarr_metadata-v*` tag newer than the latest `v*` tag must not win.

    This is the exact scenario that mis-versioned zarr-python to `0.2.x` after
    the first `zarr_metadata-v0.2.0` release tag was pushed.
    """
    command = _configured_describe_command()
    repo = tmp_path / "repo"
    _init_repo(repo)

    _commit(repo, "first")
    _git(repo, "tag", "v3.0.0")
    _commit(repo, "second")
    # A subpackage release tag that is *newer* and uses a different scheme.
    _git(repo, "tag", "zarr_metadata-v9.9.9")

    described = _run_describe(repo, command)
    assert described.startswith("v3.0.0"), (
        f"version derivation should resolve to the latest v* tag, got {described!r}"
    )
    assert "zarr_metadata" not in described


def test_describe_resolves_latest_v_tag(tmp_path: Path) -> None:
    """With several `v*` tags, the most recent one wins (sanity check)."""
    command = _configured_describe_command()
    repo = tmp_path / "repo"
    _init_repo(repo)

    _commit(repo, "first")
    _git(repo, "tag", "v3.0.0")
    _commit(repo, "second")
    _git(repo, "tag", "v3.1.0")
    _commit(repo, "third")

    described = _run_describe(repo, command)
    assert described.startswith("v3.1.0"), f"expected the most recent v* tag, got {described!r}"


def test_describe_with_only_non_v_tags_finds_none(tmp_path: Path) -> None:
    """If only non-`v*` tags exist, the describe command finds no match.

    `git describe --match v*` exits non-zero when nothing matches; hatch-vcs
    then falls back to a `0.0`-style version. We assert the command does not
    silently resolve to a `zarr_metadata-v*` tag.
    """
    command = _configured_describe_command()
    repo = tmp_path / "repo"
    _init_repo(repo)

    _commit(repo, "first")
    _git(repo, "tag", "zarr_metadata-v0.2.0")

    result = subprocess.run(
        ["git", "-C", str(repo), *command[1:]],
        capture_output=True,
        text=True,
    )
    # Either it fails to find a match, or whatever it prints is not the
    # subpackage tag. Both outcomes mean the subpackage tag did not hijack
    # the version.
    assert "zarr_metadata" not in result.stdout
