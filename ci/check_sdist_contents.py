"""
Check a built sdist ships everything its consumers need, and nothing they don't.

Usage:
    python check_sdist_contents.py SDIST

SDIST is a path to a `.tar.gz`, or a directory containing exactly one.

`[tool.hatch.build.targets.sdist]` in pyproject.toml is an allowlist. That is
the safe default -- a new subpackage under `packages/` stays out of the `zarr`
sdist unless someone opts it in -- but it inverts the failure mode: instead of
shipping too much silently, an allowlist can ship too little. This script
guards that direction.

Wherever possible the expectations are *derived* from configuration that is
maintained for other reasons, rather than restated here. A hand-written list of
"files the sdist must contain" would just be a second allowlist to forget to
update, which is the problem it is meant to solve.
"""

import sys
import tarfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()

# Paths no test opens, so the sdist test run in `releases.yml` cannot vouch for
# them, but that packagers do need. conda-forge's recipe installs the sdist
# (pyproject.toml, README.md, src/) and reads `license_file: LICENSE.txt`; the
# version file is written into the sdist by the hatch-vcs hook, because an
# unpacked sdist has no git history to derive a version from.
# https://github.com/conda-forge/zarr-feedstock/blob/main/recipe/recipe.yaml
REQUIRED_PATHS = [
    "pyproject.toml",
    "README.md",
    "LICENSE.txt",
    "PKG-INFO",
    "src/zarr/__init__.py",
    "src/zarr/py.typed",
    "src/zarr/_version.py",
]

# Each subpackage under `packages/` is its own distribution with its own PyPI
# release, its own version tags and its own sdist. Before the allowlist they
# were bundled into every `zarr` sdist -- 2.9M of unrelated sources -- which is
# the regression this check exists to prevent recurring.
FORBIDDEN_PREFIXES = ["packages/"]


def sdist_members(sdist: Path) -> set[str]:
    """Every path inside the tarball, relative to its top-level directory."""
    with tarfile.open(sdist) as tar:
        names = tar.getnames()
    # Members are `zarr-<version>/<path>`; strip the leading component so the
    # expectations below don't have to know the version.
    return {name.split("/", 1)[1] for name in names if "/" in name}


def testpaths() -> list[str]:
    """`testpaths` from pyproject.toml.

    Derived rather than duplicated: adding a directory to `testpaths` without
    adding it to the sdist allowlist is exactly the mistake this catches. It is
    also the mistake that was already live -- `docs/user-guide` has been a
    testpath while `docs/` was excluded, so `pytest` on an unpacked sdist died
    at collection.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        config = tomllib.load(f)
    return config["tool"]["pytest"]["ini_options"]["testpaths"]


def check(sdist: Path) -> int:
    print(f"Checking {sdist.name}")
    members = sdist_members(sdist)
    print(f"Found {len(members)} paths")
    print()

    missing_required = [p for p in REQUIRED_PATHS if p not in members]
    # A testpath is a directory; it is present if anything ships beneath it.
    missing_testpaths = [p for p in testpaths() if not any(m.startswith(f"{p}/") for m in members)]
    forbidden = sorted(m for m in members if any(m.startswith(p) for p in FORBIDDEN_PREFIXES))

    if not (missing_required or missing_testpaths or forbidden):
        print("OK")
        return 0

    if missing_required:
        print("Required paths missing from the sdist")
        print("-------------------------------------")
        print("\n".join(missing_required))
        print()
    if missing_testpaths:
        print("testpaths entries missing from the sdist")
        print("----------------------------------------")
        print("\n".join(missing_testpaths))
        print("`pytest` on an unpacked sdist will fail to collect these.")
        print()
    if forbidden:
        print("Paths that must not ship in the zarr sdist")
        print("------------------------------------------")
        print("\n".join(forbidden[:20]))
        if len(forbidden) > 20:
            print(f"... and {len(forbidden) - 20} more")
        print()

    print(
        "Fix by editing `[tool.hatch.build.targets.sdist]` in pyproject.toml. "
        "It is an allowlist: a path ships only if it is listed."
    )
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    target = Path(sys.argv[1]).resolve()
    if target.is_dir():
        tarballs = sorted(target.glob("*.tar.gz"))
        if len(tarballs) != 1:
            print(f"Expected exactly one .tar.gz in {target}, found {len(tarballs)}")
            sys.exit(2)
        target = tarballs[0]
    sys.exit(check(target))
