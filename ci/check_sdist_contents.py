"""
Check a built sdist ships the paths it needs and none it must not.

Usage:
    python check_sdist_contents.py SDIST

SDIST is a path to a `.tar.gz`, or a directory containing exactly one.

`[tool.hatch.build.targets.sdist]` in pyproject.toml is an allowlist, so the
mistake to catch is a needed path that was never listed. Every `testpaths`
entry must ship, read from the pyproject.toml inside the tarball so the check
depends on the artifact alone. The paths below are needed but not covered by
`testpaths`.
"""

import sys
import tarfile
import tomllib
from pathlib import Path

# `_version.py` is gitignored and written into the sdist by the hatch-vcs hook.
REQUIRED_FILES = ["src/zarr/_version.py", "src/zarr/py.typed"]

# `tests/test_examples.py` globs `examples/` and skips silently if it is empty.
REQUIRED_DIRS = ["examples"]

# Each subpackage under `packages/` is released as its own distribution.
FORBIDDEN_PREFIXES = ["packages/"]


def read_sdist(sdist: Path) -> tuple[set[str], dict[str, object]]:
    """Return the sdist's paths, relative to its top-level directory, and its pyproject.toml."""
    with tarfile.open(sdist) as tar:
        names = tar.getnames()
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            raise SystemExit(
                f"Expected one top-level directory in {sdist.name}, got {sorted(roots)}"
            )
        root = roots.pop()
        pyproject = tar.extractfile(f"{root}/pyproject.toml")
        if pyproject is None:
            raise SystemExit(f"pyproject.toml is missing from {sdist.name}")
        config = tomllib.loads(pyproject.read().decode())
    members = {name.split("/", 1)[1] for name in names if "/" in name}
    return members, config


def check(sdist: Path) -> int:
    print(f"Checking {sdist.name}")
    members, config = read_sdist(sdist)
    testpaths = config["tool"]["pytest"]["ini_options"]["testpaths"]

    missing_files = [p for p in REQUIRED_FILES if p not in members]
    missing_dirs = [
        p for p in [*testpaths, *REQUIRED_DIRS] if not any(m.startswith(f"{p}/") for m in members)
    ]
    forbidden = sorted(m for m in members if m.startswith(tuple(FORBIDDEN_PREFIXES)))

    if not (missing_files or missing_dirs or forbidden):
        print(f"OK ({len(members)} paths)")
        return 0

    for heading, paths in [
        ("Missing files", missing_files),
        ("Missing directories", missing_dirs),
        ("Paths that must not ship", forbidden),
    ]:
        if paths:
            print(f"\n{heading}:")
            print("\n".join(paths[:20]))
            if len(paths) > 20:
                print(f"... and {len(paths) - 20} more")
    print("\nFix the allowlist in `[tool.hatch.build.targets.sdist]` in pyproject.toml.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"{target} does not exist; build the sdist first with `just build`")
        sys.exit(2)
    if target.is_dir():
        tarballs = sorted(target.glob("*.tar.gz"))
        if len(tarballs) != 1:
            print(f"Expected exactly one .tar.gz in {target}, found {len(tarballs)}")
            sys.exit(2)
        target = tarballs[0]
    sys.exit(check(target))
