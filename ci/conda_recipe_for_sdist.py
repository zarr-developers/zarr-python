"""
Point the conda-forge recipe at a locally built sdist.

Usage:
    python conda_recipe_for_sdist.py RECIPE SDIST OUTPUT

The `zarr` sdist contents are an allowlist (see `[tool.hatch.build.targets.sdist]`
in pyproject.toml), so the failure mode to guard against is shipping too little.
conda-forge is the consumer most exposed to that: it builds from the sdist rather
than the wheel, and runs our test suite from a directory holding only what its
recipe copies out of the tarball.

Rather than restate that recipe here -- a copy would drift the moment the
feedstock changed -- CI fetches the real one and this script repoints it at the
sdist we just built. Everything that makes the check meaningful (the test files,
the pytest invocation, `pip_check`, `license_file`) comes from the feedstock
unmodified.

Note that only the parts of the recipe that identify *which* tarball to build are
touched. The feedstock's `requirements:` are regenerated from our pyproject.toml
by grayskull on each version bump (`bot: inspection: update-grayskull` in its
conda-forge.yml), so they already track this repo; the `tests:` block is
hand-maintained there and is exactly what we want to run verbatim.
"""

import hashlib
import sys
import tarfile
from pathlib import Path
from typing import Any

import yaml


def sdist_root(sdist: Path) -> str:
    """The tarball's single top-level directory, e.g. `zarr-3.3.1`."""
    with tarfile.open(sdist) as tar:
        roots = {name.split("/", 1)[0] for name in tar.getnames()}
    if len(roots) != 1:
        raise SystemExit(f"Expected one top-level directory in {sdist}, got {roots}")
    return roots.pop()


def conda_version(root: str) -> str:
    """Derive a conda-acceptable version from the sdist directory name.

    A build off a tag gives a clean `3.3.1`, but any other commit gives a
    setuptools-scm local version like `3.3.1.dev23+g52a63801`. Conda versions
    cannot contain `+`, and the local segment identifies the checkout rather
    than the release, so drop it. The value only labels the throwaway package
    this check builds; nothing is published from it.
    """
    version = root.split("-", 1)[1]
    return version.split("+", 1)[0]


def rewrite(recipe: dict[str, Any], sdist: Path) -> dict[str, Any]:
    root = sdist_root(sdist)
    context = recipe.setdefault("context", {})
    context["version"] = conda_version(root)
    # The recipe interpolates `${{ sha256 }}` into `source`, which is replaced
    # wholesale below; keep the key consistent anyway so a partially templated
    # recipe cannot silently reference a stale digest.
    context["sha256"] = hashlib.sha256(sdist.read_bytes()).hexdigest()

    # A `url` source rather than a `path` one, so this mirrors what conda-forge
    # actually does: fetch an archive and unpack it. It also sidesteps a trap --
    # `path` sources honour .gitignore by default, and `src/zarr/_version.py` is
    # gitignored. It is written into the sdist by the hatch-vcs build hook
    # precisely because an unpacked sdist has no git history to derive a version
    # from, so silently dropping it would break the build.
    recipe["source"] = {
        "url": sdist.resolve().as_uri(),
        "sha256": context["sha256"],
    }
    return recipe


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    recipe_path, sdist_path, output_path = (Path(p) for p in sys.argv[1:])

    with recipe_path.open("rb") as f:
        recipe = yaml.safe_load(f)

    recipe = rewrite(recipe, sdist_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(recipe, f, sort_keys=False)

    print(f"Wrote {output_path}")
    print(f"  version: {recipe['context']['version']}")
    print(f"  source:  {recipe['source']['url']}")
