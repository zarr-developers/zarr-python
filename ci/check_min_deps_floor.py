"""
Enforce the invariant: `min_deps` pins zarr-metadata to the floor of
zarr-python's declared zarr-metadata range.

zarr-python declares `zarr-metadata>=X.Y.Z,<...>` in `[project.dependencies]`.
The `min_deps` hatch env tests against the *minimum* supported deps, so it
must pin zarr-metadata to exactly that floor (e.g. `zarr-metadata==X.Y.Z`).
Without this script the two declarations can drift silently — the project's
floor could rise without `min_deps` noticing, and `min_deps` would no longer
verify what its name claims.

Run:
    python ci/check_min_deps_floor.py

Exits 0 if floors agree; non-zero with a clear message if not.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
PYPROJECT = ROOT / "pyproject.toml"

# Match `>=X.Y.Z` (with or without surrounding whitespace) inside a PEP 440
# version specifier set. Captures just the version number.
_FLOOR_RE = re.compile(r">=\s*([^,\s]+)")
# Match `==X.Y.Z` likewise. Captures the version number.
_PIN_RE = re.compile(r"==\s*([^,\s]+)")


def find_zarr_metadata_floor(deps: list[str]) -> str:
    """Return the >= floor of zarr-metadata declared in `deps`.

    `deps` is a list of PEP 508 strings, e.g. as found in
    `[project.dependencies]`. Raises if zarr-metadata is not present, or
    if its specifier set has no `>=` bound.
    """
    for dep in deps:
        # Project name is everything up to the first non-name character.
        # Quick split: package name terminates at the first occurrence of a
        # version operator, whitespace, `[`, `;`, or `(`.
        name = re.split(r"[<>=!~\s\[;(]", dep, maxsplit=1)[0].strip()
        if name == "zarr-metadata":
            match = _FLOOR_RE.search(dep)
            if not match:
                raise SystemExit(
                    f"zarr-metadata dependency has no `>=` floor: {dep!r}\n"
                    "Floor verification requires an explicit lower bound."
                )
            return match.group(1)
    raise SystemExit(
        "zarr-metadata not found in [project.dependencies]. "
        "This script assumes zarr-python depends on zarr-metadata."
    )


def find_zarr_metadata_pin(deps: list[str]) -> str:
    """Return the `==` pin of zarr-metadata declared in `deps`.

    `deps` is a list of PEP 508 strings, e.g. as found in
    `[tool.hatch.envs.min_deps.extra-dependencies]`. Raises if
    zarr-metadata is not present, or if its specifier is not a `==` pin.
    """
    for dep in deps:
        name = re.split(r"[<>=!~\s\[;(]", dep, maxsplit=1)[0].strip()
        if name == "zarr-metadata":
            match = _PIN_RE.search(dep)
            if not match:
                raise SystemExit(
                    f"min_deps zarr-metadata entry is not an `==` pin: {dep!r}\n"
                    "The min_deps env must pin zarr-metadata exactly to the floor."
                )
            return match.group(1)
    raise SystemExit(
        "zarr-metadata not found in [tool.hatch.envs.min_deps.extra-dependencies].\n"
        "Add `'zarr-metadata==<floor>'` to keep min_deps testing the declared floor."
    )


def main() -> int:
    data = tomllib.loads(PYPROJECT.read_text())

    project_deps = data["project"]["dependencies"]
    floor = find_zarr_metadata_floor(project_deps)

    min_deps_extra = data["tool"]["hatch"]["envs"]["min_deps"]["extra-dependencies"]
    pin = find_zarr_metadata_pin(min_deps_extra)

    if floor != pin:
        print(
            f"floor / min_deps pin mismatch for zarr-metadata:\n"
            f"  [project.dependencies] floor:           >={floor}\n"
            f"  [tool.hatch.envs.min_deps] pin:         =={pin}\n"
            f"\n"
            f"These must agree. Either update the floor in "
            f"[project.dependencies] or the pin in min_deps so both name "
            f"the same zarr-metadata version.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: zarr-metadata floor {floor} matches min_deps pin {pin}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
