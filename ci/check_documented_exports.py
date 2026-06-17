"""Check that every public top-level export is in the API reference.

The API reference is authored as explicit mkdocstrings directives (``::: target``)
under ``docs/api/`` -- one per documented symbol -- rather than autodoc, so a newly
added ``zarr.__all__`` entry will not appear in the docs until someone writes a page
for it (or it becomes a rendered member of an already-documented module). This script
catches that gap: it resolves every ``:::`` target, expands module directives into the
members they render (honoring ``members: false``), and asserts each name in
``zarr.__all__`` resolves to a documented object.

Usage:
    python ci/check_documented_exports.py

Raises ValueError if any public export is undocumented.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import zarr

REPO_ROOT = Path(__file__).parent.parent.resolve()
API_DOCS_ROOT = REPO_ROOT / "docs" / "api"

# Names in zarr.__all__ that are intentionally absent from the API reference.
# Keep this list short and justified -- it is the only escape hatch from the guard.
EXEMPT_EXPORTS = {
    "__version__",  # version string, not an API symbol
    "print_debug_info",  # debugging helper, deliberately not in the reference
}

# A mkdocstrings autodoc directive: `::: some.dotted.target` at the start of a line.
DIRECTIVE_RE = re.compile(r"^:::[ \t]+(?P<target>\S+)", re.MULTILINE)
# `members: false` (or `members: []`) within a directive's option block disables
# rendering of a module's members.
MEMBERS_DISABLED_RE = re.compile(r"^\s+members:\s*(false|\[\s*\])\s*$")


def resolve(target: str) -> Any:
    """Resolve a `:::` target (a dotted path) to the Python object it documents."""
    try:
        return importlib.import_module(target)
    except ImportError:
        pass
    module_path, _, attr = target.rpartition(".")
    try:
        return getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError):
        return None


def members_disabled(text: str, directive_start: int) -> bool:
    """Return True if the directive starting at `directive_start` sets members: false.

    Scans the indented option block immediately following the `:::` line, stopping at
    the first non-indented line (the end of this directive's block)."""
    for line in text[directive_start:].splitlines()[1:]:
        if line.strip() == "":
            continue
        if not line.startswith((" ", "\t")):
            break
        if MEMBERS_DISABLED_RE.match(line):
            return True
    return False


def documented_object_ids() -> set[int]:
    """Collect the id()s of every object rendered by a `:::` directive under docs/api.

    A directive pointing at an object documents that object. A directive pointing at a
    module documents the module's public members (its ``__all__`` if defined, else its
    public attributes) unless the directive sets ``members: false``."""
    documented: set[int] = set()
    for md_file in sorted(API_DOCS_ROOT.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        for match in DIRECTIVE_RE.finditer(text):
            obj = resolve(match.group("target"))
            if obj is None:
                continue
            documented.add(id(obj))
            if isinstance(obj, ModuleType) and not members_disabled(text, match.start()):
                member_names = getattr(obj, "__all__", None) or [
                    name for name in dir(obj) if not name.startswith("_")
                ]
                for name in member_names:
                    member = getattr(obj, name, None)
                    if member is not None:
                        documented.add(id(member))
    return documented


def find_undocumented_exports() -> list[str]:
    documented = documented_object_ids()
    missing = []
    for name in zarr.__all__:
        if name in EXEMPT_EXPORTS:
            continue
        if id(getattr(zarr, name)) not in documented:
            missing.append(name)
    return sorted(missing)


def main() -> None:
    if not API_DOCS_ROOT.exists():
        raise FileNotFoundError(f"{API_DOCS_ROOT} does not exist.")

    missing = find_undocumented_exports()
    if not missing:
        print(f"All {len(zarr.__all__)} public exports are documented.")
        return

    lines = [
        f"Found {len(missing)} public export(s) in zarr.__all__ missing from the API "
        "reference (docs/api/):\n",
    ]
    lines.extend(f"  - zarr.{name}" for name in missing)
    lines.append(
        "\nAdd a `::: zarr.<name>` page under docs/api/zarr/ (and register it in "
        "mkdocs.yml and docs/api/zarr/index.md), or -- if the export is intentionally "
        "undocumented -- add it to EXEMPT_EXPORTS in this script with a reason."
    )
    raise ValueError("\n".join(lines))


if __name__ == "__main__":
    main()
