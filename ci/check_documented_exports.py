"""Check that every public top-level export is in the API reference.

The API reference is authored as explicit mkdocstrings directives (``::: target``)
under ``docs/api/`` -- one per documented symbol -- rather than autodoc, so a newly
added ``zarr.__all__`` entry will not appear in the docs until someone writes a page
for it (or it becomes a rendered member of an already-documented module). This script
catches that gap: it resolves every ``:::`` target, expands module directives into the
members they render (honoring ``members: false``), and asserts each name in
``zarr.__all__`` resolves to a documented object.

Usage:
    python ci/check_documented_exports.py [API_DOCS_DIR]

API_DOCS_DIR defaults to the repo-root ``docs/api``. Exits non-zero (and prints the
undocumented exports to stderr) if any public export is missing from the reference.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import zarr

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_API_DOCS_ROOT = REPO_ROOT / "docs" / "api"

# Names in zarr.__all__ that are intentionally absent from the API reference.
# Keep this list short and justified -- it is the only escape hatch from the guard.
EXEMPT_EXPORTS = {
    "__version__",  # version string, not an API symbol
    "print_debug_info",  # debugging helper, deliberately not in the reference
}

# A mkdocstrings autodoc directive: `::: some.dotted.target` at the start of a line.
DIRECTIVE_RE = re.compile(r"^:::[ \t]+(?P<target>\S+)")
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


def iter_directives(text: str) -> Iterator[tuple[str, bool]]:
    """Yield ``(target, members_enabled)`` for each ``:::`` directive in ``text``.

    The file is split into lines once; for each directive we scan its indented option
    block -- stopping at the first non-indented line, which ends the block -- so options
    belonging to a later directive are never consulted. ``members_enabled`` is False when
    that block sets ``members: false`` (or ``members: []``)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = DIRECTIVE_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        members_enabled = True
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                i += 1
                continue
            if not line.startswith((" ", "\t")):
                break  # non-indented line: end of this directive's option block
            if MEMBERS_DISABLED_RE.match(line):
                members_enabled = False
            i += 1
        yield match.group("target"), members_enabled


def module_member_ids(module: ModuleType) -> Iterator[int]:
    """Yield the id() of each public member a module directive renders.

    The rendered members are the module's ``__all__`` if defined, else its public
    (non-underscore) attributes."""
    member_names = getattr(module, "__all__", None) or [
        name for name in dir(module) if not name.startswith("_")
    ]
    for name in member_names:
        member = getattr(module, name, None)
        if member is not None:
            yield id(member)


def documented_object_ids(api_docs_root: Path) -> set[int]:
    """Collect the id()s of every object rendered by a `:::` directive under api_docs_root.

    A directive pointing at an object documents that object. A directive pointing at a
    module documents the module's public members unless the directive sets
    ``members: false``."""
    documented: set[int] = set()
    for md_file in sorted(api_docs_root.rglob("*.md")):
        for target, members_enabled in iter_directives(md_file.read_text(encoding="utf-8")):
            obj = resolve(target)
            if obj is None:
                continue
            documented.add(id(obj))
            if isinstance(obj, ModuleType) and members_enabled:
                documented.update(module_member_ids(obj))
    return documented


def find_undocumented_exports(api_docs_root: Path) -> list[str]:
    documented = documented_object_ids(api_docs_root)
    return sorted(
        name
        for name in zarr.__all__
        if name not in EXEMPT_EXPORTS and id(getattr(zarr, name)) not in documented
    )


def main() -> int:
    args = sys.argv[1:]
    api_docs_root = Path(args[0]).resolve() if args else DEFAULT_API_DOCS_ROOT
    if not api_docs_root.exists():
        print(f"{api_docs_root} does not exist.", file=sys.stderr)
        return 1

    missing = find_undocumented_exports(api_docs_root)
    if not missing:
        print(f"All {len(zarr.__all__)} public exports are documented.")
        return 0

    print(
        f"Found {len(missing)} public export(s) in zarr.__all__ missing from the API "
        "reference (docs/api/):\n",
        file=sys.stderr,
    )
    for name in missing:
        print(f"  - zarr.{name}", file=sys.stderr)
    print(
        "\nAdd a `::: zarr.<name>` page under docs/api/zarr/ (and register it in "
        "mkdocs.yml and docs/api/zarr/index.md), or -- if the export is intentionally "
        "undocumented -- add it to EXEMPT_EXPORTS in this script with a reason.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
