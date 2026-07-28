"""Check for unlinked type annotations in built documentation.

mkdocstrings renders resolved types as <a href="..."> links and unresolved
types as <span title="fully.qualified.Name">Name</span> without an anchor.
This script finds all such unlinked types in the built HTML and reports them.

Usage:
    python ci/check_unlinked_types.py [site_dir]

Raises ValueError if unlinked types are found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches the griffe/mkdocstrings pattern for unlinked cross-references:
# <span class="n"><span title="fully.qualified.Name">Name</span></span>
UNLINKED_PATTERN = re.compile(
    r'<span class="n"><span title="(?P<qualname>[^"]+)">(?P<name>[^<]+)</span></span>'
)

# Patterns to exclude from the report
EXCLUDE_PATTERNS = [
    # TypeVars and type parameters (single brackets like Foo[T])
    re.compile(r"\[.+\]$"),
    # Dataclass field / namedtuple field references (contain parens)
    re.compile(r"\("),
    # Private names
    re.compile(r"\._"),
    # Dunder attributes
    re.compile(r"\.__\w+__$"),
    # Testing utilities
    re.compile(r"^zarr\.testing\."),
    # Third-party types (hypothesis, pytest, etc.)
    re.compile(r"^(hypothesis|pytest|typing_extensions|builtins|dataclasses)\."),
]


def should_exclude(qualname: str) -> bool:
    return any(p.search(qualname) for p in EXCLUDE_PATTERNS)


def find_unlinked_types(site_dir: Path) -> dict[str, set[str]]:
    """Find all unlinked types in built HTML files.

    Returns a dict mapping qualified type names to the set of pages where they appear.
    """
    api_dir = site_dir / "api"
    if not api_dir.exists():
        raise FileNotFoundError(f"{api_dir} does not exist. Run 'mkdocs build' first.")

    unlinked: dict[str, set[str]] = {}
    for html_file in api_dir.rglob("*.html"):
        content = html_file.read_text(errors="replace")
        rel_path = str(html_file.relative_to(site_dir))
        for match in UNLINKED_PATTERN.finditer(content):
            qualname = match.group("qualname")
            if not should_exclude(qualname):
                unlinked.setdefault(qualname, set()).add(rel_path)

    return unlinked


def main() -> None:
    site_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("site")
    unlinked = find_unlinked_types(site_dir)

    if not unlinked:
        print("No unlinked types found.")
        return

    lines = [f"Found {len(unlinked)} unlinked types:\n"]
    for qualname in sorted(unlinked):
        pages = sorted(unlinked[qualname])
        lines.append(f"  {qualname}")
        lines.extend(f"    - {page}" for page in pages)

    all_pages = {p for ps in unlinked.values() for p in ps}
    lines.append(f"\nTotal: {len(unlinked)} unlinked types across {len(all_pages)} pages")
    report = "\n".join(lines)
    raise ValueError(report)


if __name__ == "__main__":
    main()
