"""Lint docstrings and Markdown for reStructuredText markup that won't render.

This project renders API docs with mkdocstrings (``docstring_style: numpy``) and prose
with MkDocs + Markdown -- not Sphinx/reStructuredText. RST constructs that survive from
older docstrings (or muscle memory) are not interpreted: a Sphinx role passes through as
literal text instead of becoming a link, an ``.. note::`` directive renders as a stray
line, and a ``:param:`` field list never becomes a documented parameter.

Crucially, none of this is caught by the rest of the docs CI. ``mkdocs build --strict``
sees the residue as ordinary prose (no warning), and ``ci/check_unlinked_types.py`` only
finds cross-references mkdocstrings *attempted* to resolve -- a raw ``:class:`` role is
never attempted, so it leaves no unlinked-type span. This linter fills that gap with a
fast, source-level check that needs no docs build.

Checks fall into two groups -- RST markup that silently fails under MkDocs/mkdocstrings,
and a Markdown structural problem that renders as valid-but-wrong HTML (so `mkdocs build`
emits no warning):

  sphinx-role     :class:`X`, :func:`X`, :py:meth:`X`   -> [`X`][zarr.X]
  rst-directive   .. note:: / .. code-block:: python    -> MkDocs admonition / fenced code
  rst-field       :param x:, :returns:, :rtype:          -> numpydoc Parameters/Returns/Raises
  rst-link        `text <https://example>`_              -> [text](https://example)
  list-break      unindented code fence between list items -> indent the fence under its item

The ``list-break`` check catches a fenced code block at column 0 placed *between* two list
items: because the fence is not indented into the preceding item, Markdown ends the list at
the fence and the following item starts a fresh list -- renumbering an ordered list (1, 1, 2
instead of 1, 2, 3) or breaking the grouping/spacing of any list. markdownlint's MD029 only
notices this for sequentially-numbered ordered lists; lazily-numbered (1., 1.) and unordered
lists slip past it, so this structural check covers the gap.

Usage:
    python ci/lint_docs.py [PATH ...]

PATH defaults to the repo-root ``src/zarr`` and ``docs``. Each PATH may be a file or a
directory (directories are searched for ``*.py`` and ``*.md``). Exits non-zero if any
issues are found.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_PATHS = (REPO_ROOT / "src" / "zarr", REPO_ROOT / "docs")

# A Sphinx interpreted-text role: an optional domain, a role name, then a backtick
# target -- e.g. :class:`Foo` or :py:meth:`Foo.bar`. Requires the trailing backtick so
# plain "::" (RST literal markers, time strings, mkdocs-material :icon: shortcodes) and
# URLs ("https://") never match.
SPHINX_ROLE = re.compile(r":[a-zA-Z_]\w*(?::[a-zA-Z_]\w*)?:`[^`\n]+`")

# An RST directive line: ".. name::" (with or without an argument after it). RST hyperlink
# targets (".. _label:") and comments (".. text") lack the "::" and are not flagged.
RST_DIRECTIVE = re.compile(r"^\s*\.\.[ \t]+[\w-]+::")

# An RST field-list entry used for docstring fields. The role names above (class, func,
# ...) are deliberately excluded so a role is reported as a role, not a field.
RST_FIELD = re.compile(
    r"^\s*:(param|parameter|arg|argument|key|keyword|kwarg|type|returns?|rtype"
    r"|raises?|except|exception|yields?|ytype|var|cvar|ivar)\b[^:]*:"
)

# An RST external hyperlink: `text <url>`_
RST_LINK = re.compile(r"`[^`\n]+<https?://[^>\n]+>`_")

# A list item at column 0: an ordered marker (1. / 1)) or a bullet (-, *, +) followed by
# whitespace and content. Leading-whitespace (nested/continuation) lines are intentionally
# not matched -- the list-break check only fires on top-level items.
LIST_ITEM = re.compile(r"^(?:\d+[.)]|[-*+])\s+\S")

CHECKS = (
    ("sphinx-role", SPHINX_ROLE),
    ("rst-directive", RST_DIRECTIVE),
    ("rst-field", RST_FIELD),
    ("rst-link", RST_LINK),
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    category: str
    snippet: str

    def format(self) -> str:
        try:
            location: Path | str = self.path.relative_to(REPO_ROOT)
        except ValueError:
            location = self.path
        return f"  {location}:{self.line}: [{self.category}] {self.snippet.strip()}"


def _scan_line(text: str) -> list[str]:
    """Return every RST-residue category found in a single line (a line can carry more
    than one, e.g. a role and an external link)."""
    return [category for category, pattern in CHECKS if pattern.search(text)]


def lint_python(path: Path) -> list[Finding]:
    """Scan the docstrings (module, classes, functions) of a Python file.

    Only docstrings are checked -- they are what mkdocstrings renders -- so RST-looking
    text inside ordinary code or string literals is never misreported."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - surfaced, not silently skipped
        return [Finding(path, exc.lineno or 0, "syntax-error", str(exc.msg))]

    doc_nodes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    # node.body[0].value is the docstring literal; its lineno is the line the string opens
    # on, so content line i maps to source line (start + i).
    docstrings = [
        (docstring, node.body[0].value.lineno)  # type: ignore[attr-defined]
        for node in ast.walk(tree)
        if isinstance(node, doc_nodes)
        if (docstring := ast.get_docstring(node, clean=False))
    ]
    return [
        Finding(path, start + offset, category, line)
        for docstring, start in docstrings
        for offset, line in enumerate(docstring.splitlines())
        for category in _scan_line(line)
    ]


def fenced_blocks(lines: list[str]) -> list[tuple[int, int, bool]]:
    """Index every fenced code block as ``(open_index, close_index, terminated)``, 0-based.

    ``terminated`` is False for a fence with no closing delimiter before EOF; its
    ``close_index`` is the last line so callers skipping code can skip to EOF. An
    unterminated fence is malformed Markdown that `mkdocs build` surfaces anyway."""
    blocks: list[tuple[int, int, bool]] = []
    fence: str | None = None
    open_idx = -1
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if fence is None:
            if stripped.startswith(("```", "~~~")):
                fence, open_idx = stripped[:3], i
        elif stripped.startswith(fence):
            blocks.append((open_idx, i, True))
            fence = None
    if fence is not None:
        blocks.append((open_idx, len(lines) - 1, False))
    return blocks


def find_list_breaking_fences(
    lines: list[str], blocks: list[tuple[int, int, bool]]
) -> list[tuple[int, str]]:
    """Return ``(lineno, snippet)`` for each fenced code block at column 0 that splits a
    list -- i.e. one whose nearest non-blank neighbours on both sides are top-level list
    items. Such a fence is not indented into the preceding item, so Markdown closes the
    list at the fence and the following item starts a new one. The fix is to indent the
    fence (4 spaces) so it nests inside its list item. See the module docstring.

    Conservative on purpose: it requires a list item *directly* before and after (a
    continuation line or paragraph in between is not matched), keeping false positives low
    for a check that fails CI. Unterminated fences are ignored."""

    def neighbour(start: int, step: int) -> str | None:
        j = start + step
        while 0 <= j < len(lines):
            if lines[j].strip():
                return lines[j]
            j += step
        return None

    def splits_a_list(open_i: int, close_i: int) -> bool:
        if lines[open_i][:1].isspace():
            return False  # indented fence: already nested in the list item, not a break
        before = neighbour(open_i, -1)
        after = neighbour(close_i, +1)
        return bool(before and after and LIST_ITEM.match(before) and LIST_ITEM.match(after))

    return [
        (open_i + 1, lines[open_i])
        for open_i, close_i, terminated in blocks
        if terminated and splits_a_list(open_i, close_i)
    ]


def lint_markdown(path: Path) -> list[Finding]:
    """Scan a Markdown file: RST residue in prose (skipping fenced code blocks), plus
    fenced code blocks that break a list (see find_list_breaking_fences)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = fenced_blocks(lines)
    in_code = {i for open_i, close_i, _ in blocks for i in range(open_i, close_i + 1)}

    prose = [
        Finding(path, lineno, category, line)
        for lineno, line in enumerate(lines, start=1)
        if lineno - 1 not in in_code
        for category in _scan_line(line)
    ]
    breaks = [
        Finding(path, lineno, "list-break", snippet)
        for lineno, snippet in find_list_breaking_fences(lines, blocks)
    ]
    return prose + breaks


def iter_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
            files.extend(sorted(path.rglob("*.md")))
        else:
            raise FileNotFoundError(f"{path} does not exist")
    return files


LINTERS = {".py": lint_python, ".md": lint_markdown}


def lint(paths: tuple[Path, ...]) -> list[Finding]:
    return [
        finding
        for file in iter_files(paths)
        if file.suffix in LINTERS
        for finding in LINTERS[file.suffix](file)
    ]


def main() -> int:
    args = sys.argv[1:]
    paths = tuple(Path(a).resolve() for a in args) if args else DEFAULT_PATHS
    findings = lint(paths)

    if not findings:
        print("No reStructuredText residue or list-breaking fences found in docs.")
        return 0

    print(
        f"Found {len(findings)} docs issue(s) -- RST markup that will not render under "
        "MkDocs/mkdocstrings, or Markdown that renders as valid-but-wrong HTML:\n",
        file=sys.stderr,
    )
    for finding in findings:
        print(finding.format(), file=sys.stderr)
    print(
        "\nFix each issue (see ci/lint_docs.py header):\n"
        "  sphinx-role   :class:`X`           -> [`X`][zarr.X]\n"
        "  rst-directive .. note::            -> MkDocs admonition (!!! note)\n"
        "  rst-field     :param x:            -> numpydoc Parameters/Returns/Raises section\n"
        "  rst-link      `text <url>`_        -> [text](url)\n"
        "  list-break    fence between items  -> indent the fence 4 spaces to nest it",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
