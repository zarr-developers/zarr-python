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

Checks (all are RST syntax that silently fails under MkDocs/mkdocstrings):

  sphinx-role     :class:`X`, :func:`X`, :py:meth:`X`   -> [`X`][zarr.X]
  rst-directive   .. note:: / .. code-block:: python    -> MkDocs admonition / fenced code
  rst-field       :param x:, :returns:, :rtype:          -> numpydoc Parameters/Returns/Raises
  rst-link        `text <https://example>`_              -> [text](https://example)

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

    findings: list[Finding] = []
    doc_nodes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, doc_nodes):
            continue
        docstring = ast.get_docstring(node, clean=False)
        if not docstring:
            continue
        # node.body[0].value is the docstring literal; its lineno is the line the string
        # opens on, so content line i maps to source line (start + i).
        start = node.body[0].value.lineno  # type: ignore[attr-defined]
        for offset, line in enumerate(docstring.splitlines()):
            findings.extend(
                Finding(path, start + offset, category, line) for category in _scan_line(line)
            )
    return findings


def lint_markdown(path: Path) -> list[Finding]:
    """Scan a Markdown file, skipping fenced code blocks (``` or ~~~)."""
    findings: list[Finding] = []
    fence: str | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if fence is None and stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        findings.extend(Finding(path, lineno, category, line) for category in _scan_line(line))
    return findings


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


def lint(paths: tuple[Path, ...]) -> list[Finding]:
    findings: list[Finding] = []
    for file in iter_files(paths):
        if file.suffix == ".py":
            findings.append(lint_python(file))
        elif file.suffix == ".md":
            findings.append(lint_markdown(file))
    return [f for group in findings for f in group]


def main() -> int:
    args = sys.argv[1:]
    paths = tuple(Path(a).resolve() for a in args) if args else DEFAULT_PATHS
    findings = lint(paths)

    if not findings:
        print("No reStructuredText residue found in docstrings or Markdown.")
        return 0

    print(
        f"Found {len(findings)} reStructuredText construct(s) that will not render under "
        "MkDocs/mkdocstrings:\n",
        file=sys.stderr,
    )
    for finding in findings:
        print(finding.format(), file=sys.stderr)
    print(
        "\nReplace RST markup with its MkDocs equivalent (see ci/lint_docs.py header):\n"
        "  sphinx-role   :class:`X`           -> [`X`][zarr.X]\n"
        "  rst-directive .. note::            -> MkDocs admonition (!!! note)\n"
        "  rst-field     :param x:            -> numpydoc Parameters/Returns/Raises section\n"
        "  rst-link      `text <url>`_        -> [text](url)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    main()
