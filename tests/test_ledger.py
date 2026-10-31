"""The ledger of non-conformant and unregistered metadata matches the source.

`zarr.core.metadata.ledger.LEDGER` lists every kind of stored metadata zarr reads or
writes that does not conform to the Zarr specifications or is not registered, and the
Metadata compatibility page is generated from it. The code that reads or writes each
kind carries a `# NON-CONFORMANT: <key>` or `# UNREGISTERED: <key>` comment. These tests
fail if the code and the ledger disagree, so neither can change without the other.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import re
import tokenize
from pathlib import Path

import zarr
from zarr.core.metadata.ledger import LEDGER, MARKERS
from zarr.core.metadata.repair import ARRAY_REPAIRS

SRC = Path(zarr.__file__).parent
ROOT = SRC.parent.parent
MARKER = re.compile(r"#\s*(NON-CONFORMANT|UNREGISTERED):\s*(\S*)")


def _markers() -> list[tuple[str, int, str, str]]:
    """Every marker comment in `src/zarr`: (file, line, tag, key)."""
    return [
        (str(path.relative_to(SRC)), token.start[0], match[1], match[2])
        for path in sorted(SRC.rglob("*.py"))
        for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        if token.type == tokenize.COMMENT and (match := MARKER.match(token.string))
    ]


def test_every_repair_is_in_the_ledger() -> None:
    """Every repair applied to stored documents is listed, and every listed repair is
    applied."""
    applied = {repair for repairs in ARRAY_REPAIRS.values() for repair in repairs}
    listed = {repair for entry in LEDGER.values() for repair in entry.repairs}
    assert {r.__name__ for r in applied - listed} == set(), "repairs missing from the ledger"
    assert {r.__name__ for r in listed - applied} == set(), "ledger names unapplied repairs"


def test_every_marker_names_an_entry_of_its_conformance() -> None:
    wrong = [
        f"{file}:{line}: {tag}: {key!r}"
        for file, line, tag, key in _markers()
        if key not in LEDGER or MARKERS[LEDGER[key].conformance] != tag
    ]
    assert wrong == [], "markers that name no ledger entry, or the wrong tag:\n" + "\n".join(wrong)


def test_every_entry_has_a_marker() -> None:
    marked = {key for *_, key in _markers()}
    assert set(LEDGER) - marked == set(), "ledger entries no code is marked with"


def test_unstable_data_type_warnings_are_marked() -> None:
    """A data type that warns that it has no Zarr format 3 specification writes
    unregistered or non-conformant metadata, so its warning carries a marker on the line
    before it."""
    marked = {(file, line) for file, line, *_ in _markers()}
    unmarked = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path in sorted(SRC.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "v3_unstable_dtype_warning"
        and (str(path.relative_to(SRC)), node.lineno - 1) not in marked
    ]
    assert unmarked == []


def test_metadata_compatibility_page_lists_the_ledger() -> None:
    """The docs page holds the marker the docs hook replaces, and the hook renders every
    entry."""
    page = (ROOT / "docs" / "metadata-compatibility.md").read_text()
    spec = importlib.util.spec_from_file_location("mkdocs_hooks", ROOT / "mkdocs_hooks.py")
    assert spec is not None
    assert spec.loader is not None
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    assert hooks._LEDGER_MARKER in page
    rendered = hooks.ledger_markdown()
    assert [entry.title for entry in LEDGER.values() if entry.title not in rendered] == []
