"""The python code blocks in the README and the docs run without error.

As in zarr-python's docs tests, a block that runs declares `test="true"`, and
a `session` name. The blocks of one session in one file run in order, in one
namespace, so a block can use what an earlier block of its session defined. A
block that cannot run says so with `exec="false"` and a `reason`.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from pathlib import Path

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

import zarr_metadata.definition
from zarr_metadata.model import ZarrV3ArrayMetadata

PACKAGE_ROOT = Path(__file__).parent.parent
SOURCES = (PACKAGE_ROOT / "README.md", PACKAGE_ROOT / "docs")


def _examples() -> list[CodeExample]:
    """The python blocks of the README and the docs, each file's blocks in the order they are written."""
    return list(find_examples(*SOURCES))


def _sessions() -> dict[tuple[Path, str], list[CodeExample]]:
    """The blocks that declare `test="true"`, grouped by file and session."""
    sessions: defaultdict[tuple[Path, str], list[CodeExample]] = defaultdict(list)
    for example in _examples():
        settings = example.prefix_settings()
        if settings.get("test") == "true":
            sessions[example.path, settings.get("session", "_default")].append(example)
    return dict(sessions)


SESSIONS = _sessions()


@pytest.mark.parametrize(
    "key",
    list(SESSIONS),
    ids=[f"{path.relative_to(PACKAGE_ROOT)}:{session}" for path, session in SESSIONS],
)
def test_each_session_of_code_blocks_runs(
    key: tuple[Path, str],
    eval_example: EvalExample,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blocks of each session run in order, in one namespace, without an error. They run in a temporary working directory, since a block may write a file, such as a JSON Schema. The directory holds the `zarr.json` of an array, which a block may read, as a reader reads one from disk."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "zarr.json").write_bytes(
        ZarrV3ArrayMetadata.create_default(shape=(4,)).to_key_value()["zarr.json"]
    )
    module_globals: dict[str, object] = {}
    for example in SESSIONS[key]:
        module_globals.update(eval_example.run(example, module_globals=module_globals))


def test_every_python_block_says_whether_it_runs() -> None:
    """Every python block in the README and the docs declares `test="true"`, or `exec="false"` with a reason, so a block cannot skip these tests by accident."""
    unmarked = [
        f"{example.path.relative_to(PACKAGE_ROOT)}:{example.start_line}"
        for example in _examples()
        if example.prefix_settings().get("test") != "true"
        and not (
            example.prefix_settings().get("exec") == "false"
            and example.prefix_settings().get("reason", "").strip() != ""
        )
    ]
    assert unmarked == []


def test_the_extensions_page_example_is_the_definition_module_docstring_example() -> None:
    """The acme.lz4 example on the extensions page is, line for line, the example in the `zarr_metadata.definition` module docstring. Only the page's copy runs under these tests, so this is what keeps the docstring's copy from drifting."""
    page_blocks = [
        example
        for example in _examples()
        if example.path == PACKAGE_ROOT / "docs" / "extensions.md"
        and example.prefix_settings().get("session") == "extensions"
    ]
    assert len(page_blocks) == 1, (
        "expected exactly one extensions-session block on docs/extensions.md"
    )
    docstring = zarr_metadata.definition.__doc__
    assert docstring is not None, "zarr_metadata.definition has no module docstring"
    lines = docstring.splitlines()
    marker = "from collections.abc import Iterator"
    starts = [i for i, line in enumerate(lines) if line.strip() == marker]
    assert len(starts) == 1, f"expected one line {marker!r} in the docstring, found {len(starts)}"
    assert lines[starts[0] - 1].strip() == "```python", (
        f"expected {marker!r} to open a ```python fence in the docstring"
    )
    end = starts[0]
    while end < len(lines) and lines[end].strip() != "```":
        end += 1
    assert end < len(lines), "the docstring's ```python fence is not closed"
    docstring_example = textwrap.dedent("\n".join(lines[starts[0] : end])).strip()
    assert page_blocks[0].source.strip().splitlines() == docstring_example.splitlines()
