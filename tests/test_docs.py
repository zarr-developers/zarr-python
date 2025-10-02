"""
Tests for executable code blocks in markdown documentation.

This module uses pytest-examples to validate that all Python code examples
with exec="true" in the documentation execute successfully.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

pytest_examples = pytest.importorskip("pytest_examples")

# Find all markdown files with executable code blocks
docs_root = Path(__file__).parent.parent / "docs"


def find_markdown_files_with_exec() -> list[Path]:
    """Find all markdown files containing exec="true" code blocks."""
    markdown_files = []

    for md_file in docs_root.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if 'exec="true"' in content:
                markdown_files.append(md_file)
        except Exception:
            # Skip files that can't be read
            continue

    return sorted(markdown_files)


def group_examples_by_session() -> list[tuple[str, str]]:
    """
    Group examples by their session and file, maintaining order.

    Returns a list of session_key tuples where session_key is
    (file_path, session_name).
    """
    all_examples = list(pytest_examples.find_examples(docs_root))

    # Group by file and session
    sessions = defaultdict(list)

    for example in all_examples:
        settings = example.prefix_settings()
        if settings.get("exec") != "true":
            continue

        # Use file path and session name as key
        file_path = example.path
        session_name = settings.get("session", "_default")
        session_key = (str(file_path), session_name)

        sessions[session_key].append(example)

    # Return sorted list of session keys for consistent test ordering
    return sorted(sessions.keys(), key=lambda x: (x[0], x[1]))


def name_example(path: str, session: str) -> str:
    """Generate a readable name for a test case from file path and session."""
    return f"{Path(path).relative_to(docs_root)}:{session}"


# Get all example sessions
@pytest.mark.parametrize(
    "session_key", group_examples_by_session(), ids=lambda v: name_example(v[0], v[1])
)
def test_documentation_examples(
    session_key: tuple[str, str],
    eval_example: pytest_examples.EvalExample,  # type: ignore[name-defined]
) -> None:
    """
    Test that all exec="true" code examples in documentation execute successfully.

    This test groups examples by session (file + session name) and runs them
    sequentially in the same execution context, allowing code to build on
    previous examples.

    This test uses pytest-examples to:
    - Find all code examples with exec="true" in markdown files
    - Group them by session
    - Execute them in order within the same context
    - Verify no exceptions are raised
    """
    file_path, session_name = session_key

    # Get examples for this session
    all_examples = list(pytest_examples.find_examples(docs_root))
    examples = []
    for example in all_examples:
        settings = example.prefix_settings()
        if settings.get("exec") != "true":
            continue
        if str(example.path) == file_path and settings.get("session", "_default") == session_name:
            examples.append(example)

    # Run all examples in this session sequentially, preserving state
    module_globals: dict[str, object] = {}
    for example in examples:
        result = eval_example.run(example, module_globals=module_globals)
        # Update globals with the results from this execution
        module_globals.update(result)
