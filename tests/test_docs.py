"""
Tests for executable code blocks in markdown documentation.

This module uses pytest-examples to validate that all Python code examples
with exec="true" in the documentation execute successfully.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pytest_examples")
from pytest_examples import CodeExample, EvalExample, find_examples

if TYPE_CHECKING:
    from collections.abc import Generator

# Find all markdown files with executable code blocks
DOCS_ROOT = Path(__file__).parent.parent / "docs"
SOURCES_ROOT = Path(__file__).parent.parent / "src" / "zarr"


def find_markdown_files_with_exec() -> list[Path]:
    """Find all markdown files containing exec="true" code blocks."""
    markdown_files = []

    for md_file in DOCS_ROOT.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            if 'exec="true"' in content:
                markdown_files.append(md_file)
        except Exception:
            # Skip files that can't be read
            continue

    return sorted(markdown_files)


def name_example(path: str, session: str) -> str:
    """Generate a readable name for a test case from file path and session."""
    file = Path(path)
    try:
        file = file.relative_to(DOCS_ROOT)
    except ValueError:
        # Path is outside DOCS_ROOT (e.g. a tmp_path fixture in unit tests); use the
        # bare file name rather than an absolute path for a stable, readable id.
        file = Path(file.name)
    return f"{file}:{session}"


def _markers_for(settings: dict[str, str]) -> list[pytest.MarkDecorator]:
    """Translate a block's markers="a b" attribute into pytest mark decorators."""
    raw = settings.get("markers", "")
    return [getattr(pytest.mark, name) for name in raw.split() if name]


def _session_params(root: Path) -> list[Any]:
    """Group exec="true" examples by (file, session) and emit one pytest.param per
    session, carrying the union of markers declared by that session's blocks."""
    sessions: defaultdict[tuple[str, str], list[CodeExample]] = defaultdict(list)
    marks_by_session: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for example in find_examples(str(root)):
        settings = example.prefix_settings()
        if settings.get("exec") != "true":
            continue
        session_name = settings.get("session", "_default")
        key = (str(example.path), session_name)
        sessions[key].append(example)
        for mark in _markers_for(settings):
            marks_by_session[key].add(mark.name)

    params = []
    for key in sorted(sessions.keys(), key=lambda x: (x[0], x[1])):
        marks = tuple(getattr(pytest.mark, name) for name in sorted(marks_by_session[key]))
        params.append(pytest.param(key, marks=marks, id=name_example(key[0], key[1])))
    return params


S3_PORT = 5556
S3_ENDPOINT = f"http://127.0.0.1:{S3_PORT}/"
S3_BUCKET = "example-bucket"


@pytest.fixture
def docs_s3_backend() -> Generator[None, None, None]:
    """Stand up a moto mock-S3 server and set a process-wide default endpoint so docs
    blocks can use a bare s3:// URL with no storage_options (see spike in plan Task 1)."""
    moto_server = pytest.importorskip("moto.moto_server.threaded_moto_server")
    s3fs = pytest.importorskip("s3fs")
    botocore = pytest.importorskip("botocore")
    requests = pytest.importorskip("requests")

    prev_endpoint = os.environ.get("AWS_ENDPOINT_URL")
    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=S3_PORT)
    server.start()
    try:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foo")
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "foo")
        os.environ["AWS_ENDPOINT_URL"] = S3_ENDPOINT

        session = botocore.session.Session()
        client = session.create_client("s3", endpoint_url=S3_ENDPOINT, region_name="us-east-1")
        client.create_bucket(Bucket=S3_BUCKET)
        client.close()
        s3fs.S3FileSystem.clear_instance_cache()
        yield
    finally:
        requests.post(f"{S3_ENDPOINT}/moto-api/reset")
        if prev_endpoint is None:
            os.environ.pop("AWS_ENDPOINT_URL", None)
        else:
            os.environ["AWS_ENDPOINT_URL"] = prev_endpoint
        server.stop()


def test_markers_attribute_is_parsed(tmp_path: Path) -> None:
    """A block tagged markers="s3" must surface that marker on its parametrized case,
    so pytest can gate/bind it (e.g. attach the moto fixture)."""
    md = tmp_path / "ex.md"
    md.write_text(
        '```python exec="true" session="demo" markers="s3"\nimport zarr\n```\n',
        encoding="utf-8",
    )
    params = _session_params(md.parent)
    assert len(params) == 1
    marks = params[0].marks
    assert any(m.name == "s3" for m in marks)


# Get all example sessions
@pytest.mark.parametrize("session_key", _session_params(DOCS_ROOT))
def test_documentation_examples(
    session_key: tuple[str, str],
    eval_example: EvalExample,
    request: pytest.FixtureRequest,
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
    if request.node.get_closest_marker("gpu") is not None:
        pytest.importorskip("cupy")

    if request.node.get_closest_marker("s3") is not None:
        request.getfixturevalue("docs_s3_backend")

    file_path, session_name = session_key

    # Get examples for this session
    all_examples = list(find_examples(DOCS_ROOT))
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
        # TODO: uncomment this line when we are ready to fix output checks
        # result = eval_example.run_print_check(example, module_globals=module_globals)
        result = eval_example.run(example, module_globals=module_globals)
        # Update globals with the results from this execution
        module_globals.update(result)


@pytest.mark.parametrize("example", find_examples(str(SOURCES_ROOT)), ids=str)
def test_docstrings(example: CodeExample, eval_example: EvalExample) -> None:
    """Test our docstring examples."""
    if example.path.name == "config.py" and "your.module" in example.source:
        pytest.skip("Skip testing docstring example that assumes nonexistent module.")
    eval_example.run_print_check(example)
