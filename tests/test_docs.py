"""
Tests for executable code blocks in markdown documentation.

This module uses pytest-examples to validate Python code examples in the docs. A block is
validated if it renders output at build (exec="true") or is explicitly marked for testing
(test="true"); see the two-flags discussion in
docs/superpowers/specs/2026-05-29-docs-block-validation-design.md. The test_no_unvalidated_blocks
guard ensures every python block declares one of those, or an explicit exec="false" opt-out
with a reason, so a block can never silently skip validation.
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

DOCS_ROOT = Path(__file__).parent.parent / "docs"
SOURCES_ROOT = Path(__file__).parent.parent / "src" / "zarr"


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


def _is_tested(settings: dict[str, str]) -> bool:
    """A block is validated by our pytest harness if it is run at build to render output
    (exec="true") OR explicitly marked for testing (test="true"). The two flags are
    separate on purpose: exec= drives markdown-exec's build-time rendering, while test=
    lets a block be validated without being run at build (e.g. gpu/s3 blocks, which the
    build environment cannot run)."""
    return settings.get("exec") == "true" or settings.get("test") == "true"


def _session_params(root: Path) -> list[Any]:
    """Group tested examples (exec="true" or test="true") by (file, session) and emit one
    pytest.param per session, carrying the union of markers declared by that session's
    blocks."""
    sessions: defaultdict[tuple[str, str], list[CodeExample]] = defaultdict(list)
    marks_by_session: defaultdict[tuple[str, str], set[str]] = defaultdict(set)

    for example in find_examples(str(root)):
        settings = example.prefix_settings()
        if not _is_tested(settings):
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
    """A test="true" block tagged markers="s3" must surface that marker on its
    parametrized case, so pytest can gate/bind it (e.g. attach the moto fixture).
    Uses test="true" (not exec="true") because marker-bound blocks are validated by the
    harness without being run at build time."""
    md = tmp_path / "ex.md"
    md.write_text(
        '```python test="true" session="demo" markers="s3"\nimport zarr\n```\n',
        encoding="utf-8",
    )
    params = _session_params(md.parent)
    assert len(params) == 1
    marks = params[0].marks
    assert any(m.name == "s3" for m in marks)


def test_no_unvalidated_blocks() -> None:
    """Every python code block in docs/ must declare its validation state: exec="true"
    (run at build to render output), test="true" (validated by this harness without being
    run at build), or exec="false" with a reason (explicit, documented opt-out). A bare or
    mistyped fence (e.g. exec="on") fails here, so a block can never silently opt out of
    validation -- the gap that hid the invalid create_array(mode="w") example in #4016.

    Note on placement: a test="true"-only block (which markdown-exec does not execute)
    must not sit *before* an exec="true" block of the same page's session, or it disrupts
    markdown-exec's build-time execution of the later block. Keep test-only blocks last on
    the page (or on a page where they are the only python block, like gpu.md)."""
    offenders: list[str] = []
    for example in find_examples(str(DOCS_ROOT)):
        rel = Path(example.path).relative_to(DOCS_ROOT)
        # docs/superpowers/ holds design-doc caches (plans/specs), not published
        # documentation -- it is not in the mkdocs nav -- so its illustrative
        # fences are not subject to the execution guard.
        if rel.parts and rel.parts[0] == "superpowers":
            continue
        settings = example.prefix_settings()
        exec_val = settings.get("exec")
        loc = f"{rel}:{example.start_line}"
        # Validated either by build-render (exec="true") or by the test harness
        # (test="true").
        if _is_tested(settings):
            continue
        # Explicit, documented opt-out from execution.
        if exec_val == "false" and settings.get("reason", "").strip():
            continue
        offenders.append(
            f"{loc} (exec={exec_val!r}, test={settings.get('test')!r}, "
            f"reason={settings.get('reason')!r})"
        )

    assert not offenders, (
        'Docs python blocks must be exec="true", test="true", or exec="false" with a '
        "reason:\n" + "\n".join(offenders)
    )


# Get all example sessions
@pytest.mark.parametrize("session_key", _session_params(DOCS_ROOT))
def test_documentation_examples(
    session_key: tuple[str, str],
    eval_example: EvalExample,
    request: pytest.FixtureRequest,
) -> None:
    """
    Test that all validated code examples (exec="true" or test="true") in documentation
    execute successfully.

    This test groups examples by session (file + session name) and runs them
    sequentially in the same execution context, allowing code to build on
    previous examples.

    This test uses pytest-examples to:
    - Find all code examples marked exec="true" or test="true" in markdown files
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
        if not _is_tested(settings):
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
