"""The shipped examples must keep working.

An example that has quietly rotted is worse than no example: it is the first
thing a new user copies. These run the real files rather than a paraphrase of
them, so a signature change or a behavior change fails here rather than in
someone's notebook.
"""

from __future__ import annotations

import pathlib
import runpy

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
NOTEBOOK = EXAMPLES / "serve_notebook.ipynb"
SCRIPT = EXAMPLES / "serve.py"


@pytest.mark.parametrize("path", [NOTEBOOK, SCRIPT], ids=["notebook", "script"])
def test_example_exists(path: pathlib.Path) -> None:
    """Guards the paths above: a renamed or moved example would otherwise turn
    its execution test into a skip, or a no-op, that nobody notices."""
    assert path.is_file(), f"missing example at {path}"


def test_serve_script_runs() -> None:
    """Run examples/serve.py top to bottom.

    In-process rather than as a subprocess: the script's inline uv metadata
    resolves `zarr-http-server` from git, so `uv run` on it would test whatever
    is on main instead of the working tree.
    """
    runpy.run_path(str(SCRIPT), run_name="__main__")


def test_serve_notebook_executes() -> None:
    """Run every cell in a real kernel.

    The notebook asserts its own expectations -- status codes, byte ranges,
    that a PUT is refused, that the port is closed after `shutdown()` -- so
    this is not merely a check that nothing raised. `NotebookClient.execute`
    defaults to ``allow_errors=False``, so any failed cell raises
    `CellExecutionError` and fails this test with that cell's traceback.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel", reason="a kernel is needed to execute the notebook")

    notebook = nbformat.read(NOTEBOOK, as_version=4)

    # Run with the notebook's own directory as cwd, so any relative path it
    # uses means the same thing as when a reader opens it.
    nbclient.NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES)}},
    ).execute()
