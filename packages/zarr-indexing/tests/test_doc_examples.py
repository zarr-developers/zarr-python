"""Executable documentation contracts.

Two kinds of test live here, and nothing else:

1. **Structural.** Every snippet include in the rendered docs resolves to a
   real file and a balanced, non-empty region — discovered by scanning the
   markdown, never hand-registered — and every example file executes. The
   examples carry their own inline assertions, so executing one *is* the
   value check; expected values are stated once, in the example, beside the
   prose that narrates them.
2. **Behavioral.** Documented classes are exercised where the example cannot
   assert the behavior itself: error paths, cache lifecycle invariants, and
   contracts stated in prose about types the docs define.

Editorial choices — section order, exact wording, teaching progression — are
deliberately not pinned here; they belong to review. Structural breakage
belongs to `mkdocs build --strict`, whose configuration makes it actually
fail on the relevant classes: `check_paths: true` for unresolvable includes,
and `validation` set to warn (strict turns warnings into errors) for broken
link anchors and nav-omitted pages.
"""

from __future__ import annotations

import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import zarr_indexing
import zarr_indexing.lazy_array as lazy_array_module
from zarr_indexing import IndexTransform, LazyArray, ReadContext

DOCS = Path(__file__).parents[1] / "docs"
PACKAGE_ROOT = DOCS.parent
STANDALONE_EXAMPLES = PACKAGE_ROOT / "examples"
DOC_SNIPPETS_DIR = DOCS / "snippets"
CACHE_EXAMPLE = STANDALONE_EXAMPLES / "system_memory_chunk_cache" / "system_memory_chunk_cache.py"

# Mirrors `pymdownx.snippets: base_path` in mkdocs.yml. If that list changes,
# change this one in the same commit.
SNIPPET_BASE_PATHS = (DOCS, STANDALONE_EXAMPLES)

_INCLUDE = re.compile(r'--8<--\s+"(?P<target>[^":\n]+?)(?::(?P<region>[^"\n]+))?"')


def _markdown_includes() -> tuple[tuple[str, str, str | None], ...]:
    """Every snippet include in the rendered docs: (page, target, region)."""
    return tuple(
        (str(page.relative_to(DOCS)), match["target"], match["region"])
        for page in sorted(DOCS.rglob("*.md"))
        for match in _INCLUDE.finditer(page.read_text())
    )


INCLUDES = _markdown_includes()
# In-process executables: every snippet, plus the one standalone example that
# is importable as a module. The lazy_indexing_* examples are CLI scripts
# (they parse argv and call sys.exit), so they run as subprocesses below —
# no other test in the repository executes them.
EXECUTABLES = (*sorted(DOC_SNIPPETS_DIR.glob("*.py")), CACHE_EXAMPLE)
CLI_EXAMPLES = tuple(
    script for script in sorted(STANDALONE_EXAMPLES.glob("*/*.py")) if script not in EXECUTABLES
)

PATTERN_NAMESPACE: dict[str, Any] = runpy.run_path(str(DOC_SNIPPETS_DIR / "indexing_patterns.py"))
PATTERN_CASES: tuple[dict[str, Any], ...] = PATTERN_NAMESPACE["PATTERN_CASES"]
CACHE_NAMESPACE: dict[str, Any] = runpy.run_path(str(CACHE_EXAMPLE))

# The documented pattern matrix must keep covering every selection family.
REQUIRED_PATTERNS = {
    "basic-slice",
    "integer-axis-removal",
    "negative-stride",
    "empty-selection",
    "boolean-mask",
    "orthogonal",
    "vectorized",
    "broadcasting",
    "repeated-out-of-order",
}


# --------------------------------------------------------------------------- #
# Structural: the include graph and the executable examples
# --------------------------------------------------------------------------- #


def test_docs_reference_snippets_at_all() -> None:
    """An empty scan means the include regex rotted, not that the docs did."""
    assert len(INCLUDES) >= 10
    assert any(region for _, _, region in INCLUDES)


@pytest.mark.parametrize(
    ("page", "target", "region"),
    INCLUDES,
    ids=[
        f"{page}->{target}" + (f":{region}" if region else "") for page, target, region in INCLUDES
    ],
)
def test_markdown_include_resolves(page: str, target: str, region: str | None) -> None:
    """Each include names exactly one real file; each region is balanced and non-empty."""
    resolved = [base / target for base in SNIPPET_BASE_PATHS if (base / target).is_file()]
    assert len(resolved) == 1, f"{page} includes {target!r}: resolved to {resolved or 'nothing'}"
    if region is None:
        return
    source = resolved[0].read_text()
    starts = re.findall(rf"^\s*# --8<-- \[start:{re.escape(region)}\]$", source, re.MULTILINE)
    ends = re.findall(rf"^\s*# --8<-- \[end:{re.escape(region)}\]$", source, re.MULTILINE)
    assert len(starts) == 1, f"{target}:{region} needs exactly one start marker, has {len(starts)}"
    assert len(ends) == 1, f"{target}:{region} needs exactly one end marker, has {len(ends)}"
    body = source.split(starts[0], maxsplit=1)[1].split(ends[0], maxsplit=1)[0]
    assert body.strip(), f"{target}:{region} is empty"


def test_snippet_directories_hold_only_their_kind() -> None:
    """Rendered pages are markdown; executable snippets are Python."""
    assert all(p.suffix == ".md" for p in (DOCS / "examples").iterdir() if p.is_file())
    assert all(p.suffix == ".py" for p in DOC_SNIPPETS_DIR.iterdir() if p.is_file())


@pytest.mark.parametrize("example", EXECUTABLES, ids=lambda path: path.stem)
def test_documentation_example_executes(example: Path) -> None:
    """Examples are executable contracts; their inline asserts are the values check."""
    runpy.run_path(str(example), run_name="__main__")


@pytest.mark.parametrize("script", CLI_EXAMPLES, ids=lambda path: path.stem)
def test_cli_example_runs_as_a_subprocess(script: Path) -> None:
    """The CLI examples exit 0 when run the way their READMEs instruct."""
    if "dask" in script.stem:
        pytest.importorskip("dask.array")
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=script.parent,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


@pytest.mark.parametrize(
    "example_dir",
    sorted(p for p in STANDALONE_EXAMPLES.iterdir() if p.is_dir()),
    ids=lambda path: path.name,
)
def test_standalone_example_is_a_documented_script(example_dir: Path) -> None:
    """Each standalone example ships a README and a same-named runnable script."""
    script = example_dir / f"{example_dir.name}.py"
    assert script.is_file()
    assert (example_dir / "README.md").is_file()
    assert (DOCS / "examples" / f"{example_dir.name}.md").is_file()
    assert script.read_text().startswith("# /// script\n"), "examples stay PEP 723 runnable"


# --------------------------------------------------------------------------- #
# Behavioral: the documented pattern matrix
# --------------------------------------------------------------------------- #


def test_indexing_pattern_matrix_is_complete() -> None:
    assert {case["name"] for case in PATTERN_CASES} == REQUIRED_PATTERNS


def test_pattern_page_tabs_are_the_models() -> None:
    """Both tabs of every matrix entry are the executable model, in order.

    The JSON tab must be the model's canonical wire form; the Python tab
    must evaluate (in the executable matrix's namespace) to the model
    itself. Either tab drifting from the snippet fails here.
    """
    import json
    import textwrap

    page = (DOCS / "guide" / "patterns.md").read_text()

    json_blocks = re.findall(r"```json\n(.*?)```", page, re.DOTALL)
    assert len(json_blocks) == len(PATTERN_CASES)
    for block, case in zip(json_blocks, PATTERN_CASES, strict=True):
        assert json.loads(block) == case["transform"].to_json(), case["name"]

    python_blocks = [
        textwrap.dedent(block)
        for block in re.findall(r"```python\n(.*?)```", page, re.DOTALL)
        if "--8<--" not in block
    ]
    assert len(python_blocks) == len(PATTERN_CASES)
    for block, case in zip(python_blocks, PATTERN_CASES, strict=True):
        constructed = eval(block, dict(PATTERN_NAMESPACE))
        assert constructed == case["transform"], case["name"]


@pytest.mark.parametrize("case", PATTERN_CASES, ids=lambda case: case["name"])
def test_indexing_pattern_matrix_matches_numpy(case: dict[str, Any]) -> None:
    """The wrapper agrees with the matrix the snippet proves at the transform level."""
    image = PATTERN_NAMESPACE["image"]
    lazy = LazyArray.from_numpy(image)
    accessor = {
        "basic": lazy.lazy,
        "oindex": lazy.lazy.oindex,
        "vindex": lazy.lazy.vindex,
    }[case["mode"]]
    view = accessor[case["selection"]]
    result = view.result()

    np.testing.assert_array_equal(result, case["expected"])
    assert result.shape == case["shape"]
    assert ("box" if view.is_box else "query") == case["category"]


# --------------------------------------------------------------------------- #
# Behavioral: contracts the guide states about wrapped sources and parts
# --------------------------------------------------------------------------- #


def test_default_source_contract_converts_basic_selected_slabs_to_system_memory() -> None:
    """A source may return Python slabs as long as NumPy can convert each selected slab."""

    class ListSlabSource:
        def __init__(self) -> None:
            self.data = np.arange(20).reshape(4, 5)
            self.keys: list[tuple[Any, ...]] = []

        @property
        def shape(self) -> tuple[int, ...]:
            return self.data.shape

        @property
        def dtype(self) -> np.dtype[Any]:
            return self.data.dtype

        def __getitem__(self, key: tuple[Any, ...]) -> object:
            assert all(isinstance(item, (int, slice)) for item in key)
            self.keys.append(key)
            return self.data[key].tolist()

    source = ListSlabSource()
    result = LazyArray(source).with_parts((2, 3)).lazy.oindex[[3, 1, 1], 1:5:2].result()

    np.testing.assert_array_equal(result, np.array([[16, 18], [6, 8], [6, 8]]))
    assert len(source.keys) > 0
    assert all(all(isinstance(item, (int, slice)) for item in key) for key in source.keys)


def test_coordinate_array_example_preserves_order_and_duplicates() -> None:
    """Coordinate arrays are ordered sequences, not mathematical sets."""
    view = LazyArray.from_numpy(np.arange(6)).with_parts((2,)).lazy.oindex[[4, 1, 1, 3]]

    np.testing.assert_array_equal(view.result(), np.array([4, 1, 1, 3]))
    assembled = np.empty(view.shape, dtype=view.dtype)
    for part in view.parts():
        assembled[part.out_selection] = part.view.result()
    np.testing.assert_array_equal(assembled, np.array([4, 1, 1, 3]))


def test_documented_partition_transform_is_global_and_projection_is_chunk_local() -> None:
    source = np.arange(8)
    part = tuple(LazyArray.from_numpy(source).with_parts((4,)).parts())[1]

    assert part.view.transform.apply((0,)) == (4,)
    assert part.view.array[part.view.transform.apply((0,))] == 4
    assert part.projection.chunk_transform.apply((0,)) == (0,)


@pytest.mark.parametrize(
    ("mode", "parts"),
    [
        ("per-axis", ((0,), (3,))),
        ("per-axis", ((), (3,))),
        ("uniform", (1, 1)),
        ("per-axis", ((0, 0), (3,))),
    ],
    ids=["single-zero", "empty-sequence", "positive-uniform", "repeated-zero"],
)
def test_documented_zero_length_axis_partition_spellings_execute(mode: str, parts: Any) -> None:
    data = np.zeros((0, 3))
    base = LazyArray.from_numpy(data)
    view = base.with_parts(parts) if mode == "uniform" else base.with_parts_per_axis(parts)

    assert view.result().shape == (0, 3)
    assert tuple(view.parts()) == ()


def test_projection_example_exposes_paired_directions() -> None:
    """Both transforms of every projection keep their documented output ranks."""
    namespace = runpy.run_path(str(DOC_SNIPPETS_DIR / "chunk_projection.py"))
    np.testing.assert_array_equal(namespace["ADVANCED_RESULT"], namespace["ADVANCED_EXPECTED"])
    assert all(p.chunk_transform.output_rank == 2 for p in namespace["PROJECTIONS"])
    assert all(p.cell_transform.output_rank == 1 for p in namespace["PROJECTIONS"])


@pytest.mark.parametrize(
    "example",
    [DOC_SNIPPETS_DIR / "integrations.py", CACHE_EXAMPLE],
    ids=lambda path: path.stem,
)
def test_projection_examples_assemble_rank_zero_domains(example: Path) -> None:
    """The examples' shared assembly helpers handle a zero-rank cell domain."""
    namespace = runpy.run_path(str(example))
    domain = zarr_indexing.IndexDomain((), ())
    cell_points = namespace["_domain_points"](domain)
    destination = np.empty((), dtype=np.intp)

    values = namespace["_gather_and_scatter"](
        destination,
        np.array(7, dtype=np.intp),
        cell_points,
        cell_points,
    )

    assert cell_points.shape == (1, 0)
    assert values.shape == (1,)
    assert destination[()] == 7


# --------------------------------------------------------------------------- #
# Behavioral: the documented chunk cache (error paths the example cannot show)
# --------------------------------------------------------------------------- #


def make_documented_cache() -> tuple[Any, Any]:
    source_type = CACHE_NAMESPACE["RecordingChunkSource"]
    cache_type = CACHE_NAMESPACE["SystemMemoryChunkCache"]
    source = source_type(np.arange(48).reshape(6, 8), chunks=(3, 4))
    return source, cache_type(source, capacity=2)


def test_system_memory_cache_assembles_and_deduplicates_public_projections() -> None:
    source, cache = make_documented_cache()

    np.testing.assert_array_equal(cache.oindex[[1, 1], 2], np.array([10, 10]))
    assert tuple(source.reads) == ((0, 0),)
    assert cache.projection_uses == (("chunk_transform", "context.transform"),)


def test_chunk_cache_queues_all_parts_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, cache = make_documented_cache()
    planning_count = 0
    original_plan_chunks = lazy_array_module.plan_chunks

    def counted_plan_chunks(*args: Any, **kwargs: Any) -> Any:
        nonlocal planning_count
        planning_count += 1
        return original_plan_chunks(*args, **kwargs)

    with monkeypatch.context() as request_patch:
        request_patch.setattr(lazy_array_module, "plan_chunks", counted_plan_chunks)
        result = cache[1:5, 2]

    np.testing.assert_array_equal(result, np.array([10, 18, 26, 34]))
    assert planning_count == 1
    assert source.reads == [(0, 0), (1, 0)]

    assert [
        (event.chunk_coords, event.previous.value, event.current.value) for event in cache.events
    ] == [
        ((0, 0), "new", "queued"),
        ((1, 0), "new", "queued"),
        ((0, 0), "queued", "loading"),
        ((0, 0), "loading", "ready"),
        ((1, 0), "queued", "loading"),
        ((1, 0), "loading", "ready"),
    ]


def test_chunk_cache_reader_resolves_a_transform_from_cached_chunks() -> None:
    """The reader boundary consumes the prepared projections of real parts."""
    source_type = CACHE_NAMESPACE["RecordingChunkSource"]
    reader_type = CACHE_NAMESPACE["SystemMemoryChunkReader"]
    source = source_type(np.arange(48).reshape(6, 8), chunks=(3, 4))
    reader = reader_type(capacity=2)
    view = LazyArray(source).with_reader(reader).lazy[1:5, 2]
    parts = tuple(view.parts())
    out = np.empty(view.shape, dtype=source.dtype)
    for part in parts:
        destination = out[part.out_selection]
        assert (
            reader.read_into(
                source,
                ReadContext(part.view.transform, part.projection),
                destination,
            )
            is None
        )

    np.testing.assert_array_equal(out, np.array([10, 18, 26, 34]))
    assert source.reads == [(0, 0), (1, 0)]
    assert reader.projection_uses == [
        ("chunk_transform", "context.transform"),
        ("chunk_transform", "context.transform"),
    ]


def test_chunk_cache_reader_requires_a_prepared_projection() -> None:
    source_type = CACHE_NAMESPACE["RecordingChunkSource"]
    reader_type = CACHE_NAMESPACE["SystemMemoryChunkReader"]
    source = source_type(np.arange(48).reshape(6, 8), chunks=(3, 4))
    transform = IndexTransform.from_shape(source.shape)[1:3, 2].translate_domain_to((0,))
    out = np.empty(transform.domain.shape, dtype=source.dtype)

    with pytest.raises(ValueError, match="requires context.projection"):
        reader_type(capacity=2).read_into(source, ReadContext(transform), out)


def test_system_memory_cache_separates_basic_and_orthogonal_indexing() -> None:
    source_type = CACHE_NAMESPACE["RecordingChunkSource"]
    cache_type = CACHE_NAMESPACE["SystemMemoryChunkCache"]
    data = np.arange(48).reshape(6, 8)
    cache = cache_type(source_type(data, chunks=(3, 4)), capacity=4)
    row = np.array([0, 2])
    column = np.array([1, 3])

    np.testing.assert_array_equal(cache[0:3, 1:4], data[0:3, 1:4])
    np.testing.assert_array_equal(
        cache.oindex[row, column],
        data[np.ix_(row, column)],
    )


def test_system_memory_cache_does_not_treat_array_keys_as_orthogonal() -> None:
    source, cache = make_documented_cache()
    row = np.array([0, 2])
    column = np.array([1, 3])

    with pytest.raises(IndexError, match="unsupported selection type for basic indexing"):
        cache[row, column]

    assert tuple(source.reads) == ()


def test_system_memory_cache_assembles_a_scalar_selection() -> None:
    source, cache = make_documented_cache()

    result = cache[1, 2]

    assert result.shape == ()
    assert result[()] == 10
    assert tuple(source.reads) == ((0, 0),)
    assert cache.projection_uses == (("chunk_transform", "context.transform"),)


def test_system_memory_cache_is_documentation_only() -> None:
    assert not hasattr(zarr_indexing, "SystemMemoryChunkCache")
    assert not hasattr(zarr_indexing, "ChunkState")


def test_chunk_source_failure_is_retained_as_failed_with_its_cause() -> None:
    source, cache = make_documented_cache()
    source.failures.add((1, 1))

    with pytest.raises(CACHE_NAMESPACE["ChunkLoadError"]) as error:
        cache[3:5, 4:6]

    assert isinstance(error.value.__cause__, OSError)
    assert cache.state((1, 1)).value == "failed"
    assert tuple(source.reads) == ((1, 1),)


def test_failed_chunk_is_not_retried_implicitly() -> None:
    source, cache = make_documented_cache()
    source.failures.add((1, 1))
    with pytest.raises(CACHE_NAMESPACE["ChunkLoadError"]):
        cache[3:5, 4:6]
    with pytest.raises(CACHE_NAMESPACE["ChunkLoadError"]):
        cache[3:5, 4:6]

    assert tuple(source.reads) == ((1, 1),)


def test_retry_requires_a_failed_chunk() -> None:
    _, cache = make_documented_cache()
    with pytest.raises(ValueError, match="retry requires failed chunk .*new"):
        cache.retry((0, 0))


def test_illegal_chunk_transition_is_rejected() -> None:
    _, cache = make_documented_cache()
    with pytest.raises(ValueError, match="illegal chunk transition new -> ready"):
        cache.reader._transition((0, 0), CACHE_NAMESPACE["ChunkState"].READY, "test")
