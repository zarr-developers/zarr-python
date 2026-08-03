from __future__ import annotations

import ast
import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import zarr_indexing
from zarr_indexing import LazyArray

DOCS = Path(__file__).parents[1] / "docs"
PACKAGE_ROOT = DOCS.parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
STANDALONE_EXAMPLES = PACKAGE_ROOT / "examples"
CACHE_EXAMPLE_NAME = "napari_chunk_cache"
CACHE_EXAMPLE_DIR = STANDALONE_EXAMPLES / CACHE_EXAMPLE_NAME
CACHE_EXAMPLE = CACHE_EXAMPLE_DIR / f"{CACHE_EXAMPLE_NAME}.py"
STANDALONE_EXAMPLE_NAMES = (
    "lazy_indexing_numpy",
    "lazy_indexing_dask",
    CACHE_EXAMPLE_NAME,
)
DOC_EXAMPLES = tuple(
    DOCS / "examples" / name
    for name in (
        "coordinate_origins.py",
        "canonical_slice.py",
        "lazy_composition.py",
        "axis_manipulation.py",
        "chunk_projection.py",
        "indexing_patterns.py",
        "integrations.py",
    )
)
EXAMPLES = (*DOC_EXAMPLES, CACHE_EXAMPLE)
REGIONS = {
    "coordinate-origin": DOCS / "examples" / "coordinate_origins.py",
    "prepend-grid": DOCS / "examples" / "coordinate_origins.py",
    "canonical-slice": DOCS / "examples" / "canonical_slice.py",
    "landing-quickstart": DOCS / "examples" / "canonical_slice.py",
    "lazy-composition": DOCS / "examples" / "lazy_composition.py",
    "axis-shape-comparison": DOCS / "examples" / "axis_manipulation.py",
    "axis-insertion": DOCS / "examples" / "axis_manipulation.py",
    "chunk-projection": DOCS / "examples" / "chunk_projection.py",
    "advanced-projection": DOCS / "examples" / "chunk_projection.py",
    "indexing-patterns": DOCS / "examples" / "indexing_patterns.py",
    "zarr-consumer": DOCS / "examples" / "integrations.py",
    "viewport-consumer": DOCS / "examples" / "integrations.py",
    "chunk-cache-types": CACHE_EXAMPLE,
    "chunk-cache-source": CACHE_EXAMPLE,
    "chunk-cache-wrapper": CACHE_EXAMPLE,
    "chunk-cache-worked-example": CACHE_EXAMPLE,
}
TUTORIAL_SECTIONS = (
    ("An index selects coordinates", "an-index-selects-coordinates"),
    ("Coordinates are addresses", "coordinates-are-addresses"),
    ("Lazy views compose", "lazy-views-compose"),
    ("An index defines a result array", "an-index-defines-a-result-array"),
    ("A request becomes a chunk plan", "a-request-becomes-a-chunk-plan"),
    ("One cell domain, two projections", "one-cell-domain-two-projections"),
)
RETIRED_TUTORIAL_FILES = (
    "01-indexing.md",
    "02-transforms.md",
    "03-composition.md",
    "04-chunks.md",
    "05-projections.md",
)
TUTORIAL_SNIPPETS = (
    "examples/canonical_slice.py:canonical-slice",
    "examples/coordinate_origins.py:coordinate-origin",
    "examples/coordinate_origins.py:prepend-grid",
    "examples/lazy_composition.py:lazy-composition",
    "examples/axis_manipulation.py:axis-shape-comparison",
    "examples/axis_manipulation.py:axis-insertion",
    "examples/chunk_projection.py:chunk-projection",
    "examples/chunk_projection.py:advanced-projection",
)
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
PATTERN_NAMESPACE: dict[str, Any] = runpy.run_path(str(DOCS / "examples" / "indexing_patterns.py"))
PATTERN_CASES: tuple[dict[str, Any], ...] = PATTERN_NAMESPACE["PATTERN_CASES"]
CACHE_NAMESPACE: dict[str, Any] = runpy.run_path(str(CACHE_EXAMPLE))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_documentation_example_executes(example: Path) -> None:
    """Examples remain executable contracts rather than display-only fragments."""
    assert example.is_file(), f"missing executable documentation example: {example}"
    runpy.run_path(str(example), run_name="__main__")


def test_chunk_cache_example_has_one_package_scoped_canonical_source() -> None:
    """The chunk-cache example has one executable package-scoped source."""
    readme = CACHE_EXAMPLE_DIR / "README.md"
    page = DOCS / "examples" / f"{CACHE_EXAMPLE_NAME}.md"
    old_source = DOCS / "examples" / f"{CACHE_EXAMPLE_NAME}.py"

    assert readme.is_file()
    assert CACHE_EXAMPLE.is_file()
    assert page.is_file()
    assert not old_source.exists()

    script_source = CACHE_EXAMPLE.read_text()
    assert script_source.startswith("# /// script\n")
    assert '# requires-python = ">=3.12"' in script_source
    assert '"zarr-indexing>=0.1"' in script_source
    assert '"numpy==2.4.3"' in script_source

    page_source = page.read_text()
    assert '--8<-- "napari_chunk_cache/README.md"' in page_source
    assert '--8<-- "napari_chunk_cache/napari_chunk_cache.py"' in page_source
    assert '```python\n--8<-- "napari_chunk_cache/napari_chunk_cache.py"\n```' in page_source

    package_nav = (PACKAGE_ROOT / "mkdocs.yml").read_text()
    repository_nav = (REPOSITORY_ROOT / "mkdocs.yml").read_text()
    assert "System-memory chunk cache: examples/napari_chunk_cache.md" in package_nav
    assert "examples/napari_chunk_cache.md" not in repository_nav
    assert not (REPOSITORY_ROOT / "examples" / CACHE_EXAMPLE_NAME).exists()

    guide = (DOCS / "guide" / "integrations.md").read_text()
    for region in (
        "chunk-cache-types",
        "chunk-cache-source",
        "chunk-cache-wrapper",
        "chunk-cache-worked-example",
    ):
        canonical_snippet = f"napari_chunk_cache/napari_chunk_cache.py:{region}"
        assert canonical_snippet in guide
        assert f"examples/napari_chunk_cache.py:{region}" not in guide
    assert "../examples/napari_chunk_cache.md" in guide


@pytest.mark.parametrize(("region", "path"), REGIONS.items())
def test_documentation_region_is_nonempty_and_balanced(region: str, path: Path) -> None:
    """Each displayed fragment is a complete named region in its runnable example."""
    assert path.is_file(), f"missing example for snippet region {region!r}: {path}"
    source = path.read_text()
    start = re.findall(rf"^# --8<-- \[start:{re.escape(region)}\]$", source, re.MULTILINE)
    end = re.findall(rf"^# --8<-- \[end:{re.escape(region)}\]$", source, re.MULTILINE)
    assert len(start) == 1, f"region {region!r} needs exactly one start marker"
    assert len(end) == 1, f"region {region!r} needs exactly one end marker"

    match = re.search(
        rf"^# --8<-- \[start:{re.escape(region)}\]$\n(?P<body>.*?)^# --8<-- \[end:{re.escape(region)}\]$",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"region {region!r} is not balanced"
    assert match.group("body").strip(), f"region {region!r} must not be empty"


def test_landing_quickstart_produces_the_shown_result() -> None:
    """The landing snippet imports, selects, and materializes its promised values."""
    namespace: dict[str, Any] = runpy.run_path(
        str(DOCS / "examples" / "canonical_slice.py"), run_name="__main__"
    )
    np.testing.assert_array_equal(namespace["LANDING_QUICKSTART_RESULT"], np.array([12, 13, 14]))


def test_canonical_slice_introduces_one_dimensional_lazy_indexing_before_partitioning() -> None:
    source = (DOCS / "examples" / "canonical_slice.py").read_text()
    region = source.split("# --8<-- [start:canonical-slice]", maxsplit=1)[1].split(
        "# --8<-- [end:canonical-slice]", maxsplit=1
    )[0]

    assert "source = np.array([10, 11, 12, 13, 14, 15])" in region
    assert "lazy = LazyArray(source)" in region
    assert "view = lazy.lazy[2:5]" in region
    assert ".with_parts(" not in region
    assert "[1," not in region


def test_lazy_composition_stays_one_dimensional_until_the_later_axis_section() -> None:
    source = (DOCS / "examples" / "lazy_composition.py").read_text()
    region = source.split("# --8<-- [start:lazy-composition]", maxsplit=1)[1].split(
        "# --8<-- [end:lazy-composition]", maxsplit=1
    )[0]

    assert "source[2:5]" in region
    assert "[::-1]" in region
    assert ".lazy[1:]" in region
    assert "[1," not in region


def test_pre_result_array_snippets_do_not_change_rank_with_integer_indexes() -> None:
    """The coordinate introduction uses slices until result-axis behavior is introduced."""
    guide = (DOCS / "guide" / "index.md").read_text()
    introduction = guide.split("## An index defines a result array", maxsplit=1)[0]
    snippets = re.findall(r'--8<-- "(?P<path>[^"\n]+\.py):(?P<region>[^"]+)"', introduction)

    assert snippets
    for path, region in snippets:
        source = (DOCS / path).read_text()
        body = source.split(f"# --8<-- [start:{region}]", maxsplit=1)[1].split(
            f"# --8<-- [end:{region}]", maxsplit=1
        )[0]
        rank_changing_indexes = [
            node
            for node in ast.walk(ast.parse(body))
            if isinstance(node, ast.Subscript)
            and any(
                (isinstance(index, ast.Constant) and isinstance(index.value, int))
                or (
                    isinstance(index, ast.UnaryOp)
                    and isinstance(index.op, (ast.UAdd, ast.USub))
                    and isinstance(index.operand, ast.Constant)
                    and isinstance(index.operand.value, int)
                )
                for index in (
                    node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
                )
            )
        ]

        assert not rank_changing_indexes, f"{path}:{region} changes rank with integer indexing"


def test_axis_manipulation_examples_define_result_axes() -> None:
    """Integer, slice, and None indexes produce their documented result shapes."""
    namespace = runpy.run_path(str(DOCS / "examples" / "axis_manipulation.py"))

    np.testing.assert_array_equal(namespace["INTEGER_RESULT"], np.array([4, 5, 6, 7]))
    np.testing.assert_array_equal(namespace["SLICE_RESULT"], np.array([[4, 5, 6, 7]]))
    np.testing.assert_array_equal(namespace["INSERTED_RESULT"], np.array([[4], [5], [6], [7]]))
    assert namespace["INTEGER_RESULT"].shape == (4,)
    assert namespace["SLICE_RESULT"].shape == (1, 4)
    assert namespace["INSERTED_RESULT"].shape == (4, 1)


def test_zarr_consumer_reads_public_projections_and_assembles_exact_values() -> None:
    """A chunk source reads only projected keys and assembles through both transforms."""
    namespace: dict[str, Any] = runpy.run_path(
        str(DOCS / "examples" / "integrations.py"), run_name="__main__"
    )
    assert namespace["ZARR_SOURCE_KEYS"] == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert namespace["ZARR_SOURCE_READS"] == ((0, 0), (0, 1))
    assert all(chunk == cell for chunk, cell in namespace["ZARR_SHARED_DOMAINS"])
    assert namespace["ZARR_CHUNK_LOCAL_COORDS"] == (
        ((1, 0), (1, 1)),
        ((1, 0), (1, 1)),
    )
    assert namespace["ZARR_REQUEST_COORDS"] == (((0,), (1,)), ((2,), (3,)))
    assert namespace["ZARR_READ_VALUES"] == ((4, 5), (6, 7))
    np.testing.assert_array_equal(namespace["ZARR_RESULT"], np.array([4, 5, 6, 7]))


def test_viewport_consumer_uses_exact_non_crossing_source_keys() -> None:
    """Exact selectors prevent a read crossing into a hidden neighboring chunk."""
    namespace: dict[str, Any] = runpy.run_path(
        str(DOCS / "examples" / "integrations.py"), run_name="__main__"
    )
    assert namespace["VIEWPORT_READS_BEFORE_RESULT"] == ()
    assert namespace["VIEWPORT_SOURCE_KEYS"] == (
        (slice(1, 2, 1), slice(0, 2, 1)),
        (slice(1, 2, 1), slice(2, 4, 1)),
    )
    assert namespace["VIEWPORT_SOURCE_CHUNKS"] == ((0, 0), (0, 1))


def test_indexing_pattern_matrix_is_complete() -> None:
    namespace = runpy.run_path(str(DOCS / "examples" / "indexing_patterns.py"))
    cases = namespace["PATTERN_CASES"]
    assert {case["name"] for case in cases} == REQUIRED_PATTERNS


@pytest.mark.parametrize("case", PATTERN_CASES, ids=lambda case: case["name"])
def test_indexing_pattern_matrix_matches_numpy(case: dict[str, Any]) -> None:
    image = PATTERN_NAMESPACE["image"]
    lazy = LazyArray(image)
    accessor = {
        "basic": lazy.lazy,
        "orthogonal": lazy.lazy.oindex,
        "vectorized": lazy.lazy.vindex,
    }[case["mode"]]
    view = accessor[case["selection"]]
    result = view.result()

    np.testing.assert_array_equal(result, case["expected"])
    assert result.shape == case["shape"]
    assert ("box" if view.is_box else "query") == case["category"]


def test_projection_example_exposes_paired_directions() -> None:
    namespace = runpy.run_path(str(DOCS / "examples" / "chunk_projection.py"))
    np.testing.assert_array_equal(namespace["ADVANCED_RESULT"], namespace["ADVANCED_EXPECTED"])
    assert all(chunk == cell for chunk, cell in namespace["PAIRED_DOMAINS"])
    assert all(p.chunk_transform.output_rank == 2 for p in namespace["PROJECTIONS"])
    assert all(p.cell_transform.output_rank == 1 for p in namespace["PROJECTIONS"])


def test_prepend_projection_keeps_shared_chunk_local_and_request_spaces_distinct() -> None:
    """The shared cell coordinates project to local storage and literal request addresses."""
    namespace = runpy.run_path(str(DOCS / "examples" / "coordinate_origins.py"))

    assert namespace["PREPEND_SHARED_CELL_COORDS"] == ((0,), (1,), (2,))
    assert namespace["PREPEND_CHUNK_LOCAL_COORDS"] == ((0,), (1,), (2,))
    assert namespace["PREPEND_REQUEST_COORDS"] == ((-3,), (-2,), (-1,))


def test_half_open_interval_examples_explain_ordered_concatenation_before_negative_coordinates() -> (
    None
):
    chapter = (DOCS / "guide" / "index.md").read_text()
    definition = "start at 0, inclusive, and stop at 1, exclusive"
    examples = (
        ("[0, 1) = {0}", "Start at 0 and stop before 1, so the interval contains only 0."),
        (
            "[1, 3) = {1, 2}",
            "Start at 1 and stop before 3, so the interval contains 1 and 2.",
        ),
        (
            "concat([0, 1), [1, 3)) = [0, 3)",
            "Append the second interval after the first. Their matching "
            "exclusive/inclusive boundary produces one continuous interval without "
            "a gap or duplicated coordinate.",
        ),
    )
    interval_section = chapter[: chapter.index("[-2, 3)")]
    normalized_section = " ".join(interval_section.split())

    assert definition in chapter
    assert all(
        equation in normalized_section and explanation in normalized_section
        for equation, explanation in examples
    )
    assert " ∪ " not in interval_section  # noqa: RUF001
    assert chapter.index(definition) < chapter.index("[-2, 3)")
    assert "half-open-intervals.svg" not in chapter


def test_visual_guide_is_one_scrolling_document_with_stable_section_links() -> None:
    guide = (DOCS / "guide" / "index.md").read_text()
    headings = tuple(f"## {title} {{#{anchor}}}" for title, anchor in TUTORIAL_SECTIONS)

    assert all(guide.count(heading) == 1 for heading in headings)
    positions = tuple(guide.index(heading) for heading in headings)
    assert positions == tuple(sorted(positions))
    assert all(f"](#{anchor})" in guide for _, anchor in TUTORIAL_SECTIONS)
    assert all(guide.count(snippet) == 1 for snippet in TUTORIAL_SNIPPETS)
    assert all(not (DOCS / "guide" / name).exists() for name in RETIRED_TUTORIAL_FILES)


@pytest.mark.parametrize("name", STANDALONE_EXAMPLE_NAMES)
def test_standalone_example_source_uses_a_plain_python_fence(name: str) -> None:
    source = (DOCS / "examples" / f"{name}.md").read_text()
    expected_include = f'```python\n--8<-- "{name}/{name}.py"\n```'

    assert expected_include in source
    assert "```python " not in source


@pytest.mark.parametrize(
    ("name", "actual", "expected"),
    [
        ("no eager reads", CACHE_NAMESPACE["READS_BEFORE_SELECTION"], ()),
        ("initial values", CACHE_NAMESPACE["INITIAL_RESULT"].tolist(), [10, 18, 26, 34]),
        ("initial reads", CACHE_NAMESPACE["INITIAL_READS"], ((0, 0), (1, 0))),
        ("overlap values", CACHE_NAMESPACE["OVERLAP_RESULT"].tolist(), [26, 34]),
        ("overlap reads", CACHE_NAMESPACE["OVERLAP_NEW_READS"], ()),
        ("eviction values", CACHE_NAMESPACE["EVICTION_RESULT"].tolist(), [5, 13]),
        ("eviction read", CACHE_NAMESPACE["EVICTION_NEW_READS"], ((0, 1),)),
        ("after eviction", CACHE_NAMESPACE["AFTER_EVICTION_RESIDENT"], ((0, 1), (1, 0))),
        ("reload values", CACHE_NAMESPACE["RELOAD_RESULT"].tolist(), [10, 18, 26, 34]),
        ("reload", CACHE_NAMESPACE["RELOAD_NEW_READS"], ((0, 0),)),
        ("after reload", CACHE_NAMESPACE["AFTER_RELOAD_RESIDENT"], ((0, 0), (1, 0))),
        ("failed reads", CACHE_NAMESPACE["FAILURE_READ_COUNTS"], (1, 1)),
        ("retry read", CACHE_NAMESPACE["RETRY_NEW_READS"], ((1, 1),)),
        ("retry values", CACHE_NAMESPACE["RETRY_RESULT"].tolist(), [[28, 29], [36, 37]]),
        ("retry state", CACHE_NAMESPACE["RETRY_STATE"], "ready"),
        (
            "failure lifecycle",
            CACHE_NAMESPACE["FAILED_TRANSITIONS"],
            ("queued", "loading", "failed", "queued", "loading", "ready"),
        ),
        (
            "failure event log",
            CACHE_NAMESPACE["FAILED_EVENT_ROWS"],
            (
                ("new", "queued", "requested"),
                ("queued", "loading", "queue drained"),
                ("loading", "failed", "source read failed"),
                ("failed", "queued", "explicit retry"),
                ("queued", "loading", "queue drained"),
                ("loading", "ready", "source read completed"),
            ),
        ),
    ],
)
def test_system_memory_cache_worked_sequence(name: str, actual: object, expected: object) -> None:
    assert actual == expected, name


def test_system_memory_cache_assembles_and_deduplicates_public_projections() -> None:
    cache_type = CACHE_NAMESPACE["SystemMemoryChunkCache"]
    source_type = CACHE_NAMESPACE["RecordingChunkSource"]
    source = source_type(np.arange(48).reshape(6, 8), chunks=(3, 4))
    cache = cache_type(source, capacity=2)

    np.testing.assert_array_equal(cache.oindex[[1, 1], 2], np.array([10, 10]))
    assert tuple(source.reads) == ((0, 0),)
    assert cache.projection_uses == (("chunk_transform", "cell_transform"),)


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
    cache_type = CACHE_NAMESPACE["SystemMemoryChunkCache"]
    source_type = CACHE_NAMESPACE["RecordingChunkSource"]
    source = source_type(np.arange(48).reshape(6, 8), chunks=(3, 4))
    cache = cache_type(source, capacity=2)

    result = cache[1, 2]

    assert result.shape == ()
    assert result[()] == 10
    assert tuple(source.reads) == ((0, 0),)
    assert cache.projection_uses == (("chunk_transform", "cell_transform"),)


def test_system_memory_cache_is_documentation_only() -> None:
    assert not hasattr(zarr_indexing, "SystemMemoryChunkCache")
    assert not hasattr(zarr_indexing, "ChunkState")


@pytest.mark.parametrize("name", STANDALONE_EXAMPLE_NAMES)
def test_lazy_indexing_examples_are_scoped_to_the_indexing_package(name: str) -> None:
    readme = STANDALONE_EXAMPLES / name / "README.md"
    assert readme.is_file()
    assert (STANDALONE_EXAMPLES / name / f"{name}.py").is_file()
    assert (DOCS / "examples" / f"{name}.md").is_file()
    assert f"uv run --with-editable . examples/{name}/{name}.py" in readme.read_text()

    assert not (REPOSITORY_ROOT / "examples" / name).exists()
    assert not (REPOSITORY_ROOT / "docs" / "user-guide" / "examples" / f"{name}.md").exists()

    package_nav = (PACKAGE_ROOT / "mkdocs.yml").read_text()
    repository_nav = (REPOSITORY_ROOT / "mkdocs.yml").read_text()
    assert f"examples/{name}.md" in package_nav
    assert f"user-guide/examples/{name}.md" not in repository_nav


@pytest.mark.parametrize(
    "example",
    [
        DOCS / "examples" / "integrations.py",
        DOCS / "examples" / "coordinate_origins.py",
        CACHE_EXAMPLE,
    ],
    ids=lambda path: path.stem,
)
def test_projection_examples_use_batched_transform_application(example: Path) -> None:
    source = example.read_text()

    assert "def evaluate_point" not in source
    assert ".chunk_transform.apply_many(" in source
    assert ".cell_transform.apply_many(" in source


@pytest.mark.parametrize(
    "example",
    [
        DOCS / "examples" / "integrations.py",
        CACHE_EXAMPLE,
    ],
    ids=lambda path: path.stem,
)
def test_projection_examples_assemble_rank_zero_domains(example: Path) -> None:
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


def test_integration_guide_states_the_cache_boundary_and_prior_art() -> None:
    guide = (DOCS / "guide" / "integrations.md").read_text()
    assert "system-memory chunk cache" in guide.lower()
    assert "not a napari integration" in guide.lower()
    assert "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-types" in guide
    assert "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-source" in guide
    assert "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-wrapper" in guide
    assert "napari_chunk_cache/napari_chunk_cache.py:chunk-cache-worked-example" in guide
    assert "../examples/napari_chunk_cache.md" in guide
    assert "https://napari.org/dev/howtos/layers/image.html" in guide
    assert "https://github.com/google/neuroglancer/blob/master/src/chunk_manager/base.ts" in guide
    assert "conceptual prior art" in guide.lower()


def make_documented_cache() -> tuple[Any, Any]:
    source_type = CACHE_NAMESPACE["RecordingChunkSource"]
    cache_type = CACHE_NAMESPACE["SystemMemoryChunkCache"]
    source = source_type(np.arange(48).reshape(6, 8), chunks=(3, 4))
    return source, cache_type(source, capacity=2)


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
        cache._transition((0, 0), CACHE_NAMESPACE["ChunkState"].READY, "test")
