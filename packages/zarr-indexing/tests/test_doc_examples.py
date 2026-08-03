from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import zarr_indexing
from zarr_indexing import LazyArray

DOCS = Path(__file__).parents[1] / "docs"
EXAMPLES = tuple(
    DOCS / "examples" / name
    for name in (
        "coordinate_origins.py",
        "canonical_slice.py",
        "lazy_composition.py",
        "chunk_projection.py",
        "indexing_patterns.py",
        "integrations.py",
        "napari_chunk_cache.py",
    )
)
REGIONS = {
    "coordinate-origin": DOCS / "examples" / "coordinate_origins.py",
    "prepend-grid": DOCS / "examples" / "coordinate_origins.py",
    "canonical-slice": DOCS / "examples" / "canonical_slice.py",
    "landing-quickstart": DOCS / "examples" / "canonical_slice.py",
    "lazy-composition": DOCS / "examples" / "lazy_composition.py",
    "chunk-projection": DOCS / "examples" / "chunk_projection.py",
    "advanced-projection": DOCS / "examples" / "chunk_projection.py",
    "indexing-patterns": DOCS / "examples" / "indexing_patterns.py",
    "zarr-consumer": DOCS / "examples" / "integrations.py",
    "viewport-consumer": DOCS / "examples" / "integrations.py",
    "chunk-cache-types": DOCS / "examples" / "napari_chunk_cache.py",
    "chunk-cache-source": DOCS / "examples" / "napari_chunk_cache.py",
    "chunk-cache-wrapper": DOCS / "examples" / "napari_chunk_cache.py",
    "chunk-cache-worked-example": DOCS / "examples" / "napari_chunk_cache.py",
}
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
CACHE_NAMESPACE: dict[str, Any] = runpy.run_path(str(DOCS / "examples" / "napari_chunk_cache.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_documentation_example_executes(example: Path) -> None:
    """Examples remain executable contracts rather than display-only fragments."""
    assert example.is_file(), f"missing executable documentation example: {example}"
    runpy.run_path(str(example), run_name="__main__")


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
    np.testing.assert_array_equal(
        namespace["LANDING_QUICKSTART_RESULT"], np.array([10, 18, 26, 34])
    )


def test_zarr_consumer_reads_public_projections_and_assembles_exact_values() -> None:
    """A chunk source reads only projected keys and assembles through both transforms."""
    namespace: dict[str, Any] = runpy.run_path(
        str(DOCS / "examples" / "integrations.py"), run_name="__main__"
    )
    assert namespace["ZARR_SOURCE_KEYS"] == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert namespace["ZARR_SOURCE_READS"] == ((0, 0), (1, 0))
    assert all(chunk == cell for chunk, cell in namespace["ZARR_SHARED_DOMAINS"])
    assert namespace["ZARR_CHUNK_LOCAL_COORDS"] == (
        ((1, 2), (2, 2)),
        ((0, 2), (1, 2)),
    )
    assert namespace["ZARR_REQUEST_COORDS"] == (((0,), (1,)), ((2,), (3,)))
    assert namespace["ZARR_READ_VALUES"] == ((10, 18), (26, 34))
    np.testing.assert_array_equal(namespace["ZARR_RESULT"], np.array([10, 18, 26, 34]))


def test_viewport_consumer_uses_exact_non_crossing_source_keys() -> None:
    """Exact selectors prevent a read crossing into a hidden neighboring chunk."""
    namespace: dict[str, Any] = runpy.run_path(
        str(DOCS / "examples" / "integrations.py"), run_name="__main__"
    )
    assert namespace["VIEWPORT_READS_BEFORE_RESULT"] == ()
    assert namespace["VIEWPORT_SOURCE_KEYS"] == (
        (slice(1, 3, 1), slice(2, 3, 1)),
        (slice(3, 5, 1), slice(2, 3, 1)),
    )
    assert namespace["VIEWPORT_SOURCE_CHUNKS"] == ((0, 0), (1, 0))


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


def test_half_open_interval_is_defined_before_negative_coordinates() -> None:
    chapter = (DOCS / "guide" / "02-transforms.md").read_text()
    definition = "start at 0, inclusive, and stop at 1, exclusive"
    concatenation = "[0, 1) ∪ [1, 3) = [0, 3)"  # noqa: RUF001

    assert definition in chapter
    assert concatenation in chapter
    assert chapter.index(definition) < chapter.index("[-2, 3)")
    assert chapter.index("half-open-intervals.svg") < chapter.index("[-2, 3)")


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

    np.testing.assert_array_equal(cache[[1, 1], 2], np.array([10, 10]))
    assert tuple(source.reads) == ((0, 0),)
    assert cache.projection_uses == (("chunk_transform", "cell_transform"),)


def test_system_memory_cache_is_documentation_only() -> None:
    assert not hasattr(zarr_indexing, "SystemMemoryChunkCache")
    assert not hasattr(zarr_indexing, "ChunkState")


@pytest.mark.parametrize(
    "example",
    [
        DOCS / "examples" / "integrations.py",
        DOCS / "examples" / "coordinate_origins.py",
        DOCS / "examples" / "napari_chunk_cache.py",
    ],
    ids=lambda path: path.stem,
)
def test_projection_examples_use_batched_transform_application(example: Path) -> None:
    source = example.read_text()

    assert "def evaluate_point" not in source
    assert ".chunk_transform.apply_many(" in source
    assert ".cell_transform.apply_many(" in source


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
