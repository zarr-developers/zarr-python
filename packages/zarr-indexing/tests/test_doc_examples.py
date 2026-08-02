from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
    )
)
REGIONS = {
    "coordinate-origin": DOCS / "examples" / "coordinate_origins.py",
    "prepend-grid": DOCS / "examples" / "coordinate_origins.py",
    "canonical-slice": DOCS / "examples" / "canonical_slice.py",
    "lazy-composition": DOCS / "examples" / "lazy_composition.py",
    "chunk-projection": DOCS / "examples" / "chunk_projection.py",
    "advanced-projection": DOCS / "examples" / "chunk_projection.py",
    "indexing-patterns": DOCS / "examples" / "indexing_patterns.py",
    "zarr-consumer": DOCS / "examples" / "integrations.py",
    "viewport-consumer": DOCS / "examples" / "integrations.py",
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


def test_integration_consumers_touch_only_planned_source_chunks() -> None:
    """Dispatch and viewport examples stay within the same two-chunk plan."""
    namespace: dict[str, Any] = runpy.run_path(
        str(DOCS / "examples" / "integrations.py"), run_name="__main__"
    )
    expected_chunks = ((0, 0), (1, 0))
    assert namespace["ZARR_DISPATCHED_CHUNKS"] == expected_chunks
    assert namespace["VIEWPORT_READS_BEFORE_RESULT"] == ()
    assert namespace["VIEWPORT_SOURCE_CHUNKS"] == expected_chunks


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
