from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


def test_viewport_consumer_reads_only_touched_source_chunks() -> None:
    """Changing the viewport must not make the executable example over-read its source."""
    namespace: dict[str, Any] = runpy.run_path(
        str(DOCS / "examples" / "integrations.py"), run_name="__main__"
    )
    assert namespace["VIEWPORT_SOURCE_CHUNKS"] == ((0, 0), (1, 0))


def test_projection_example_exposes_paired_directions() -> None:
    namespace = runpy.run_path(str(DOCS / "examples" / "chunk_projection.py"))
    np.testing.assert_array_equal(namespace["ADVANCED_RESULT"], namespace["ADVANCED_EXPECTED"])
    assert all(chunk == cell for chunk, cell in namespace["PAIRED_DOMAINS"])
    assert all(p.chunk_transform.output_rank == 2 for p in namespace["PROJECTIONS"])
    assert all(p.cell_transform.output_rank == 1 for p in namespace["PROJECTIONS"])
