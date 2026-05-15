"""Validate rectilinear chunk grid fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.chunk_grid.rectilinear import RectilinearChunkGridMetadata

CASES: dict[str, object] = json.loads((Path(__file__).parent / "cases.json").read_text())


@pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
def test_chunk_grid(case: object) -> None:
    TypeAdapter(RectilinearChunkGridMetadata).validate_python(case)
