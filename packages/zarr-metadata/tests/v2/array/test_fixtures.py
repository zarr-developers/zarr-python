"""Decode v2 array metadata fixtures via pydantic.

Each `*.json` file in this directory is a representative on-disk
`.zarray` that should validate cleanly as `ArrayMetadataV2`.
Fixtures cover simple/structured dtypes, presence/absence of compressor
and filters, and both row/column orderings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v2.array import ArrayMetadataV2

FIXTURES_DIR = Path(__file__).parent
FIXTURES = sorted(FIXTURES_DIR.glob("*.json"))
ADAPTER = TypeAdapter(ArrayMetadataV2)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_validate(fixture: Path) -> None:
    ADAPTER.validate_python(json.loads(fixture.read_text()))
