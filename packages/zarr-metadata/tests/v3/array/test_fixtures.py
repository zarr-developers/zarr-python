"""Decode v3 array metadata fixtures via pydantic.

Each `*.json` file in this directory is a representative on-disk
`zarr.json` that should validate cleanly as `ArrayMetadataV3`.
Fixtures are named for the variant they exercise (regular vs rectilinear
grid, blosc/gzip/zstd/sharding_indexed codecs, named-config dtypes, optional
fields, extra fields).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.array import ArrayMetadataV3

FIXTURES_DIR = Path(__file__).parent
FIXTURES = sorted(FIXTURES_DIR.glob("*.json"))
ADAPTER = TypeAdapter(ArrayMetadataV3)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_validate(fixture: Path) -> None:
    ADAPTER.validate_python(json.loads(fixture.read_text()))
