"""Decode v3 group metadata fixtures via pydantic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.group import ZarrV3GroupMetadataJSON

FIXTURES_DIR = Path(__file__).parent
FIXTURES = sorted(FIXTURES_DIR.glob("*.json"))
ADAPTER = TypeAdapter(ZarrV3GroupMetadataJSON)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_validate(fixture: Path) -> None:
    ADAPTER.validate_python(json.loads(fixture.read_text()))
