"""Decode v2 group metadata fixtures via pydantic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v2.group import GroupMetadataV2

FIXTURES_DIR = Path(__file__).parent
FIXTURES = sorted(FIXTURES_DIR.glob("*.json"))
ADAPTER = TypeAdapter(GroupMetadataV2)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_validate(fixture: Path) -> None:
    ADAPTER.validate_python(json.loads(fixture.read_text()))
