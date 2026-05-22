"""Validate zstd codec fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.codec.zstd import ZstdCodecMetadata

CASES: dict[str, object] = json.loads((Path(__file__).parent / "cases.json").read_text())


@pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
def test_codec(case: object) -> None:
    TypeAdapter(ZstdCodecMetadata).validate_python(case)
