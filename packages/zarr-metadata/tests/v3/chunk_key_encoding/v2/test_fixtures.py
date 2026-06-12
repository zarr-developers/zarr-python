"""Validate v2-compatibility chunk-key encoding fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from zarr_metadata.v3.chunk_key_encoding.v2 import V2ChunkKeyEncodingMetadata

CASES: dict[str, object] = json.loads((Path(__file__).parent / "cases.json").read_text())


@pytest.mark.parametrize("case", CASES.values(), ids=list(CASES))
def test_chunk_key_encoding(case: object) -> None:
    TypeAdapter(V2ChunkKeyEncodingMetadata).validate_python(case)
