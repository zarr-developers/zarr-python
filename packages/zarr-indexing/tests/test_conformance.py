"""ndsel conformance corpus harness.

Runs the vendored, language-agnostic ndsel fixtures (see
`tests/conformance/PROVENANCE.md`) against this package's message layer
(`zarr_indexing.messages`). An implementation is conformant iff:

- for every *success* fixture, `normalize_ndsel(input)` equals the fixture's
  `normalized` value by structural JSON equality;
- for every *error* fixture, `normalize_ndsel(input)` is rejected with an
  `NdselError` carrying the fixture's `error` reason code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zarr_indexing.messages import NdselError, normalize_ndsel

_CONFORMANCE_DIR = Path(__file__).parent / "conformance"


def _load_cases() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(_CONFORMANCE_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        cases.extend((f"{path.stem}::{case['name']}", case) for case in data)
    return cases


_CASES = _load_cases()
_SUCCESS = [(name, c) for name, c in _CASES if "normalized" in c]
_ERROR = [(name, c) for name, c in _CASES if "error" in c]


def test_corpus_is_present() -> None:
    # Guard against an empty/missing vendored corpus silently passing.
    assert len(_SUCCESS) > 0
    assert len(_ERROR) > 0


@pytest.mark.parametrize(("name", "case"), _SUCCESS, ids=[name for name, _ in _SUCCESS])
def test_success_fixture(name: str, case: dict[str, Any]) -> None:
    result = normalize_ndsel(case["input"])
    assert result == case["normalized"]


@pytest.mark.parametrize(("name", "case"), _ERROR, ids=[name for name, _ in _ERROR])
def test_error_fixture(name: str, case: dict[str, Any]) -> None:
    with pytest.raises(NdselError) as excinfo:
        normalize_ndsel(case["input"])
    assert excinfo.value.reason == case["error"]
