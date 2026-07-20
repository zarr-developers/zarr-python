"""Message-layer tests beyond the vendored conformance corpus.

The corpus (see `test_conformance.py`) covers the desugaring matrix and error
codes. These tests pin behaviors the corpus does not: `normalize` idempotence,
`parse_ndsel`, 64-bit boundary handling, and schema-valid-but-redundant maps.
"""

from __future__ import annotations

from typing import Any

import pytest

from zarr_transforms.messages import NdselError, normalize_ndsel, parse_ndsel

_MESSAGES = [
    {"kind": "point", "coords": [4, 7]},
    {"kind": "box", "inclusive_min": [0, 0], "exclusive_max": [3, 4]},
    {"kind": "box", "inclusive_min": [["-inf"], 0], "exclusive_max": [["+inf"], 4]},
    {"kind": "slice", "start": [5], "stop": [10], "step": [2]},
    {"kind": "points", "coords": [[1, 10], [2, 20]]},
    {
        "kind": "transform",
        "input_inclusive_min": [0],
        "input_exclusive_max": [3],
        "output": [{"offset": 7}, {"input_dimension": 0, "stride": 2}, {"index_array": [1, 2, 3]}],
    },
]


@pytest.mark.parametrize("message", _MESSAGES)
def test_normalize_is_idempotent(message: dict[str, Any]) -> None:
    once = normalize_ndsel(message)
    twice = normalize_ndsel({"kind": "transform", **once})
    assert twice == once


@pytest.mark.parametrize("message", _MESSAGES)
def test_parse_returns_message_unchanged(message: dict[str, Any]) -> None:
    assert parse_ndsel(message) == message


def test_parse_rejects_invalid() -> None:
    with pytest.raises(NdselError) as excinfo:
        parse_ndsel({"kind": "slice", "start": [0]})
    assert excinfo.value.reason == "invalid_json"


def test_constant_map_drops_redundant_stride() -> None:
    # A constant map (no input_dimension, no index_array) is schema-valid even
    # with a stray stride; it canonicalizes to offset-only.
    result = normalize_ndsel(
        {"kind": "transform", "input_rank": 0, "output": [{"offset": 5, "stride": 9}]}
    )
    assert result["output"] == [{"offset": 5}]


def test_i64_min_and_max_round_trip() -> None:
    i64_min, i64_max = -(2**63), 2**63 - 1
    result = normalize_ndsel({"kind": "point", "coords": [i64_min, i64_max]})
    assert result["output"] == [{"offset": i64_min}, {"offset": i64_max}]


def test_i64_overflow_rejected() -> None:
    with pytest.raises(NdselError) as excinfo:
        normalize_ndsel({"kind": "point", "coords": [2**63]})
    assert excinfo.value.reason == "invalid_json"


def test_bool_in_output_offset_rejected() -> None:
    with pytest.raises(NdselError) as excinfo:
        normalize_ndsel({"kind": "transform", "input_rank": 0, "output": [{"offset": True}]})
    assert excinfo.value.reason == "invalid_json"


def test_sentinel_not_allowed_in_plain_integer_position() -> None:
    with pytest.raises(NdselError) as excinfo:
        normalize_ndsel({"kind": "point", "coords": ["+inf"]})
    assert excinfo.value.reason == "invalid_json"


def test_not_an_object_rejected() -> None:
    with pytest.raises(NdselError) as excinfo:
        normalize_ndsel([1, 2, 3])
    assert excinfo.value.reason == "invalid_json"


def test_empty_string_kind_is_unknown_kind() -> None:
    with pytest.raises(NdselError) as excinfo:
        normalize_ndsel({"kind": ""})
    assert excinfo.value.reason == "unknown_kind"
