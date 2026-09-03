"""Tests for the `check_*` tagged-union entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args, get_type_hints

import pytest

from zarr_metadata.rules import (
    Invalid,
    Valid,
    ValidationResult,
    check_array_metadata_v2,
    check_array_metadata_v3,
    check_group_metadata_v2,
    check_group_metadata_v3,
)
from zarr_metadata.rules._documents import validate_array_metadata_v3

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

V3_ARRAY: Mapping[str, object] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4, 4),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}
V2_ARRAY: Mapping[str, object] = {
    "zarr_format": 2,
    "shape": (4,),
    "chunks": (2,),
    "dtype": "<i4",
    "compressor": None,
    "fill_value": 0,
    "order": "C",
    "filters": None,
}
V3_GROUP: Mapping[str, object] = {"zarr_format": 3, "node_type": "group"}
V2_GROUP: Mapping[str, object] = {"zarr_format": 2}

CASES: dict[str, tuple[Callable[[object], object], Mapping[str, object]]] = {
    "v3-array": (check_array_metadata_v3, V3_ARRAY),
    "v2-array": (check_array_metadata_v2, V2_ARRAY),
    "v3-group": (check_group_metadata_v3, V3_GROUP),
    "v2-group": (check_group_metadata_v2, V2_GROUP),
}


@pytest.mark.parametrize(("check", "doc"), CASES.values(), ids=list(CASES))
def test_valid_documents(check: Callable[[object], object], doc: Mapping[str, object]) -> None:
    result = check(doc)
    assert isinstance(result, Valid)
    assert result.valid is True
    assert result.document == doc


def test_valid_normalizes_like_parse() -> None:
    # A Valid carries the canonical document, not the caller's spelling.
    result = check_array_metadata_v3({**V3_ARRAY, "shape": [4, 4], "codecs": ["bytes"]})
    assert isinstance(result, Valid)
    assert result.document["shape"] == (4, 4)
    assert result.document["codecs"] == ("bytes",)


def test_error_invalid_carries_every_problem() -> None:
    result = check_array_metadata_v3({**V3_ARRAY, "node_type": "grid", "fill_value": 300})
    assert isinstance(result, Invalid)
    assert result.valid is False
    assert len(result.problems) != 0
    # the same report validate_* would give, not a summary of it
    assert result.problems == validate_array_metadata_v3(
        {**V3_ARRAY, "node_type": "grid", "fill_value": 300}
    )


def test_error_invalid_cannot_have_an_empty_report() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Invalid(())


def test_public_result_type_is_runtime_subscriptable() -> None:
    assert get_args(ValidationResult[int]) == (Valid[int], Invalid)


def test_public_check_annotations_resolve_at_runtime() -> None:
    hints = get_type_hints(check_array_metadata_v3)
    assert get_args(hints["return"])[1] is Invalid


def test_discriminant_narrows_both_ways() -> None:
    # The point of the union: `valid` selects which member is readable.
    good = check_array_metadata_v3(V3_ARRAY)
    if good.valid:
        assert good.document["zarr_format"] == 3
    else:  # pragma: no cover - the fixture is valid
        pytest.fail("expected a Valid result")

    bad = check_array_metadata_v3({**V3_ARRAY, "fill_value": 300})
    if bad.valid:  # pragma: no cover - the fixture is invalid
        pytest.fail("expected an Invalid result")
    else:
        assert any(problem.loc == ("fill_value",) for problem in bad.problems)
