"""Canonicalization: the simplest spelling with the same meaning.

Two properties carry the weight. Canonicalizing twice must change nothing
further, or the form is not canonical. And canonicalizing must never
change a verdict, or it is not meaning-preserving — that one is what
catches a "simplification" that quietly says something else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings

from tests.rules.strategies import valid_documents
from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.rules._canonical import Canonical, Invalid, canonicalize_array_metadata_v3

if TYPE_CHECKING:
    from collections.abc import Mapping

_SLOW = settings(max_examples=300, deadline=None, suppress_health_check=(HealthCheck.too_slow,))

BASE: Mapping[str, object] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (64,),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (32,)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}


def _canonical(**overrides: object) -> Mapping[str, object]:
    result = canonicalize_array_metadata_v3({**BASE, **overrides})  # type: ignore[arg-type]
    assert isinstance(result, Canonical), result
    return result.document


# (what it should simplify, the field it lands in, the expected canonical value)
SIMPLIFICATIONS: dict[str, tuple[dict[str, object], str, object]] = {
    "object-with-no-configuration": ({"codecs": ({"name": "bytes"},)}, "codecs", ("bytes",)),
    "object-with-empty-configuration": (
        {"codecs": ({"name": "bytes", "configuration": {}},)},
        "codecs",
        ("bytes",),
    ),
    "defaulted-must-understand": (
        {"codecs": ({"name": "bytes", "must_understand": True},)},
        "codecs",
        ("bytes",),
    ),
    "chunk-key-encoding-shorthand": (
        {"chunk_key_encoding": {"name": "default"}},
        "chunk_key_encoding",
        "default",
    ),
    "rectilinear-runs-encode": (
        {
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": ((32, 32),)},
            }
        },
        "chunk_grid",
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (((32, 2),),)},
        },
    ),
}


@pytest.mark.parametrize(
    ("overrides", "field", "expected"), SIMPLIFICATIONS.values(), ids=list(SIMPLIFICATIONS)
)
def test_simplifies(overrides: dict[str, object], field: str, expected: object) -> None:
    assert _canonical(**overrides)[field] == expected


def test_error_an_extension_point_may_not_be_declared_ignorable() -> None:
    # `must_understand` belongs to the kind of metadata, not to a use of
    # it, and no extension point is skippable -- so there is nothing for
    # canonicalization to keep.
    result = canonicalize_array_metadata_v3(
        {**BASE, "codecs": ({"name": "bytes", "must_understand": False},)}  # type: ignore[arg-type]
    )
    assert isinstance(result, Invalid)
    assert [problem.loc for problem in result.problems] == [("codecs", 0, "must_understand")]


def test_blosc_drops_a_typesize_that_shuffle_renders_ignored() -> None:
    configuration = {"cname": "zstd", "clevel": 5, "blocksize": 0, "typesize": 4}
    dropped = _canonical(
        codecs=(
            "bytes",
            {"name": "blosc", "configuration": {**configuration, "shuffle": "noshuffle"}},
        )
    )["codecs"][1]  # type: ignore[index]
    assert "typesize" not in dropped["configuration"]  # type: ignore[index]
    kept = _canonical(
        codecs=(
            "bytes",
            {"name": "blosc", "configuration": {**configuration, "shuffle": "shuffle"}},
        )
    )["codecs"][1]  # type: ignore[index]
    assert kept["configuration"]["typesize"] == 4  # type: ignore[index]


def test_dimension_names_of_nothing_but_nulls_are_dropped() -> None:
    assert "dimension_names" not in _canonical(dimension_names=(None,))
    assert _canonical(dimension_names=("x",))["dimension_names"] == ("x",)


def test_a_rectilinear_step_is_not_expanded() -> None:
    # A bare integer repeats to cover the extent, so it is not equivalent
    # to any fixed list -- and a one-element list is not equivalent to it.
    for spec in (32, (32,)):
        grid = {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (spec,)},
        }
        shape = (64,) if spec == 32 else (32,)
        result = canonicalize_array_metadata_v3({**BASE, "shape": shape, "chunk_grid": grid})  # type: ignore[arg-type]
        assert isinstance(result, Canonical), result
        assert result.document["chunk_grid"]["configuration"]["chunk_shapes"] == (spec,)  # type: ignore[index]


def test_error_a_semantically_invalid_document_reports_instead() -> None:
    result = canonicalize_array_metadata_v3({**BASE, "fill_value": 999})  # type: ignore[arg-type]
    assert isinstance(result, Invalid)
    # The field is the location, not part of the message: the data type
    # says what it accepts, and the document says where it was asked.
    assert [problem.loc for problem in result.problems] == [("fill_value",)]


def test_error_invalid_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Invalid(())


@given(valid_documents())
@_SLOW
def test_canonicalizing_twice_changes_nothing_further(doc: Mapping[str, object]) -> None:
    once = canonicalize_array_metadata_v3(doc)  # type: ignore[arg-type]
    assert isinstance(once, Canonical), once
    twice = canonicalize_array_metadata_v3(once.document)
    assert isinstance(twice, Canonical), twice
    assert twice.document == once.document


@given(valid_documents())
@_SLOW
def test_canonicalizing_never_changes_the_verdict(doc: Mapping[str, object]) -> None:
    # A simplification that changed meaning would show up here as a
    # document that validated before and does not after.
    result = canonicalize_array_metadata_v3(doc)  # type: ignore[arg-type]
    assert isinstance(result, Canonical), result
    assert validate_array_metadata_v3(result.document) == ()
