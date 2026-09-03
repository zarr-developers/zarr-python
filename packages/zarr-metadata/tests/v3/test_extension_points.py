"""Tests for extension-point name canonicalization."""

from __future__ import annotations

import pytest

from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3._extension_points import (
    CODECS,
    DATA_TYPE,
    RAW_BYTES_FAMILY,
    canonical_name,
)

# (field, name, expected canonical key) — identity everywhere except the
# parameterized raw-bytes family.
CANONICAL_CASES: dict[str, tuple[str, str, str]] = {
    "plain-dtype": (DATA_TYPE, "uint8", "uint8"),
    "dotted-dtype": (DATA_TYPE, "numpy.datetime64", "numpy.datetime64"),
    "raw-8": (DATA_TYPE, "r8", RAW_BYTES_FAMILY),
    "raw-24": (DATA_TYPE, "r24", RAW_BYTES_FAMILY),
    # Malformed members canonicalize into the family too: a misspelling of
    # something we model must be reported as such, not pass as an unknown
    # third-party extension.
    "raw-not-multiple-of-8": (DATA_TYPE, "r12", RAW_BYTES_FAMILY),
    "raw-zero": (DATA_TYPE, "r0", RAW_BYTES_FAMILY),
    # Canonicalization is field-aware: the r<N> family is a data type.
    "raw-shaped-codec-name": (CODECS, "r8", "r8"),
    "codec": (CODECS, "blosc", "blosc"),
    "unknown": (CODECS, "zfpy", "zfpy"),
}


@pytest.mark.parametrize(
    ("field", "name", "expected"), CANONICAL_CASES.values(), ids=list(CANONICAL_CASES)
)
def test_canonical_name(field: str, name: str, expected: str) -> None:
    assert canonical_name(field, name) == expected  # type: ignore[arg-type]


def test_squatted_names_are_judged_against_the_definition_they_squat() -> None:
    # Zarr identifiers are registry-allocated. A private codec named
    # `bytes` has left the compatibility contract, and saying so is the
    # correct answer rather than a limitation, so nothing here defends
    # against collisions.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4, 4),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
        "chunk_key_encoding": "default",
        "codecs": ({"name": "bytes", "configuration": {"width": 7}},),
    }
    problems = validate_array_metadata_v3(document)
    assert [(p.loc, p.kind) for p in problems] == [(("codecs", 0, "configuration"), "unknown_key")]


def test_forging_the_family_sentinel_cannot_change_a_verdict() -> None:
    # A literal "r<N>" data type mislabels nothing: the rules layer matches
    # the family through the name pattern, not through the table key, so
    # no validation verdict depends on the sentinel being unforgeable.
    assert canonical_name(DATA_TYPE, RAW_BYTES_FAMILY) == RAW_BYTES_FAMILY
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4, 4),
        "data_type": RAW_BYTES_FAMILY,
        "fill_value": (1,),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
        "chunk_key_encoding": "default",
        "codecs": ("bytes",),
    }
    # Unjudged as an unknown data type, exactly as any unmodelled name is.
    assert validate_array_metadata_v3(document) == ()
