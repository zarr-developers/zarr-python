"""Generative invariants for raw-bytes name canonicalization.

Scoped to canonicalization deliberately. Two earlier tests here asserted
that a shape verdict exists exactly when `(field, canonical_name(...))`
is in `modelled_entities()` — but both sides were computed from
`_ENTITY_SHAPES` through the same call, so they restated the lookup
rather than testing it, and could not fail. Worse, they could not catch
the bug class they named (a lookup passing the wrong field), because both
sides used the same field. `tests/rules/test_registry.py` covers that
with real assertions.

Canonicalization is a genuine fit for generative testing: the family is
unbounded, so an example-based test can only sample it.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from zarr_metadata.v3._extension_points import (
    CHUNK_GRID,
    CODECS,
    DATA_TYPE,
    RAW_BYTES_FAMILY,
    canonical_name,
)


@given(width=st.integers(min_value=0, max_value=2**32))
def test_every_numeric_r_spelling_folds_to_one_key(width: int) -> None:
    # Including malformed widths (0, 12, anything not a multiple of 8):
    # canonicalization is by grammar shape, not validity, so a misspelled
    # member of a family we model is reported as a misspelling rather than
    # passing as an unknown third-party extension.
    assert canonical_name(DATA_TYPE, f"r{width}") == RAW_BYTES_FAMILY


@given(width=st.integers(min_value=0, max_value=2**32), field=st.sampled_from([CODECS, CHUNK_GRID]))
def test_r_shaped_names_are_identity_outside_data_types(width: int, field: str) -> None:
    # The family belongs to `data_type`; a codec that happens to be named
    # `r8` must not be folded into it.
    name = f"r{width}"
    assert canonical_name(field, name) == name  # type: ignore[arg-type]


@given(
    name=st.text(min_size=1).filter(lambda s: not (s.startswith("r") and s[1:].isdigit())),
)
def test_non_family_names_are_identity(name: str) -> None:
    assert canonical_name(DATA_TYPE, name) == name
