"""Generative invariants for how a family is resolved.

Scoped to resolution deliberately. Two earlier tests here asserted that a
shape verdict exists exactly when `(field, canonical_name(...))` is in
`modelled_entities()` — but both sides were computed from `_ENTITY_SHAPES`
through the same call, so they restated the lookup rather than testing it,
and could not fail. Worse, they could not catch the bug class they named
(a lookup passing the wrong field), because both sides used the same
field. `tests/rules/test_registry.py` covers that with real assertions.

A family is a genuine fit for generative testing: it is unbounded, so an
example-based test can only sample it.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from zarr_metadata.v3.data_type.raw import RawBytesDataType
from zarr_metadata.v3.entity import (
    CHUNK_GRID,
    CODECS,
    CORE_AND_EXTENSIONS,
    DATA_TYPE,
)


@given(width=st.integers(min_value=0, max_value=2**32))
def test_every_numeric_r_spelling_resolves_to_the_family(width: int) -> None:
    # Including malformed widths (0, 12, anything not a multiple of 8):
    # the family claims a name by grammar shape, not by validity, so a
    # misspelled member of a family we model is reported as a misspelling
    # rather than passing as an unknown third-party extension.
    assert CORE_AND_EXTENSIONS.resolve(DATA_TYPE, f"r{width}") is RawBytesDataType


@given(width=st.integers(min_value=0, max_value=2**32), field=st.sampled_from([CODECS, CHUNK_GRID]))
def test_r_shaped_names_resolve_to_nothing_outside_data_types(width: int, field: str) -> None:
    # The family belongs to `data_type`; a codec that happens to be named
    # `r8` must not reach it.
    assert CORE_AND_EXTENSIONS.resolve(field, f"r{width}") is None  # type: ignore[arg-type]


# The scan `resolve` falls back to asks every entity, so a name no entity
# claims has to come back as nothing however many are registered.
_UNCLAIMED = st.text(min_size=1).filter(
    lambda name: (
        not (name.startswith("r") and name[1:].isdigit())
        and name not in CORE_AND_EXTENSIONS.entities["data_type"]
    )
)


@given(name=_UNCLAIMED)
def test_a_name_no_entity_claims_resolves_to_nothing(name: str) -> None:
    assert CORE_AND_EXTENSIONS.resolve(DATA_TYPE, name) is None
