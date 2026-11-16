"""The helpers that measure and bound how often a strategy draws something."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import strategies as st

from tests.testing._shares import bounds, examples_of

_COUNTER = itertools.count()


def test_bounds() -> None:
    """Past each extreme by a quarter of its distance from the median, rounded outward."""
    assert bounds([0.1, 0.2, 0.3]) == (0.07, 0.33)
    assert bounds([0.3, 0.1, 0.2]) == (0.07, 0.33)
    assert bounds([0.5, 0.5, 0.5]) == (0.5, 0.5)
    assert bounds([0.0, 0.4, 0.4, 1.0]) == (-0.1, 1.15)


def test_a_strategy_that_draws_otherwise_on_a_second_run_is_an_assertion_error() -> None:
    drifting = st.builds(lambda: next(_COUNTER))
    with pytest.raises(AssertionError, match="second derandomized run"):
        examples_of(drifting, examples=5)
