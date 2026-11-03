"""How often a strategy draws something, in the 100 examples a test runs by default.

A reach test asks whether a target can be drawn at all, in as many examples
as it takes, which says nothing of whether a test ever meets it. Runs are
derandomized, so a share is the same on every run of a test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from hypothesis import HealthCheck, given, settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypothesis.strategies import SearchStrategy

T = TypeVar("T")


def share_of(
    strategy: SearchStrategy[T], counts: Callable[[T], bool], *, examples: int = 300
) -> float:
    """The share of the examples of `strategy` that `counts` holds of."""
    tally: list[bool] = []
    run = settings(
        max_examples=examples,
        database=None,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )(given(strategy)(lambda value: tally.append(counts(value))))
    run()
    return sum(tally) / len(tally)
