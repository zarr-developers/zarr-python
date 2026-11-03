"""How often a strategy draws something, in a few hundred examples.

A reach test asks whether a target can be drawn at all, in as many examples
as it takes, which says nothing of whether a test, at the 100 examples it
runs by default, ever meets it. Runs are derandomized, which keeps most shares
fixed, but a share can still move a few points between processes, so a bound
needs room: this is no place to bound what is drawn a few percent of the time.
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
