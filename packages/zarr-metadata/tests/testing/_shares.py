"""How often a strategy draws something, in a few hundred examples.

A reach test asks whether a target can be drawn at all, in as many examples
as it takes, which says nothing of whether a test, at the 100 examples it
runs by default, ever meets it. Runs are derandomized, so a share is a fixed
number for a given strategy. A change to the strategy draws other examples,
as another seed would, so a bound sits where the share stays across seeds:
this is no place to bound what is drawn a few percent of the time.

A bound is set from the shares of 100 seeds, each run as `examples_of`
runs, but with `derandomize=False` under `hypothesis.seed`. It lies past
the most extreme share on its side by a quarter of that share's distance
from the median, rounded outward to a hundredth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from hypothesis import HealthCheck, given, settings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from hypothesis.strategies import SearchStrategy

T = TypeVar("T")

_DERANDOMIZED = settings(
    database=None,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)


def examples_of(strategy: SearchStrategy[T], *, examples: int = 300) -> list[T]:
    """The examples a derandomized run of `strategy` draws, checked to be the same in a second run.

    Hypothesis's mutator copies spans between the parts of an example that
    share a label, and a label can follow an object's address: a
    `sampled_from` of elements that hold a strategy built for the draw is
    labeled by that strategy's `id`. Such a strategy draws other examples
    from run to run, and from machine to machine, however derandomized. A
    second run in the same process lays objects out at other addresses, so
    it catches this.
    """
    first = _draw(strategy, examples)
    second = _draw(strategy, examples)
    if [repr(value) for value in first] != [repr(value) for value in second]:
        raise AssertionError(
            f"{strategy!r} drew other examples on a second derandomized run. "
            "A strategy built during a draw is likely in a `sampled_from`, which "
            "labels it by its address; sample an index into the options instead."
        )
    return first


def share(values: Sequence[T], counts: Callable[[T], bool]) -> float:
    """The share of `values` that `counts` holds of."""
    return sum(counts(value) for value in values) / len(values)


def share_of(
    strategy: SearchStrategy[T], counts: Callable[[T], bool], *, examples: int = 300
) -> float:
    """The share of the examples of `strategy` that `counts` holds of."""
    return share(examples_of(strategy, examples=examples), counts)


def _draw(strategy: SearchStrategy[T], examples: int) -> list[T]:
    drawn: list[T] = []
    # A lambda, not `drawn.append`: Hypothesis cannot wrap a builtin method,
    # and derandomizes from the test function's source, so this line is the seed.
    run = given(strategy)(lambda value: drawn.append(value))
    settings(_DERANDOMIZED, max_examples=examples)(run)()
    return drawn
