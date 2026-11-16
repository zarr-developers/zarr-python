"""How often a strategy draws something, in a few hundred examples.

A reach test asks whether a target can be drawn at all, in as many examples
as it takes, which says nothing of whether a test, at the 100 examples it
runs by default, ever meets it. Runs are derandomized, so a share is a fixed
number for a given strategy. A change to the strategy draws other examples,
as another seed would, and so does a change to the line in `_draw` that
Hypothesis derandomizes from, which reseeds every share at once. So a bound
sits where the share stays across seeds: this is no place to bound what is
drawn a few percent of the time.

A bound is reset with `bounds(across_seeds(strategy, statistic))`, which
runs 100 seeds and applies the rule `bounds` states.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any, TypeVar, cast

from hypothesis import HealthCheck, given, seed, settings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from hypothesis.strategies import SearchStrategy

T = TypeVar("T")

_SETTINGS = settings(
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)


def examples_of(strategy: SearchStrategy[T], *, examples: int = 300) -> tuple[T, ...]:
    """The examples a derandomized run of `strategy` draws, drawn once per strategy and checked against a second run.

    Hypothesis's mutator copies spans between the parts of an example that
    share a label, and a label can follow an object's address: a
    `sampled_from` of elements that hold a strategy is labeled by that
    strategy's `id`. Such a strategy draws other examples from run to run,
    and from machine to machine, however derandomized. A second run in the
    same process often lays objects out at other addresses, so it often
    catches this, but it is a tripwire, not a proof.
    """
    return cast("tuple[T, ...]", _checked(strategy, examples))


def share(values: Iterable[T], counts: Callable[[T], bool]) -> float:
    """The share of `values` that `counts` holds of."""
    tally = [counts(value) for value in values]
    return sum(tally) / len(tally)


def share_of(
    strategy: SearchStrategy[T], counts: Callable[[T], bool], *, examples: int = 300
) -> float:
    """The share of the examples of `strategy` that `counts` holds of."""
    return share(examples_of(strategy, examples=examples), counts)


def across_seeds(
    strategy: SearchStrategy[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    seeds: Iterable[int] = range(100),
    examples: int = 300,
) -> list[float]:
    """`statistic` of the examples of `strategy` under each of `seeds`, as a change to the strategy would move it."""
    return [statistic(_draw(strategy, examples, chosen)) for chosen in seeds]


def bounds(statistics: Sequence[float]) -> tuple[float, float]:
    """A low and a high bound for a statistic measured across seeds.

    Each lies past the most extreme measure on its side by a quarter of that
    measure's distance from the median, rounded outward to a hundredth.
    """
    ordered = sorted(statistics)
    median = ordered[len(ordered) // 2]
    low = ordered[0] - (median - ordered[0]) / 4
    high = ordered[-1] + (ordered[-1] - median) / 4
    return math.floor(round(low * 100, 6)) / 100, math.ceil(round(high * 100, 6)) / 100


@functools.cache
def _checked(strategy: SearchStrategy[Any], examples: int) -> tuple[Any, ...]:
    first = _draw(strategy, examples)
    second = _draw(strategy, examples)
    if [repr(value) for value in first] != [repr(value) for value in second]:
        raise AssertionError(
            f"{strategy!r} drew other examples on a second derandomized run. "
            "A strategy is likely in a `sampled_from`, which labels it by its "
            "address; sample a name or an index instead."
        )
    return first


def _draw(strategy: SearchStrategy[T], examples: int, chosen: int | None = None) -> tuple[T, ...]:
    """Derandomized examples of `strategy`, or seeded with `chosen`."""
    drawn: list[T] = []
    # A lambda, not `drawn.append`: Hypothesis cannot wrap a builtin method,
    # and derandomizes from the test function's source, so this line is the seed.
    run = given(strategy)(lambda value: drawn.append(value))
    if chosen is None:
        settings(_SETTINGS, max_examples=examples, derandomize=True)(run)()
    else:
        settings(_SETTINGS, max_examples=examples, derandomize=False)(seed(chosen)(run))()
    return tuple(drawn)
