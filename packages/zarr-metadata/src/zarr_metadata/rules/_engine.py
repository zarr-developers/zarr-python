"""Rules gated by the document fields they read.

A `Rule` runs when every key in `requires` is present, so one rule set
serves complete documents and partial ones without imposing field order.
Gating rather than ordering is conventional — Ecto changesets, Clojure
spec, Valibot's `partialCheck`, JSON Schema's `dependentSchemas` — and
has two consequences here.

**Order-free by construction.** No topological sort, so mutually
dependent rules are expressible.

**Absence is deliberately inexpressible.** A rule cannot ask whether a
field is missing: that is negation-as-failure, sound only under a
closed-world assumption, and a partially built document is an open world
where the key may still arrive. Required-key checks therefore stay in
structural validation.

Rules may receive structurally invalid values. A rule that cannot safely
interpret its inputs leaves the problem to structural validation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import cast

from zarr_metadata.model._validation import ValidationProblem

RuleCheck = Callable[[Mapping[str, object]], tuple[ValidationProblem, ...]]
"""A rule's check: the whole document in, every problem it finds out."""


@dataclass(frozen=True, slots=True)
class Rule:
    """One composition check over a (possibly partial) metadata document.

    `requires` are the document keys the check reads; the rule fires only
    when all of them are present. `check` receives the whole document (so
    coupled fields are examined together) and returns every problem it
    finds, empty when the rule passes.
    """

    requires: frozenset[str]
    check: RuleCheck


def applicable(rules: Sequence[Rule], present: AbstractSet[str]) -> Iterator[Rule]:
    """The subset of `rules` whose required keys are all present."""
    return (rule for rule in rules if rule.requires <= present)


def run_rules(
    rules: Sequence[Rule], document: Mapping[str, object]
) -> tuple[ValidationProblem, ...]:
    """Run every applicable rule over `document`, collecting all problems."""
    problems: list[ValidationProblem] = []
    for rule in applicable(rules, document.keys()):
        problems.extend(rule.check(document))
    return tuple(problems)


def as_string_mapping(value: object) -> Mapping[str, object] | None:
    """`value` as a string-keyed mapping, or None if it is not one."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast("Mapping[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        return None
    return cast("Mapping[str, object]", mapping)


def as_sequence(value: object) -> Sequence[object] | None:
    """`value` as a JSON-array-shaped sequence, or None if it is not one."""
    if isinstance(value, (list, tuple)):
        return cast("Sequence[object]", value)
    return None


def prefixed(
    loc: tuple[str | int, ...], problems: Sequence[ValidationProblem]
) -> tuple[ValidationProblem, ...]:
    """Re-base every problem's `loc` under `loc` (for nested documents)."""
    return tuple(
        ValidationProblem((*loc, *problem.loc), problem.message, problem.kind)
        for problem in problems
    )


__all__ = [
    "Rule",
    "RuleCheck",
    "applicable",
    "run_rules",
]
