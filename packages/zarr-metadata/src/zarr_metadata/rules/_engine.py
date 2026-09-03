"""Rules gated by the document fields they read.

A `Rule` runs when every key in `requires` is present. The same rule set
therefore supports complete documents and partial builders without
imposing field order.

Prior art
---------
The gate is conventional, which is the point. Ecto's
`Ecto.Changeset.validate_change/3` invokes a validator "only if a change
for the given field exists", so one changeset serves both full inserts
and partial updates. Clojure spec's two-phase `s/keys` separates
required-key presence from key/value conformance precisely because "we
routinely deal with optional and partial data". Valibot's `partialCheck`
takes the paths a cross-field rule reads and runs it "whenever the
selected part of the data is valid". Presence-conditional rules-as-data
are JSON Schema's `dependentSchemas` / `dependentRequired` applicators.

- https://hexdocs.pm/ecto/Ecto.Changeset.html
- https://clojure.org/about/spec
- https://valibot.dev/api/partialCheck/
- https://www.learnjsonschema.com/2020-12/applicator/dependentschemas/

Two consequences follow from gating rather than ordering.

**Order-free by construction.** No topological sort, so mutually
dependent rules are expressible — unlike Yup, whose equivalent `deps`
orders rules and therefore rejects cycles outright.

**Absence is deliberately inexpressible.** A rule cannot ask whether a
field is missing: that is negation-as-failure, sound only under a
closed-world assumption, and a partially built document is an open world
where the key may still arrive. Required-key checks therefore stay in
structural validation — the same stratification Ecto
(`validate_required`), spec (`:req`), and JSON Schema (`required`) apply.

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
