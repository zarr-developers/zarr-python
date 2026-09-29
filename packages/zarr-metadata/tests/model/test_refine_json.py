"""A value refined to JSON, or the reasons it is not."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest

from zarr_metadata._json import JSON_DEPTH, refine_json, refine_user_data, shown

if TYPE_CHECKING:
    from collections.abc import Callable

    from zarr_metadata._common import JSONValue
    from zarr_metadata.model import ValidationProblem

    _Refiner = Callable[[object], tuple[JSONValue | None, tuple[ValidationProblem, ...]]]


@pytest.mark.parametrize(
    ("value", "refined"),
    [
        (1, 1),
        (2.5, 2.5),
        (True, True),
        ("x", "x"),
        (None, None),
        ([1, [2, 3]], (1, (2, 3))),
        ((1, 2), (1, 2)),
        ({"a": [1], "b": {"c": None}}, {"a": (1,), "b": {"c": None}}),
        (OrderedDict(k=[0]), {"k": (0,)}),
    ],
    ids=["int", "float", "bool", "str", "null", "nested-lists", "tuple", "object", "any-mapping"],
)
def test_json_is_refined_to_tuples_and_dicts(value: object, refined: object) -> None:
    # One walk normalizes and judges.
    assert refine_json(value) == (refined, ())


def test_user_data_holds_non_finite_numbers() -> None:
    # User data is JSON as Python's `json` module reads and writes it: the
    # walk is `refine_json`'s, and a non-finite number is the float it is.
    assert refine_user_data({"range": [-math.inf, math.inf]}) == (
        {"range": (-math.inf, math.inf)},
        (),
    )
    refined, problems = refine_user_data(math.nan)
    assert problems == ()
    assert isinstance(refined, float)
    assert math.isnan(refined)


# User data is refused, as JSON is, wherever it is not JSON.
@pytest.mark.parametrize("refine", [refine_json, refine_user_data], ids=["json", "user-data"])
@pytest.mark.parametrize(
    ("value", "loc"),
    # `bytes` is a sequence of integers, and still not a JSON array.
    [({"a": object()}, ("a",)), (b"bytes", ())],
    ids=["object", "bytes"],
)
def test_error_a_leaf_that_is_not_json_is_located(
    refine: _Refiner, value: object, loc: tuple[str | int, ...]
) -> None:
    refined, problems = refine(value)
    assert refined is None
    assert [(problem.loc, problem.kind) for problem in problems] == [(loc, "invalid_type")]


@pytest.mark.parametrize("refine", [refine_json, refine_user_data], ids=["json", "user-data"])
def test_error_a_non_string_key_is_located_at_its_object(refine: _Refiner) -> None:
    refined, problems = refine([1, [2, {3: 4}]])
    assert refined is None
    assert [(problem.loc, problem.kind) for problem in problems] == [((1, 1), "invalid_type")]


@pytest.mark.parametrize(
    ("value", "loc"),
    [({"x": math.inf}, ("x",)), ({"x": [math.nan]}, ("x", 0))],
    ids=["infinite", "nan-in-array"],
)
def test_error_a_non_finite_number_is_not_json(value: object, loc: tuple[str | int, ...]) -> None:
    refined, problems = refine_json(value)
    assert refined is None
    assert [(problem.loc, problem.kind) for problem in problems] == [(loc, "invalid_value")]


def test_error_every_leaf_that_is_not_json_is_reported() -> None:
    refined, problems = refine_json({"a": object(), "b": [1, object()]})
    assert refined is None
    assert [problem.loc for problem in problems] == [("a",), ("b", 1)]


def test_a_value_nested_hundreds_deep_is_read() -> None:
    # One frame per level of nesting, up to `JSON_DEPTH` of them.
    deep: dict[str, object] = {}
    for _ in range(JSON_DEPTH - 1):
        deep = {"k": deep}
    refined, problems = refine_json(deep)
    assert problems == ()
    assert refined is not None


def test_error_a_value_nested_deeper_than_a_reader_walks() -> None:
    # The level past the last is the problem, wherever it sits, so no
    # document takes a reader past what the interpreter allows.
    deep: dict[str, object] = {}
    for _ in range(JSON_DEPTH + 40):
        deep = {"k": deep}
    refined, problems = refine_json({"a": [deep]})
    assert refined is None
    assert [(len(p.loc), p.kind, p.message) for p in problems] == [
        (JSON_DEPTH, "invalid_value", f"nested deeper than the {JSON_DEPTH} levels a reader walks")
    ]
    assert problems[0].loc[:2] == ("a", 0)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, "null"),
        (True, "true"),
        ("C", '"C"'),
        ((1, (2,)), "[1, [2]]"),
        ({"a": None}, '{"a": null}'),
        (float("nan"), "NaN"),
        ({1: 2}, "{1: 2}"),
    ],
    ids=["null", "true", "string", "array", "object", "non-finite", "not-json"],
)
def test_a_value_is_shown_as_the_json_a_document_writes(value: object, text: str) -> None:
    # What a problem's message says a document holds: its JSON, and the
    # value's repr only when it is not JSON.
    assert shown(value) == text


def test_bytes_past_the_levels_a_reader_walks_are_not_json_rather_than_nested() -> None:
    # `bytes` is a sequence to Python and no container to JSON, wherever
    # it sits: past the cap it is still what it is, not nesting.
    deep: list[object] = [b""]
    for _ in range(JSON_DEPTH - 1):
        deep = [deep]
    _, problems = refine_json(deep)
    assert [(len(p.loc), p.kind, p.message[:31]) for p in problems] == [
        (JSON_DEPTH, "invalid_type", "not a JSON-serializable value: ")
    ]
