"""A value refined to JSON, or the reasons it is not."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest

from zarr_metadata.model._validation import refine_json, refine_user_data

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
    # One frame per level of nesting: as deep as the interpreter goes, less
    # what the test runner's own frames take.
    deep: dict[str, object] = {}
    for _ in range(600):
        deep = {"k": deep}
    refined, problems = refine_json(deep)
    assert problems == ()
    assert refined is not None
