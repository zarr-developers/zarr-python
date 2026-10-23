"""Tests for refining a value to JSON, or reporting why it is not JSON."""

from __future__ import annotations

import math
from collections import OrderedDict, UserDict
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

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
    # A single walk both normalizes and validates.
    assert refine_json(value) == (refined, ())


def test_user_data_holds_non_finite_numbers() -> None:
    # User data is JSON as Python's `json` module reads and writes it. The
    # walk is the same as in `refine_json`, and a non-finite number stays a float.
    assert refine_user_data({"range": [-math.inf, math.inf]}) == (
        {"range": (-math.inf, math.inf)},
        (),
    )
    refined, problems = refine_user_data(math.nan)
    assert problems == ()
    assert isinstance(refined, float)
    assert math.isnan(refined)


# User data, like JSON, is refused at any position where it is not JSON.
@pytest.mark.parametrize("refine", [refine_json, refine_user_data], ids=["json", "user-data"])
@pytest.mark.parametrize(
    ("value", "loc"),
    # `bytes` is a sequence of integers, but it is not a JSON array.
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
    # The walk uses one frame per level of nesting, up to `JSON_DEPTH` levels.
    deep: dict[str, object] = {}
    for _ in range(JSON_DEPTH - 1):
        deep = {"k": deep}
    refined, problems = refine_json(deep)
    assert problems == ()
    assert refined is not None


def test_error_a_value_nested_deeper_than_a_reader_walks() -> None:
    # The first level past the limit is reported as the problem, at any
    # position. This keeps any document from pushing a reader past the
    # interpreter's recursion limit.
    deep: dict[str, object] = {}
    for _ in range(JSON_DEPTH + 40):
        deep = {"k": deep}
    refined, problems = refine_json({"a": [deep]})
    assert refined is None
    assert [(len(p.loc), p.kind, p.message) for p in problems] == [
        (JSON_DEPTH, "invalid_value", f"nested deeper than the {JSON_DEPTH} levels a reader walks")
    ]
    assert problems[0].loc[:2] == ("a", 0)


def _nested(depth: int, innermost: object) -> list[object]:
    """Wrap `innermost` in `depth` nested arrays, so `innermost` is `depth` levels down."""
    value: object = innermost
    for _ in range(depth):
        value = [value]
    return cast("list[object]", value)


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
        (_nested(JSON_DEPTH, []), "a value nested too deep to show"),
        (_nested(JSON_DEPTH, {1}), "[" * JSON_DEPTH + "{1}" + "]" * JSON_DEPTH),
    ],
    ids=[
        "null",
        "true",
        "string",
        "array",
        "object",
        "non-finite",
        "not-json",
        "past-the-levels",
        "not-json-at-the-last-level",
    ],
)
def test_a_value_is_shown_as_the_json_a_document_writes(value: object, text: str) -> None:
    """A problem's message shows a value as JSON, or as its repr when it is not JSON. When a container is nested past the levels a reader walks, the message says so instead. A non-JSON value on the last level is still shown."""
    assert shown(value) == text


def test_bytes_past_the_levels_a_reader_walks_are_not_json_rather_than_nested() -> None:
    # `bytes` is a sequence to Python but not a container to JSON, at any
    # position. Past the depth cap it is still reported as a non-JSON value,
    # not as nesting.
    deep: list[object] = [b""]
    for _ in range(JSON_DEPTH - 1):
        deep = [deep]
    _, problems = refine_json(deep)
    assert [(len(p.loc), p.kind, p.message[:31]) for p in problems] == [
        (JSON_DEPTH, "invalid_type", "not a JSON-serializable value: ")
    ]


def test_error_an_integer_of_more_digits_than_json_text_holds_is_a_problem(
    interpreter_writes_4300_digits: None,
) -> None:
    """An integer with more digits than `sys.get_int_max_str_digits` allows cannot be converted to text. It is reported as an `invalid_value` problem at its location, so no reader raises on it. The document could not be written back. `validate_*` and `read_*` report the same problems."""
    from zarr_metadata.model import (
        ZarrV3ArrayMetadata,
        read_array_metadata_v3,
        validate_array_metadata_v3,
    )

    refined, problems = refine_json({"x": [10**5000]})
    assert refined is None
    assert [(p.loc, p.kind) for p in problems] == [(("x", 0), "invalid_value")]
    assert "digits" in problems[0].message
    document = {**ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json()}
    document["attributes"] = {"x": 10**5000}
    found = validate_array_metadata_v3(document)
    assert [(p.loc, p.kind) for p in found] == [(("attributes", "x"), "invalid_value")]
    assert read_array_metadata_v3(document).problems == found
    shaped = {**document, "attributes": {}, "shape": [10**5000]}
    assert [p.loc for p in validate_array_metadata_v3(shaped)] == [("shape", 0)]


def test_a_str_or_int_subclass_is_refined_to_the_json_type_it_is() -> None:
    """A `StrEnum` member, an `IntEnum` member, or any `str`, `int` or `float` subclass is refined to the plain value that JSON writes for it. A `Literal` or a tag then reads it as that value. `bool` stays separate from `int`."""
    import enum

    from zarr_metadata.model import ZarrV3GroupMetadata

    class NodeType(enum.StrEnum):
        GROUP = "group"

    class Format(enum.IntEnum):
        THREE = 3

    class Score(float):
        pass

    refined, problems = refine_json(
        {"a": NodeType.GROUP, "b": Format.THREE, "c": Score(1.5), "d": True}
    )
    assert problems == ()
    assert refined == {"a": "group", "b": 3, "c": 1.5, "d": True}
    assert isinstance(refined, dict)
    assert all(type(refined[key]) in (str, int, float, bool) for key in refined)
    assert type(refined["d"]) is bool
    model = ZarrV3GroupMetadata.from_json(
        {"zarr_format": Format.THREE, "node_type": NodeType.GROUP}
    )
    assert model.to_json() == {"zarr_format": 3, "node_type": "group"}


def test_a_mapping_or_sequence_of_another_type_is_refined_as_a_dict_or_list_is() -> None:
    """A mapping proxy, a `UserDict`, a subclass of `dict` or `list`, or a tuple is refined to the same JSON as the equivalent dict or list. It is validated the same way, with each problem at its location."""

    class Listed(list[object]):
        pass

    class Mapped(dict[str, object]):
        pass

    held: dict[str, object] = {"a": [1, {"b": (2, 3)}], "c": "x", "d": [True, None, 2.5]}
    expected = refine_json(held)
    assert expected == ({"a": (1, {"b": (2, 3)}), "c": "x", "d": (True, None, 2.5)}, ())
    for other in (
        MappingProxyType(held),
        UserDict(held),
        Mapped(held),
        {**held, "a": Listed(cast("list[object]", held["a"]))},
    ):
        assert refine_json(other) == expected
    bad: dict[str, object] = {"a": [1, {"b": math.nan}], "c": {1, 2}}
    for other in (
        bad,
        MappingProxyType(bad),
        Mapped(bad),
        {**bad, "a": Listed([1, {"b": math.nan}])},
    ):
        refined, problems = refine_json(other)
        assert refined is None
        assert [(p.loc, p.kind) for p in problems] == [
            (("a", 1, "b"), "invalid_value"),
            (("c",), "invalid_type"),
        ]
