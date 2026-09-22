"""The first layer of reading: a value refined to JSON, or the reasons it is not."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING, cast

import pytest

from zarr_metadata.model import MetadataValidationError, ValidationProblem
from zarr_metadata.model._validation import refine_json, refine_node_json, stored_json_problems

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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
    # One walk normalizes and judges; what comes back is what every later
    # layer takes, and nothing later normalizes again.
    assert refine_json(value) == (refined, ())


@pytest.mark.parametrize(
    ("value", "loc", "kind"),
    [
        ({"a": object()}, ("a",), "invalid_type"),
        ([1, [2, {3: 4}]], (1, 1), "invalid_type"),
        ({"x": math.inf}, ("x",), "invalid_value"),
        ({"x": [math.nan]}, ("x", 0), "invalid_value"),
        (b"bytes", (), "invalid_type"),
    ],
    ids=["not-json-leaf", "non-string-key", "infinite", "nan-in-array", "bytes"],
)
def test_error_a_value_that_is_not_json_is_none_with_the_leaf_located(
    value: object, loc: tuple[str | int, ...], kind: str
) -> None:
    refined, problems = refine_json(value)
    assert refined is None
    assert [(problem.loc, problem.kind) for problem in problems] == [(loc, kind)]


def test_error_every_leaf_that_is_not_json_is_reported() -> None:
    refined, problems = refine_json({"a": object(), "b": [1, object()]})
    assert refined is None
    assert [problem.loc for problem in problems] == [("a",), ("b", 1)]


def test_error_the_error_refuses_what_is_not_a_problem() -> None:
    # `problem()` in the entity layer returns a one-element tuple; a list
    # of those passes the type at the call and fails far away otherwise.
    with pytest.raises(TypeError, match="collect with `extend`, not `append`"):
        MetadataValidationError([(ValidationProblem(("a",), "bad a", "invalid_value"),)])  # pyright: ignore[reportArgumentType]


# A node's attributes are user data: Python's `json` writes a non-finite
# number there as `NaN`/`Infinity`, and zarr-python writes attributes that
# way, so the model reads, validates, and writes one there -- and nowhere
# else, where the spec spells those numbers as strings.

_INLINE: dict[str, object] = {"kind": "inline", "must_understand": False}


def _at(value: object, path: tuple[str | int, ...]) -> object:
    for part in path:
        if isinstance(part, str):
            value = cast("Mapping[str, object]", value)[part]
        else:
            value = cast("Sequence[object]", value)[part]
    return value


@pytest.mark.parametrize(
    ("document", "path"),
    [
        ({"attributes": {"_FillValue": math.nan}}, ("attributes", "_FillValue")),
        ({"attributes": {"valid_range": [-math.inf, math.inf]}}, ("attributes", "valid_range", 1)),
        (
            {"attributes": {"cf": {"missing_value": -math.inf}}},
            ("attributes", "cf", "missing_value"),
        ),
        (
            {
                "consolidated_metadata": {
                    **_INLINE,
                    "metadata": {"a": {"attributes": {"x": math.nan}}},
                }
            },
            ("consolidated_metadata", "metadata", "a", "attributes", "x"),
        ),
    ],
    ids=["nan", "infinities-in-an-array", "nested-object", "consolidated-node"],
)
def test_a_node_documents_attributes_hold_non_finite_numbers(
    document: dict[str, object], path: tuple[str | int, ...]
) -> None:
    refined, problems = refine_node_json(document)
    assert problems == ()
    held = _at(refined, path)
    assert isinstance(held, float)
    assert not math.isfinite(held)
    # What the model reads, it writes back, in the spelling Python's
    # `json` module and zarr-python use.
    assert stored_json_problems("zarr.json", document) == ()
    assert stored_json_problems(".zattrs", _at(document, path[:-1])) == ()


@pytest.mark.parametrize(
    ("document", "loc"),
    [
        ({"fill_value": math.nan}, ("fill_value",)),
        (
            {"codecs": ({"name": "scale_offset", "configuration": {"scale": math.inf}},)},
            ("codecs", 0, "configuration", "scale"),
        ),
        ({"extension": {"must_understand": False, "x": math.nan}}, ("extension", "x")),
        (
            {"consolidated_metadata": {**_INLINE, "metadata": {"a": {"fill_value": math.nan}}}},
            ("consolidated_metadata", "metadata", "a", "fill_value"),
        ),
    ],
    ids=["fill-value", "codec-configuration", "extension-field", "consolidated-node"],
)
def test_error_a_non_finite_number_outside_attributes_is_not_json(
    document: dict[str, object], loc: tuple[str | int, ...]
) -> None:
    refined, problems = refine_node_json(document)
    assert refined is None
    assert [(problem.loc, problem.kind) for problem in problems] == [(loc, "invalid_value")]
    assert [problem.loc for problem in stored_json_problems("zarr.json", document)] == [loc]


def test_error_a_zarray_holds_no_user_data() -> None:
    # A v2 array's attributes live in `.zattrs`; its `.zarray` is RFC 8259
    # throughout -- which is where zarr-python 3.0 once wrote a bare `NaN`.
    problems = stored_json_problems("a/.zarray", {"fill_value": math.nan})
    assert [(problem.loc, problem.kind) for problem in problems] == [
        (("fill_value",), "invalid_value")
    ]


def test_error_a_zmetadata_entry_is_judged_as_the_document_its_key_names() -> None:
    consolidated = {
        "zarr_consolidated_format": 1,
        "metadata": {"a/.zattrs": {"x": math.nan}, "a/.zarray": {"fill_value": math.nan}},
    }
    problems = stored_json_problems(".zmetadata", consolidated)
    assert [problem.loc for problem in problems] == [("metadata", "a/.zarray", "fill_value")]
