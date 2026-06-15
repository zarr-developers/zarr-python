"""Tests for :func:`zarr.core.json_parse.parse_json`.

These cover every dispatch category described in section 4.3 of the SDD:
primitives (incl. the bool/int edge case), ``Literal``, unions / ``Optional``,
fixed-length and variadic ``tuple``, ``Sequence`` / ``list`` (with coercion to
``tuple``), ``Mapping`` / ``dict``, ``TypedDict`` (required/optional/nested),
and the fallback ``TypeError`` for unsupported annotations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Union

import pytest

# ``parse_json`` (the module under test) uses ``typing_extensions`` for its
# TypedDict bookkeeping, so we build TypedDicts the same way for consistency.
from typing_extensions import NotRequired, TypedDict

from zarr.core.json_parse import parse_json

if TYPE_CHECKING:
    from typing import Any


# ---------------------------------------------------------------------------
# None / type(None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("annotation", [None, type(None)])
def test_none_valid(annotation: Any) -> None:
    assert parse_json(None, annotation) is None


@pytest.mark.parametrize("annotation", [None, type(None)])
@pytest.mark.parametrize("value", [0, "", False, [], {}])
def test_none_invalid(annotation: Any, value: Any) -> None:
    with pytest.raises(ValueError, match="Expected None"):
        parse_json(value, annotation)


# ---------------------------------------------------------------------------
# Primitives: str, int, float, bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        ("hello", str),
        ("", str),
        (1, int),
        (-5, int),
        (1.5, float),
        (-0.0, float),
        (True, bool),
        (False, bool),
    ],
)
def test_primitive_valid(value: Any, annotation: type) -> None:
    result = parse_json(value, annotation)
    assert result == value
    # The exact object is returned unchanged for primitives.
    assert result is value


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        (1, str),
        (1.0, str),
        (True, str),
        ("1", int),
        (1.0, int),
        ("x", float),
        (1, float),  # an int is NOT a float here
        ("true", bool),
        (0, bool),
    ],
)
def test_primitive_wrong_type_raises(value: Any, annotation: type) -> None:
    with pytest.raises(ValueError, match="Expected"):
        parse_json(value, annotation)


# --- The CRITICAL bool/int edge cases -------------------------------------


def test_bool_not_accepted_as_int() -> None:
    """``True`` (an ``int`` subclass instance) must NOT satisfy ``int``."""
    with pytest.raises(ValueError, match="Expected int"):
        parse_json(True, int)
    with pytest.raises(ValueError, match="Expected int"):
        parse_json(False, int)


def test_int_not_accepted_as_bool() -> None:
    """A plain ``int`` must NOT satisfy ``bool``."""
    with pytest.raises(ValueError, match="Expected bool"):
        parse_json(1, bool)
    with pytest.raises(ValueError, match="Expected bool"):
        parse_json(0, bool)


def test_bool_accepted_as_bool() -> None:
    assert parse_json(True, bool) is True
    assert parse_json(False, bool) is False


def test_int_accepted_as_int() -> None:
    assert parse_json(1, int) == 1
    assert parse_json(0, int) == 0


def test_bool_not_accepted_as_float() -> None:
    """``bool`` is an ``int`` subclass and must not satisfy ``float`` either."""
    with pytest.raises(ValueError, match="Expected float"):
        parse_json(True, float)


# ---------------------------------------------------------------------------
# Literal[...]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["C", "F"])
def test_literal_str_member_valid(value: str) -> None:
    assert parse_json(value, Literal["C", "F"]) == value


@pytest.mark.parametrize("value", ["Q", "c", "", 0])
def test_literal_str_non_member_raises(value: Any) -> None:
    with pytest.raises(ValueError, match="Expected one of"):
        parse_json(value, Literal["C", "F"])


@pytest.mark.parametrize("value", [1, 2])
def test_literal_int_member_valid(value: int) -> None:
    assert parse_json(value, Literal[1, 2]) == value


def test_literal_int_non_member_raises() -> None:
    with pytest.raises(ValueError, match="Expected one of"):
        parse_json(3, Literal[1, 2])


def test_literal_true_does_not_match_int_literal() -> None:
    """``True == 1`` in Python, but ``True`` must NOT satisfy ``Literal[1, 2]``."""
    with pytest.raises(ValueError, match="Expected one of"):
        parse_json(True, Literal[1, 2])


def test_literal_one_does_not_match_bool_literal() -> None:
    """Conversely, ``1`` must not satisfy a bool literal."""
    with pytest.raises(ValueError, match="Expected one of"):
        parse_json(1, Literal[True, False])


def test_literal_mixed_members() -> None:
    annotation = Literal["a", 3, None]
    assert parse_json("a", annotation) == "a"
    assert parse_json(3, annotation) == 3
    # ``None`` is a valid literal member.
    assert parse_json(None, annotation) is None


# ---------------------------------------------------------------------------
# Union / Optional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["hello", 5])
def test_union_each_member_accepted(value: Any) -> None:
    assert parse_json(value, Union[str, int]) == value


def test_union_pep604_syntax() -> None:
    """The ``X | Y`` syntax (``types.UnionType``) is handled too."""
    assert parse_json("hello", str | int) == "hello"
    assert parse_json(5, str | int) == 5


def test_union_no_member_matches_raises() -> None:
    with pytest.raises(ValueError, match="Expected a value matching one of"):
        parse_json(1.5, Union[str, int])


def test_union_error_lists_each_member() -> None:
    with pytest.raises(ValueError, match="Tried each union member"):
        parse_json([], Union[str, int])


@pytest.mark.parametrize("value", [None, "x"])
def test_optional_accepts_none_and_value(value: Any) -> None:
    assert parse_json(value, Optional[str]) == value


def test_optional_rejects_other_types() -> None:
    with pytest.raises(ValueError, match="Expected a value matching one of"):
        parse_json(5, Optional[str])


def test_union_with_coercion_returns_coerced_value() -> None:
    """A union member that coerces (Sequence -> tuple) returns the coerced form."""
    result = parse_json([1, 2, 3], Union[str, Sequence[int]])
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# tuple[...]
# ---------------------------------------------------------------------------


def test_tuple_fixed_length_valid() -> None:
    result = parse_json([1, "a"], tuple[int, str])
    assert result == (1, "a")
    assert isinstance(result, tuple)


def test_tuple_fixed_length_wrong_length_raises() -> None:
    with pytest.raises(TypeError, match="Expected a sequence of length 2"):
        parse_json([1], tuple[int, str])
    with pytest.raises(TypeError, match="Expected a sequence of length 2"):
        parse_json([1, "a", "extra"], tuple[int, str])


def test_tuple_fixed_length_wrong_element_type_raises() -> None:
    # Second element should be str but is an int -> inner primitive ValueError.
    with pytest.raises(ValueError, match="Expected str"):
        parse_json([1, 2], tuple[int, str])


def test_tuple_variadic_valid() -> None:
    result = parse_json([1, 2, 3], tuple[int, ...])
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_tuple_variadic_empty() -> None:
    assert parse_json([], tuple[int, ...]) == ()


def test_tuple_variadic_wrong_element_raises() -> None:
    with pytest.raises(ValueError, match="Expected int"):
        parse_json([1, "x", 3], tuple[int, ...])


def test_tuple_accepts_tuple_input() -> None:
    assert parse_json((1, "a"), tuple[int, str]) == (1, "a")


def test_tuple_rejects_non_sequence() -> None:
    with pytest.raises(TypeError, match="Expected a sequence"):
        parse_json(5, tuple[int, ...])


def test_tuple_rejects_str_input() -> None:
    """A ``str`` is not treated as a sequence for tuple parsing."""
    with pytest.raises(TypeError, match="Expected a sequence"):
        parse_json("ab", tuple[str, str])


def test_tuple_empty_annotation() -> None:
    """``tuple[()]`` matches the empty sequence."""
    assert parse_json([], tuple[()]) == ()


# ---------------------------------------------------------------------------
# Sequence[T] / list[T]  (coerced to tuple)
# ---------------------------------------------------------------------------


def test_sequence_coerces_to_tuple() -> None:
    result = parse_json([1, 2, 3], Sequence[int])
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_list_coerces_to_tuple() -> None:
    result = parse_json([1, 2, 3], list[int])
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_sequence_validates_elements() -> None:
    with pytest.raises(ValueError, match="Expected int"):
        parse_json([1, "two", 3], Sequence[int])


def test_sequence_rejects_str() -> None:
    """A bare string is a primitive in JSON terms, not a sequence."""
    with pytest.raises(TypeError, match="Expected a sequence"):
        parse_json("abc", Sequence[str])


def test_sequence_rejects_non_sequence() -> None:
    with pytest.raises(TypeError, match="Expected a sequence"):
        parse_json(42, Sequence[int])


def test_sequence_of_sequence_nested() -> None:
    result = parse_json([[1, 2], [3]], Sequence[Sequence[int]])
    assert result == ((1, 2), (3,))
    assert isinstance(result, tuple)
    assert all(isinstance(inner, tuple) for inner in result)


# ---------------------------------------------------------------------------
# Mapping[str, T] / dict[str, T]
# ---------------------------------------------------------------------------


def test_mapping_valid_returns_dict() -> None:
    result = parse_json({"a": 1, "b": 2}, Mapping[str, int])
    assert result == {"a": 1, "b": 2}
    assert isinstance(result, dict)


def test_dict_valid_returns_dict() -> None:
    result = parse_json({"a": 1}, dict[str, int])
    assert result == {"a": 1}
    assert isinstance(result, dict)


def test_mapping_wrong_value_type_raises() -> None:
    with pytest.raises(ValueError, match="Expected int"):
        parse_json({"a": "not an int"}, Mapping[str, int])


def test_mapping_non_str_key_raises() -> None:
    with pytest.raises(TypeError, match="Expected mapping key to be str"):
        parse_json({1: 1}, Mapping[str, int])


def test_mapping_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="Expected a mapping"):
        parse_json([("a", 1)], Mapping[str, int])


def test_mapping_nested_values() -> None:
    result = parse_json({"a": [1, 2], "b": [3]}, Mapping[str, Sequence[int]])
    assert result == {"a": (1, 2), "b": (3,)}
    assert isinstance(result, dict)
    assert all(isinstance(v, tuple) for v in result.values())


# ---------------------------------------------------------------------------
# TypedDict
# ---------------------------------------------------------------------------


class Point(TypedDict):
    x: int
    y: int


# NOTE: This module uses ``from __future__ import annotations``, which stringizes
# annotations. With the *class* TypedDict syntax, ``typing_extensions`` cannot see
# a ``NotRequired[...]`` wrapper at class-creation time (the annotation is a string),
# so the key is wrongly recorded as required. This is a known limitation of
# stringized annotations + ``NotRequired`` -- NOT a bug in ``json_parse``. The
# functional ``TypedDict(...)`` form keeps ``NotRequired`` as a real runtime object,
# so ``__optional_keys__`` is populated correctly; we use it here to genuinely
# exercise NotRequired. (``total=False``, used by ``PartialConfig`` below, is
# class-level and works correctly even with stringized annotations.)
Config = TypedDict("Config", {"name": str, "count": NotRequired[int]})


class PartialConfig(TypedDict, total=False):
    a: int
    b: str


class Outer(TypedDict):
    label: str
    point: Point


def test_typeddict_valid() -> None:
    result = parse_json({"x": 1, "y": 2}, Point)
    assert result == {"x": 1, "y": 2}
    assert isinstance(result, dict)


def test_typeddict_missing_required_key_raises() -> None:
    with pytest.raises(ValueError, match="Expected required key"):
        parse_json({"x": 1}, Point)


def test_typeddict_wrong_value_type_raises() -> None:
    with pytest.raises(ValueError, match="Expected int"):
        parse_json({"x": 1, "y": "two"}, Point)


def test_typeddict_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="Expected a mapping"):
        parse_json([1, 2], Point)


def test_typeddict_notrequired_present() -> None:
    result = parse_json({"name": "a", "count": 3}, Config)
    assert result == {"name": "a", "count": 3}


def test_typeddict_notrequired_absent() -> None:
    result = parse_json({"name": "a"}, Config)
    assert result == {"name": "a"}


def test_typeddict_notrequired_wrong_type_raises() -> None:
    with pytest.raises(ValueError, match="Expected int"):
        parse_json({"name": "a", "count": "x"}, Config)


def test_typeddict_total_false_all_optional() -> None:
    # All keys optional: empty dict is valid.
    assert parse_json({}, PartialConfig) == {}
    assert parse_json({"a": 1}, PartialConfig) == {"a": 1}
    assert parse_json({"a": 1, "b": "y"}, PartialConfig) == {"a": 1, "b": "y"}


def test_typeddict_total_false_validates_present_keys() -> None:
    with pytest.raises(ValueError, match="Expected int"):
        parse_json({"a": "not int"}, PartialConfig)


def test_typeddict_nested_valid() -> None:
    result = parse_json({"label": "L", "point": {"x": 1, "y": 2}}, Outer)
    assert result == {"label": "L", "point": {"x": 1, "y": 2}}


def test_typeddict_nested_invalid_recurses() -> None:
    with pytest.raises(ValueError, match="Expected int"):
        parse_json({"label": "L", "point": {"x": 1, "y": "bad"}}, Outer)


def test_typeddict_nested_missing_required_recurses() -> None:
    with pytest.raises(ValueError, match="Expected required key"):
        parse_json({"label": "L", "point": {"x": 1}}, Outer)


def test_typeddict_preserves_extra_keys() -> None:
    """Extra keys not declared on the TypedDict are preserved unchanged."""
    result = parse_json({"x": 1, "y": 2, "extra": "kept"}, Point)
    assert result == {"x": 1, "y": 2, "extra": "kept"}


# ---------------------------------------------------------------------------
# Fallback / error quality
# ---------------------------------------------------------------------------


def test_unsupported_annotation_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="unsupported type annotation"):
        parse_json(1, complex)


def test_unsupported_annotation_names_value_and_annotation() -> None:
    with pytest.raises(TypeError) as excinfo:
        parse_json(object(), set[int])
    msg = str(excinfo.value)
    assert "set" in msg


def test_error_message_names_offending_value() -> None:
    """Primitive errors name the offending value via ``repr``."""
    with pytest.raises(ValueError, match=r"Expected int, got 'nope' instead"):
        parse_json("nope", int)


def test_error_message_names_expected_literal_choices() -> None:
    with pytest.raises(ValueError, match=r"Expected one of \('C', 'F'\)"):
        parse_json("Q", Literal["C", "F"])
