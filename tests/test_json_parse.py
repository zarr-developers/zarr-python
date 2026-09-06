"""Tests for :mod:`zarr.core.json_parse`.

``convert`` delegates JSON type coercion to :func:`msgspec.convert` (translating
``msgspec.ValidationError`` into ``TypeError``); ``validate_json_value`` is the
hand-written fallback for the recursive ``JSON`` alias msgspec cannot build,
including a nesting-depth limit. The final group is a regression test for the
``parse_storage_transformers`` fix that motivated the depth limit work.
"""

from __future__ import annotations

from typing import Literal

import pytest

from zarr.core.json_parse import MAX_JSON_DEPTH, convert, parse_field, validate_json_value
from zarr.core.metadata.v3 import parse_storage_transformers


class TestConvert:
    def test_literal(self) -> None:
        assert convert(3, Literal[3]) == 3
        assert convert("array", Literal["array", "group"]) == "array"

    def test_literal_rejects_non_member(self) -> None:
        with pytest.raises(ValueError, match="Expected instance of"):
            convert(4, Literal[3])
        with pytest.raises(ValueError, match="Expected instance of"):
            convert("Q", Literal["C", "F"])

    def test_sequence_coerced_to_tuple(self) -> None:
        assert convert([1, 2, 3], tuple[int, ...]) == (1, 2, 3)
        assert convert([1, 2], tuple[int, int]) == (1, 2)

    def test_int(self) -> None:
        assert convert(5, int) == 5

    def test_bool_int_strictness(self) -> None:
        # bool is an int subclass, but the two must not be interchangeable.
        with pytest.raises(ValueError):
            convert(True, int)
        with pytest.raises(ValueError):
            convert(1, bool)
        # ... and True must not satisfy Literal[1].
        with pytest.raises(ValueError):
            convert(True, Literal[1])


class TestParseField:
    def test_valid_passthrough(self) -> None:
        assert parse_field(3, Literal[3], "zarr_format") == 3

    def test_wraps_with_field_context(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse input for 'zarr_format'"):
            parse_field(4, Literal[3], "zarr_format")

    def test_custom_error_type_and_chaining(self) -> None:
        class MyError(ValueError):
            pass

        with pytest.raises(MyError, match="Failed to parse input for 'node_type'") as exc_info:
            parse_field(5, Literal["array"], "node_type", error=MyError)
        # the generic type error is chained as the cause
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestValidateJsonValue:
    @pytest.mark.parametrize("value", [None, True, 1, 1.5, "s"])
    def test_primitives(self, value: object) -> None:
        assert validate_json_value(value) is value

    def test_nested(self) -> None:
        value = {"a": [1, 2.0, "x", True, None], "b": {"c": [{}]}}
        assert validate_json_value(value) is value

    def test_rejects_non_str_keys(self) -> None:
        with pytest.raises(TypeError, match="keys must be str"):
            validate_json_value({1: "x"})

    def test_rejects_non_json_leaf(self) -> None:
        with pytest.raises(TypeError, match="not a valid JSON value"):
            validate_json_value(object())
        with pytest.raises(TypeError, match="not a valid JSON value"):
            validate_json_value({"a": object()})

    def test_depth_limit(self) -> None:
        def nest(depth: int) -> object:
            v: object = "leaf"
            for _ in range(depth):
                v = {"k": v}
            return v

        # At the limit it passes; one level deeper it is rejected. This bound is
        # new behavior the previous per-field parsers never had.
        assert validate_json_value(nest(MAX_JSON_DEPTH)) is not None
        with pytest.raises(ValueError, match="maximum depth"):
            validate_json_value(nest(MAX_JSON_DEPTH + 1))


class TestStorageTransformersRegression:
    """`parse_storage_transformers` used to call `len(tuple(data))` and then
    return `data` itself, exhausting a one-shot iterable and returning a value
    typed as a tuple but not actually a tuple."""

    def test_none(self) -> None:
        assert parse_storage_transformers(None) == ()

    def test_empty(self) -> None:
        assert parse_storage_transformers([]) == ()

    def test_list_returns_tuple(self) -> None:
        result = parse_storage_transformers([{"a": 1}])
        assert result == ({"a": 1},)
        assert isinstance(result, tuple)

    def test_generator_not_exhausted(self) -> None:
        result = parse_storage_transformers(iter([{"a": 1}, {"b": 2}]))
        assert result == ({"a": 1}, {"b": 2})

    def test_non_iterable_rejected(self) -> None:
        with pytest.raises(TypeError, match="Expected an iterable"):
            parse_storage_transformers(5)
