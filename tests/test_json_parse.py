"""Tests for :mod:`zarr.core.json_parse`.

``convert`` delegates JSON type coercion to :func:`msgspec.convert` (translating
``msgspec.ValidationError`` into ``ValueError``). The final group covers the
``parse_storage_transformers`` generator-exhaustion regression.
"""

from __future__ import annotations

from typing import Literal

import pytest

from zarr.core.json_parse import convert, parse_field
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
