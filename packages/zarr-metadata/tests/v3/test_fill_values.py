"""What each data type accepts as a fill value.

The knowledge used to live in a table keyed by data type name; now each
type answers for itself, so these exercise the families directly rather
than through a document.
"""

from __future__ import annotations

import pytest

from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS
from zarr_metadata.v3.entity import DataTypeEntity, resolve

# (data type metadata, a fill value it accepts)
ACCEPTED: dict[str, tuple[object, object]] = {
    "bool": ("bool", True),
    "int8-low": ("int8", -128),
    "uint64-high": ("uint64", 2**64 - 1),
    "float64-number": ("float64", 1.5),
    "float64-integer": ("float64", 1),
    "float32-special": ("float32", "NaN"),
    "float16-hex": ("float16", "0x3e00"),
    "complex64-pair": ("complex64", (1.0, "-Infinity")),
    "string": ("string", "hello"),
    "bytes-base64": ("bytes", "aGk="),
    "bytes-array": ("bytes", (1, 2, 3)),
    "raw-exact-width": ("r16", (0, 255)),
    "time-integer": (
        {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}},
        -1,
    ),
    "time-nat": (
        {"name": "numpy.timedelta64", "configuration": {"unit": "s", "scale_factor": 1}},
        "NaT",
    ),
    "struct": (
        {"name": "struct", "configuration": {"fields": ({"name": "a", "data_type": "uint8"},)}},
        {"a": 7},
    ),
}

# (data type metadata, a fill value it rejects, part of the reason)
REJECTED: dict[str, tuple[object, object, str]] = {
    "bool-integer": ("bool", 1, "expected a boolean"),
    "int8-above-range": ("int8", 128, "[-128, 127]"),
    "uint8-negative": ("uint8", -1, "[0, 255]"),
    "float64-not-a-number": ("float64", "nope", "hex string"),
    "float64-boolean": ("float64", True, "expected a number or string"),
    "complex64-single": ("complex64", 1.0, "[real, imag] pair"),
    "complex64-bad-part": ("complex64", (1.0, "nope"), "invalid component"),
    "string-integer": ("string", 3, "expected a string"),
    "bytes-bad-base64": ("bytes", "!!", "standard-alphabet base64"),
    "raw-wrong-width": ("r16", (1, 2, 3), "expected 2 byte values"),
    "raw-byte-out-of-range": ("r8", (256,), "[0, 255]"),
    "time-not-integer": (
        {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}},
        1.5,
        "signed 64-bit integer or 'NaT'",
    ),
    "struct-missing-field": (
        {"name": "struct", "configuration": {"fields": ({"name": "a", "data_type": "uint8"},)}},
        {},
        "missing fill value",
    ),
    "struct-unknown-field": (
        {"name": "struct", "configuration": {"fields": ({"name": "a", "data_type": "uint8"},)}},
        {"a": 1, "b": 2},
        "unknown struct fill field",
    ),
}


def _data_type(metadata: object) -> DataTypeEntity:
    entity, problems = resolve(metadata, DataTypeEntity, CORE_AND_EXTENSIONS)
    assert problems == (), problems
    assert isinstance(entity, DataTypeEntity), metadata
    return entity


@pytest.mark.parametrize(("metadata", "fill"), ACCEPTED.values(), ids=list(ACCEPTED))
def test_accepts(metadata: object, fill: object) -> None:
    assert _data_type(metadata).fill_value_problems(fill) == ()


@pytest.mark.parametrize(("metadata", "fill", "reason"), REJECTED.values(), ids=list(REJECTED))
def test_error_rejects(metadata: object, fill: object, reason: str) -> None:
    problems = _data_type(metadata).fill_value_problems(fill)
    assert problems, f"expected {fill!r} to be rejected"
    assert any(reason in problem.message for problem in problems), problems


def test_error_a_malformed_raw_name_has_no_entity_to_ask() -> None:
    # `r12` is not a width, so the data type does not exist and there is
    # nothing to put a fill value to.
    entity, problems = resolve("r12", DataTypeEntity, CORE_AND_EXTENSIONS)
    assert not isinstance(entity, DataTypeEntity)
    assert [problem.message for problem in problems] == [
        "Expected 'r<N>' where N is a positive multiple of 8, got 'r12'"
    ]


def test_an_unmodelled_data_type_judges_nothing() -> None:
    # Extension openness: a fill value we cannot interpret is not wrong.
    assert CORE_AND_EXTENSIONS.claimant(DataTypeEntity, "mycorp.decimal") is None
